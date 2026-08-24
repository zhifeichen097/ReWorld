# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from ..modules.action_model import pm_rope_delta_apply, _POSE_DIM, _se3_block_apply
from ..modules import action_model as _am  # runtime switches (_PROPE_LOG_T etc.)
from ..modules.causal_model import slice_per_head_local_attn_for_sp_rank
from .ulysses import distributed_attention, distributed_flex_attention
from .util import gather_forward, get_rank, get_world_size
import math
from utils.debug_option import DEBUG


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def sp_rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        sp_size = get_world_size()
        f = f * sp_size
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        start_idx = sp_rank * s_per_rank
        end_idx = (sp_rank + 1) * s_per_rank
        freqs_i_rank = freqs_i[start_idx:end_idx, :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        if DEBUG and i == 0:
            print(f"[sp_rope_apply] sp_rank={sp_rank}, f={f}, h={h}, w={w}, "
                f"freqs_i shape={freqs_i.shape}, start_idx={start_idx}, end_idx={end_idx}, "
                f"freqs_i_rank[0]={freqs_i_rank[0, 0, :3]}, "
                f"freqs_i_rank[-1]={freqs_i_rank[-1, 0, :3]}")

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = sp_rope_apply(q, grid_sizes, freqs)
    k = sp_rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x




def sp_dit_causal_forward_train(
    self,
    x,
    t,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    clip_fea=None,
    y=None,
    combined_action_ids=None,
    prope_context=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    # print("forward train")
    # raise NotImplementedError()
    if DEBUG:
        print("SP forward train is called")

    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # Construct blockwise causal attn mask (the frame dim is split across SP; total frames = per-rank frames * sp_size)
    sp_size = get_world_size()

    # [BLOCKER-2 fix] Ulysses all_to_all scatters heads in CONTIGUOUS slices
    # (ulysses.py: scatter_dim=2 → rank r owns heads [r*Nl, (r+1)*Nl)). Every mask
    # built here is consumed by distributed_flex_attention on the rank-LOCAL head
    # slice, so a per-head window list must be sliced to this rank's heads FIRST
    # (otherwise every rank silently uses rows 0..Nl-1 of the GLOBAL list).
    # Composes with BLOCKER-1: head groups are then computed on the LOCAL slice.
    # Backward-compat gate inside the helper: scalar / uniform list (e.g. 24×-1)
    # is returned unchanged → original code path, byte-identical.
    # [ablation] re-sample which heads are global(-1)/local THIS train step BEFORE the
    # per-rank slice. The permutation is identical on every SP rank (lockstep-seeded
    # CPU generator), so the rank-local head slice below stays globally coherent.
    self._maybe_randomize_routing()
    local_attn_size_rank = slice_per_head_local_attn_for_sp_rank(
        getattr(self, "local_attn_size", -1), get_rank(), sp_size)

    # chunk_drop / landmark training augmentation under SP — MUST be applied here:
    # this SP forward is what runs when sequence_parallel_size>1, and it previously
    # had NO chunk_drop logic, so any drop mask was a SILENT no-op (A1b bug). We build
    # a fresh per-step teacher-forcing mask with the dropped columns, mirroring the
    # non-SP path exactly. The global chunk count uses the SAME num_frames=x.shape[2]*
    # sp_size the cached mask uses, and prope_context is broadcast FULL across the SP
    # group (see _sync_batch_for_sequence_parallel) so global per-chunk poses are
    # directly available. Do NOT cache (self.block_mask stays the clean mask for eval).
    chunk_drop_active = (
        torch.is_grad_enabled()
        and isinstance(self.chunk_drop_config, dict)
        and self.chunk_drop_config.get("enabled", False)
        and clean_x is not None
        and not self.independent_first_frame
    )
    block_mask_step = None
    if chunk_drop_active:
        frame_seqlen_sp = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        num_chunks_total = math.ceil((x.shape[2] * sp_size) / self.num_frame_per_block)
        # prope_context is sharded along time (dim=1) across the SP group by the trainer
        # (_chunk_tensor: contiguous chunk[sp_rank]), so this rank only holds 1/sp_size of
        # the frames. Gather back to the FULL global sequence so landmark/pose selection
        # sees ALL chunks. all_gather is over the SP group (set_sequence_parallel_group),
        # in rank order, so concat(dim=1) reconstructs the original ordering. The gate is
        # data-independent (identical across ranks), so every rank enters this collective.
        # After the gather every rank in the group holds the IDENTICAL full sequence ⇒ the
        # deterministic landmark sampler (sync=False) yields an identical mask within the
        # group (consistent for all_to_all) and a correctly-different one across DP groups.
        prope_full = (gather_forward(prope_context, dim=1)
                      if (prope_context is not None and sp_size > 1) else prope_context)
        dropped_chunks_tensor, _mode = self._sample_chunk_drop_for_step(
            num_chunks_total=num_chunks_total,
            num_frame_per_block=self.num_frame_per_block,
            prope_context=prope_full,
            device=device,
        )
        block_mask_step = self._prepare_teacher_forcing_mask(
            device, num_frames=x.shape[2] * sp_size,
            frame_seqlen=frame_seqlen_sp,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=local_attn_size_rank,  # BLOCKER-2: rank-local head slice
            dropped_chunks_tensor=dropped_chunks_tensor,
        )
    elif self.block_mask is None:
        if clean_x is not None:
            if self.independent_first_frame:
                raise NotImplementedError()
            else:
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=local_attn_size_rank,  # BLOCKER-2: rank-local head slice
                )
        else:
            if self.independent_first_frame:
                self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=local_attn_size_rank  # BLOCKER-2: rank-local head slice
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=local_attn_size_rank  # BLOCKER-2: rank-local head slice
                )

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    # SP fix: RoPE needs global frame count; scale up now, restore before unpatchify
    # if sp_size > 1:
    #     grid_sizes[:, 0] = grid_sizes[:, 0] * sp_size
    x = [u.flatten(2).transpose(1, 2) for u in x]

    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    max_len = int(seq_lens.max().item())
    assert max_len > 0, "Token sequence length is zero after patch embedding"
    # assert max_len <= seq_len, f"seq_len overflow: max_len={max_len} > seq_len={seq_len}"
    # Pad all samples to the batch max length instead of the first sample length
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        # t = t.expand(t.size(0), seq_len)
        raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")
    # with torch.amp.autocast('cuda', dtype=torch.float32):
    bt = t.size(0)
    t_len = t.size(1)
    t_ori_shape = t.shape
    t = t.flatten()
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, t_len)).type_as(x))
    e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C
    # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clean_x is not None:
        clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
        clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

        seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
        # assert seq_lens_clean.max() <= seq_len
        clean_x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
        ])

        x = torch.cat([clean_x, x], dim=1)
        if aug_t is None:
            aug_t = torch.zeros(t_ori_shape, device=t.device, dtype=t.dtype)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        bt_clean = aug_t.size(0)
        t_clean_len = aug_t.size(1)
        aug_t = aug_t.flatten()
        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t).unflatten(0, (bt_clean, t_clean_len)).type_as(x))
        e0_clean = self.time_projection(e_clean).unflatten(2, (6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)


    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=(block_mask_step if chunk_drop_active else self.block_mask),
        combined_action_ids=combined_action_ids,
        teacher_forcing=(clean_x is not None),
        prope_context=prope_context,
    )

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            return module(*inputs, **kwargs)
        return custom_forward

    for block in self.blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, **kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **kwargs)

    if clean_x is not None:
        x = x[:, x.shape[1] // 2:]

    # head
    x = self.head(x, e.unsqueeze(2))

    # # Context Parallel
    # x = gather_forward_with_grad(x, dim=1)

    # unpatchify — restore local frame count for correct reshape
    # if sp_size > 1:
    #     grid_sizes[:, 0] = grid_sizes[:, 0] // sp_size
    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)



def sp_causal_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
    prope_context=None,
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if cache_start is None:
        cache_start = current_start

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    is_pm_rope = hasattr(self, 'pose_mlp') and prope_context is not None

    if kv_cache is None:
        is_tf = (s == seq_lens[0].item() * 2)

        if is_pm_rope:
            # ── CC-full path ──────────────────────────────────────────────
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()

            # QK phase offsets
            R_flat = prope_context[..., :3, :3].flatten(-2)
            t_vec  = prope_context[..., :3, 3]
            if _am._PROPE_LOG_T:   # [metric-prope] keep in lockstep with action_model.py
                t_vec = torch.sign(t_vec) * torch.log1p(t_vec.abs())
            c_all  = torch.cat([R_flat, t_vec], dim=-1)

            T_half = s // (2 * frame_seqlen) if is_tf else s // frame_seqlen
            c      = c_all[:, -T_half:]
            delta  = self.pose_mlp(c)
            delta_tok = (delta.unsqueeze(2)
                               .expand(-1, -1, frame_seqlen, -1)
                               .reshape(b, T_half * frame_seqlen, d // 2)
                               .unsqueeze(2))
            if is_tf:
                delta_tok = torch.cat([delta_tok, delta_tok], dim=1)

            # V-side P_inv residual
            if is_tf:
                half = s // 2
                v_p = torch.cat([
                    _se3_block_apply(v[:, :half], prope_context, 'P_inv', frame_seqlen),
                    _se3_block_apply(v[:, half:], prope_context, 'P_inv', frame_seqlen),
                ], dim=1)
            else:
                v_p = _se3_block_apply(v, prope_context, 'P_inv', frame_seqlen)
            v = (v.flatten(2) + self.value_proj(v_p.flatten(2))).view(b, s, n, d)

            # SP-aware RoPE + δ
            def rope_and_delta(q_in, k_in, dt):
                rq = sp_rope_apply(q_in, grid_sizes, freqs).type_as(v)
                rk = sp_rope_apply(k_in, grid_sizes, freqs).type_as(v)
                return pm_rope_delta_apply(rq, dt), pm_rope_delta_apply(rk, dt)

            if is_tf:
                half = s // 2
                rq0, rk0 = rope_and_delta(q[:, :half], k[:, :half], delta_tok[:, :half])
                rq1, rk1 = rope_and_delta(q[:, half:], k[:, half:], delta_tok[:, half:])
                roped_query = torch.cat([rq0, rq1], dim=1)
                roped_key   = torch.cat([rk0, rk1], dim=1)
            else:
                roped_query, roped_key = rope_and_delta(q, k, delta_tok)

            x = distributed_flex_attention(roped_query, roped_key, v, block_mask, is_tf=is_tf)

            # O-side P residual
            out = self.o(x.flatten(2))
            if is_tf:
                half = x.shape[1] // 2
                x_p = torch.cat([
                    _se3_block_apply(x[:, :half], prope_context, 'P', frame_seqlen),
                    _se3_block_apply(x[:, half:], prope_context, 'P', frame_seqlen),
                ], dim=1)
            else:
                x_p = _se3_block_apply(x, prope_context, 'P', frame_seqlen)
            return out + self.prope_proj(x_p.flatten(2))

        else:
            # ── Standard SP-RoPE path ─────────────────────────────────────
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = torch.cat([
                    sp_rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    for ii in range(2)
                ], dim=1)
                roped_key = torch.cat([
                    sp_rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    for ii in range(2)
                ], dim=1)
            else:
                roped_query = sp_rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key   = sp_rope_apply(k, grid_sizes, freqs).type_as(v)

            x = distributed_flex_attention(roped_query, roped_key, v, block_mask, is_tf=is_tf)
            x = x.flatten(2)
            return self.o(x)

    else:
        raise NotImplementedError()
