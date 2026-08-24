# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# NOTE: dropped `from transformers.models.x_clip.modeling_x_clip import x_clip_loss`
# — dead import (never referenced) and removed in transformers 5.x.
from wan_5b.modules.attention import attention
from wan_5b.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WanCrossAttention,
    rope_params,
    sinusoidal_embedding_1d,
    WanCrossAttention,
    flash_attention
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
# With long sequences + chunk_drop (e.g. T=96, TF mask S=168,960 / 1320 blocks) the
# inductor may pick XBLOCK=8192 for a slicing kernel, exceeding its conservative
# 4096 heuristic cap and raising a compile-time AssertionError ('XBLOCK' too large).
# The cap is a heuristic, not a hardware limit: 8192/block is legal for pointwise
# kernels, so we only raise the X dimension here.
# NOTE: must run with TORCHINDUCTOR_COMPILE_THREADS=1 — the default subprocess
# compile pool does not inherit this patch, and the assert still fires in workers.
from torch._inductor.runtime import hints as _inductor_hints
if _inductor_hints.TRITON_MAX_BLOCK.get("X", 0) < 8192:
    _inductor_hints.TRITON_MAX_BLOCK["X"] = 8192
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

from utils.debug_option import DEBUG

# wan 5b model compilation for flexattention
flex_attention = torch.compile(
    flex_attention, dynamic=False)


# =====================================================================
# [BLOCKER-1 fix] Per-head MIXED local_attn_size under chunk_drop.
#
# Problem: when chunk_drop is active the mask builders force H=None (to avoid
# the H=24 dense-mask OOM, ~159GB), so the BlockMask STRUCTURE is evaluated at
# h=0 only — a mixed per-head window list (e.g. lineB_routing's 18×(+12) +
# 6×(-1)) silently collapses to list[0]'s window for every head.
#
# Fix (2-group scheme): partition heads by DISTINCT window value, build ONE
# H=None BlockMask per group (each group's window is a scalar inside its own
# closure → structure is exact, and H stays None so the OOM guarantee that
# motivated the original clobber is preserved — we NEVER build an H=num_heads
# mask), then run flex_attention once per group on that group's head slice of
# q/k/v and stitch the outputs back in head order.
#
# BACKWARD-COMPAT GATE (load-bearing): the grouped path only triggers when the
# list has >= 2 DISTINCT window values. A uniform list (e.g. memexp/prope_only
# 24×-1) or a scalar takes the ORIGINAL single-mask code path byte-identically.
# =====================================================================
class PerHeadGroupedBlockMask:
    """Container for per-head-group BlockMasks (mixed local_attn_size + chunk_drop).

    groups: list of (head_indices, BlockMask) pairs. head_indices index the head
    dim of the q/k/v this mask will be applied to (i.e. LOCAL head indices when
    built from an SP-rank-sliced window list). Each BlockMask is built with
    H=None (shared across that group's heads).
    """

    def __init__(self, groups, num_heads):
        self.groups = groups
        self.num_heads = num_heads

    def __repr__(self):
        desc = ", ".join(f"heads{ids}" for ids, _ in self.groups)
        return f"PerHeadGroupedBlockMask(num_heads={self.num_heads}, groups=[{desc}])"


def has_mixed_per_head_windows(local_attn_size):
    """True iff local_attn_size is a per-head list with >= 2 distinct values."""
    return (
        isinstance(local_attn_size, (list, tuple))
        and len({int(s) for s in local_attn_size}) >= 2
    )


def partition_heads_by_window(local_attn_size):
    """[(window_value, [head indices with that value]), ...] sorted by value."""
    vals = [int(s) for s in local_attn_size]
    return [
        (w, [h for h, s in enumerate(vals) if s == w])
        for w in sorted(set(vals))
    ]


def slice_per_head_local_attn_for_sp_rank(local_attn_size, sp_rank, sp_size):
    """[BLOCKER-2 fix] Slice a per-head local_attn_size list to one SP rank's heads.

    Ulysses all_to_all (scatter_dim=2) gives each SP rank a CONTIGUOUS head
    slice (rank r → heads [r*Nl, (r+1)*Nl), Nl = num_heads // sp_size), but the
    SP mask builders historically used the full per-head list → every rank's
    mask rows corresponded to heads 0..Nl-1 of the GLOBAL list (rank1 silently
    wrong). Masks must therefore be built from the LOCAL slice of the list.

    Backward-compat gate: returns local_attn_size UNCHANGED (same object) for
    scalars, uniform lists, or sp_size <= 1 — the original code path is taken.
    """
    if sp_size is None or sp_size <= 1:
        return local_attn_size
    if not has_mixed_per_head_windows(local_attn_size):
        return local_attn_size
    num_heads = len(local_attn_size)
    assert num_heads % sp_size == 0, (
        f"per-head local_attn_size len {num_heads} not divisible by sp_size {sp_size}"
    )
    n_local = num_heads // sp_size
    return [int(s) for s in local_attn_size[sp_rank * n_local:(sp_rank + 1) * n_local]]


def run_flex_attention_with_grouped_mask(flex_fn, query, key, value, block_mask):
    """Dispatch flex attention for a plain BlockMask or a PerHeadGroupedBlockMask.

    query/key/value: [B, N, L, D] (head dim = 1, i.e. already transposed for
    flex_attention). For a grouped mask, runs flex_fn once per head group on
    that group's head slice and stitches outputs back in original head order.
    For a plain BlockMask this is byte-identical to calling flex_fn directly.
    """
    if not isinstance(block_mask, PerHeadGroupedBlockMask):
        return flex_fn(query=query, key=key, value=value, block_mask=block_mask)

    assert query.shape[1] == block_mask.num_heads, (
        f"grouped mask num_heads {block_mask.num_heads} != q heads {query.shape[1]} "
        "(SP: mask must be built from the rank-LOCAL head slice of the window list)"
    )
    outs = []
    order = []
    for head_ids, group_mask in block_mask.groups:
        outs.append(flex_fn(
            query=query[:, head_ids],
            key=key[:, head_ids],
            value=value[:, head_ids],
            block_mask=group_mask,
        ))
        order.extend(head_ids)
    out = torch.cat(outs, dim=1)
    if order != list(range(block_mask.num_heads)):
        inv = torch.empty(len(order), dtype=torch.long, device=out.device)
        inv[torch.as_tensor(order, dtype=torch.long, device=out.device)] = torch.arange(
            len(order), dtype=torch.long, device=out.device)
        out = out[:, inv]
    return out


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class MultiShotT2VCrossAttention(WanCrossAttention):

    def forward(self, x, context, context_lens, is_teacher_forcing=False, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B * num_chunks, L2, C]
            context_lens(Tensor): Shape [B] or [B * num_chunks]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        # Original batch size (videos)
        b_orig, L1, C = x.size()
        n, d = self.num_heads, self.head_dim

        # Effective batch size for cross-attention (videos * chunks)
        b_ctx = context.size(0)
        assert b_ctx % b_orig == 0, f"context batch ({b_ctx}) must be a multiple of x batch ({b_orig})"
        num_chunks = b_ctx // b_orig

        # Prepare context_lens for [B * num_chunks] if needed
        if context_lens is not None and context_lens.numel() == b_orig:
            context_lens = context_lens.repeat_interleave(num_chunks)
        elif context_lens is not None:
            assert context_lens.numel() == b_ctx, \
                f"context_lens must have length {b_orig} or {b_ctx}, got {context_lens.numel()}"
        # Helper to run standard cross-attention on a given x_chunk of shape [B * num_chunks, L_chunk, C]
        def _cross_attend(x_chunk):
            b_eff, L_chunk, _ = x_chunk.size()

            # compute query, key, value
            q = self.norm_q(self.q(x_chunk)).view(b_eff, -1, n, d)

            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    crossattn_cache["is_init"] = True
                    k = self.norm_k(self.k(context)).view(b_eff, -1, n, d)
                    v = self.v(context).view(b_eff, -1, n, d)
                    crossattn_cache["k"] = k
                    crossattn_cache["v"] = v
                else:
                    k = crossattn_cache["k"]
                    v = crossattn_cache["v"]
            else:
                k = self.norm_k(self.k(context)).view(b_eff, -1, n, d)
                v = self.v(context).view(b_eff, -1, n, d)

            # compute attention
            x_attn = flash_attention(q, k, v, k_lens=context_lens)

            # output projection
            x_attn = x_attn.flatten(2)
            x_attn = self.o(x_attn)
            return x_attn

        if not is_teacher_forcing:
            # -------------------------------
            # Regular multi-shot: all tokens attend text, we just chunk along L1
            # x: [B, L1, C] -> [B * num_chunks, L1 / num_chunks, C]
            # -------------------------------
            assert L1 % num_chunks == 0, \
                f"L1 ({L1}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L1 // num_chunks

            x_chunked = x.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]

            # reshape back to [B, L1, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L1, C)
            return x_attn

        # -------------------------------
        # Teacher forcing:
        # x is typically [B, 2 * L_tf, C]: the first half is clean, the second half noisy.
        # Multi-shot cross-attention is applied to both the clean and the noisy half.
        # -------------------------------
        assert L1 % 2 == 0, f"In teacher-forcing mode, L1 ({L1}) should be even."
        half = L1 // 2
        x_clean = x[:, :half, :]       # [B, L_tf, C]
        x_noisy = x[:, half:, :]       # [B, L_tf, C]

        def _chunk_and_attend(x_part):
            L_part = x_part.size(1)
            assert L_part % num_chunks == 0, \
                f"Segment length ({L_part}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L_part // num_chunks

            # [B, L_part, C] -> [B * num_chunks, L_part / num_chunks, C]
            x_chunked = x_part.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L_part, C)
            return x_attn

        x_clean_attn = _chunk_and_attend(x_clean)
        x_noisy_attn = _chunk_and_attend(x_noisy)

        # Reassemble the full sequence: both clean and noisy use the cross-attn results
        x_out = torch.cat([x_clean_attn, x_noisy_attn], dim=1)  # [B, L1, C]
        return x_out


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Attention window (in tokens) depends on the runtime latent grid size (H, W).
        # We compute it in forward when slicing KV cache.
        self.max_attention_size = None

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # grouped-mask aware (BLOCKER-1): plain BlockMask → identical
                # single flex_attention call; PerHeadGroupedBlockMask → one call
                # per head group, outputs stitched back in head order.
                x = run_flex_attention_with_grouped_mask(
                    flex_attention,
                    padded_roped_query.transpose(2, 1),
                    padded_roped_key.transpose(2, 1),
                    padded_v.transpose(2, 1),
                    block_mask,
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # grouped-mask aware (BLOCKER-1), see teacher-forcing branch above.
                x = run_flex_attention_with_grouped_mask(
                    flex_attention,
                    padded_roped_query.transpose(2, 1),
                    padded_roped_key.transpose(2, 1),
                    padded_v.transpose(2, 1),
                    block_mask,
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            if isinstance(self.local_attn_size, (list, tuple)):
                finite_sizes = [abs(s) for s in self.local_attn_size if s != -1]
                cache_local_attn_size = max(finite_sizes) if finite_sizes else -1
            else:
                cache_local_attn_size = abs(self.local_attn_size) if self.local_attn_size != -1 else -1
            if cache_local_attn_size == -1:
                self.max_attention_size = None
            else:
                self.max_attention_size = int(cache_local_attn_size) * int(frame_seqlen)
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            # Store pre-RoPE k and v in cache
            if cache_local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v

            # Slice cached (pre-RoPE) k and v
            if self.max_attention_size is None:
                slice_start = 0
                k_slice = kv_cache["k"][:, :local_end_index]
                v_slice = kv_cache["v"][:, :local_end_index]
            else:
                slice_start = max(0, local_end_index - self.max_attention_size)
                k_slice = kv_cache["k"][:, slice_start:local_end_index]
                v_slice = kv_cache["v"][:, slice_start:local_end_index]

            # Apply RoPE to cached k_slice on-the-fly based on global positions
            # Cache layout: [sink frames (global 0..sink_size-1)] [local window (contiguous recent frames)]
            # Global start of local window can be derived: current_end - (local_end_index - sink_tokens)
            local_window_global_start = current_end - (local_end_index - sink_tokens)

            if sink_tokens == 0 or slice_start >= sink_tokens:
                global_offset = local_window_global_start + (slice_start - sink_tokens)
                slice_start_frame = global_offset // frame_seqlen
                slice_num_frames = k_slice.shape[1] // frame_seqlen
                slice_grid = grid_sizes.clone()
                slice_grid[:, 0] = slice_num_frames
                roped_k_slice = causal_rope_apply(
                    k_slice, slice_grid, freqs, start_frame=slice_start_frame).type_as(v)
            else:
                # Slice spans both sink region and local window (non-contiguous in global space)
                num_sink_in_slice = sink_tokens - slice_start
                sink_part = k_slice[:, :num_sink_in_slice]
                local_part = k_slice[:, num_sink_in_slice:]

                sink_start_frame = slice_start // frame_seqlen
                sink_num_frames = num_sink_in_slice // frame_seqlen
                sink_grid = grid_sizes.clone()
                sink_grid[:, 0] = sink_num_frames
                roped_sink = causal_rope_apply(
                    sink_part, sink_grid, freqs, start_frame=sink_start_frame).type_as(v)

                local_start_frame_val = local_window_global_start // frame_seqlen
                local_num_frames = local_part.shape[1] // frame_seqlen
                local_grid = grid_sizes.clone()
                local_grid[:, 0] = local_num_frames
                roped_local = causal_rope_apply(
                    local_part, local_grid, freqs, start_frame=local_start_frame_val).type_as(v)

                roped_k_slice = torch.cat([roped_sink, roped_local], dim=1)

            x = attention(roped_query, roped_k_slice, v_slice)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim,
                                            num_heads,
                                            (-1, -1),
                                            qk_norm,
                                            eps)

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        is_tf = (x.shape[1] == seq_lens[0].item() * 2)
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache, is_teacher_forcing=is_tf)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video with causal attention.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 rope_spatial_interp=False,
                 rope_spatial_ref_grid=None,
                 rope_spatial_cur_grid=None):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video), 'i2v' (image-to-video), or 'ti2v'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # OmegaConf ListConfig / any sequence-like -> plain Python list,
        # so downstream isinstance(x, (list, tuple)) checks just work.
        if local_attn_size is not None and not isinstance(local_attn_size, int):
            try:
                local_attn_size = [int(s) for s in local_attn_size]
            except TypeError:
                pass
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                  local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE spatial Position-Interpolation (480p->720p warmstart fix).
        # The H/W post-patch grid grows with resolution (480p 12x20 -> 720p 22x40), so at
        # higher res the spatial RoPE rows for indices the 480p weights NEVER trained on get
        # hit cold -> miscalibrated spatial binding -> needle re-id fails on look-back.
        # Remap H/W positions back onto the trained range by scaling the position axis
        # (Llama Position Interpolation): scale_h = ref_h/cur_h, scale_w = ref_w/cur_w.
        # TEMPORAL is NOT scaled (frame count identical across resolutions).
        sh = sw = 1.0
        if rope_spatial_interp and rope_spatial_ref_grid is not None and rope_spatial_cur_grid is not None:
            sh = float(rope_spatial_ref_grid[0]) / float(rope_spatial_cur_grid[0])
            sw = float(rope_spatial_ref_grid[1]) / float(rope_spatial_cur_grid[1])
            print(f"[RoPE-interp] spatial Position-Interpolation ON: "
                  f"ref_grid={list(rope_spatial_ref_grid)} cur_grid={list(rope_spatial_cur_grid)} "
                  f"-> scale_h={sh:.4f} scale_w={sw:.4f} (temporal unscaled)", flush=True)
        self.rope_spatial_scale = (sh, sw)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),                  # temporal (F) — unscaled
            rope_params(1024, 2 * (d // 6), pos_scale=sh),        # height  (H) — interpolated
            rope_params(1024, 2 * (d // 6), pos_scale=sw)         # width   (W) — interpolated
        ],
            dim=1)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

        # chunk_drop training augmentation config (set externally from wrapper).
        # When enabled, training-time forward will randomly drop a subset of past
        # context chunks from the K/V side of the attention mask. This trains the
        # model to be robust to "missing KV" so inference-time KV compression
        # (e.g. sink+recency) does not collapse. Format:
        #   {"enabled": bool, "prob": float, "protect_sink": int, "protect_recency": int}
        # See _sample_dropped_chunks_tensor and _forward_train override.
        self.chunk_drop_config = None

        # [head-routing ABLATION] when enabled, each TRAIN step re-samples WHICH
        # heads are global(-1) vs local(window), preserving the configured
        # global:local head budget of local_attn_size. Set from the wrapper.
        # Default off → byte-identical legacy behaviour.
        self.random_head_routing = False
        self._routing_step = 0
        self._routing_base = None

    def _set_gradient_checkpointing(self, *args, enable=None, value=False, **kwargs):
        # NOTE: accept BOTH diffusers call conventions —
        # old (<=0.31): _set_gradient_checkpointing(module, value=bool)
        # new (>=0.33): _set_gradient_checkpointing(enable=bool, gradient_checkpointing_func=...)
        # The forward path gates on the plain bool self.gradient_checkpointing
        # (see _forward_train), so the passed checkpointing func is ignored.
        self.gradient_checkpointing = value if enable is None else enable

    def _maybe_randomize_routing(self):
        """[ablation] Re-sample which attention heads are global(-1) vs local each
        TRAINING step, preserving the configured global:local head budget (e.g. the
        6×-1 / 18×12 routing list → still 6 global + 18 local, random head order).

        Off unless self.random_head_routing. Skipped under no_grad (eval/inference
        keep the fixed pattern). The permutation comes from a CPU Generator seeded
        by a lockstep step counter, so it is IDENTICAL on every SP rank — the
        per-rank head slice (slice_per_head_local_attn_for_sp_rank) therefore stays
        globally coherent. Called at the top of BOTH the non-SP `_forward_train`
        (action_model) and the SP `sp_dit_causal_forward_train`.
        """
        if not getattr(self, "random_head_routing", False) or not torch.is_grad_enabled():
            return
        base = getattr(self, "_routing_base", None)
        if base is None:
            las = getattr(self, "local_attn_size", -1)
            if not isinstance(las, (list, tuple)) or len({int(s) for s in las}) < 2:
                return  # scalar / uniform window → nothing to randomize
            base = self._routing_base = [int(s) for s in las]
        # FIXED POOL of K distinct routings, CYCLED every step (NO torch._dynamo.reset()).
        # Why: every distinct routing makes flex_attention compile + retain ~1-2GB GPU mem that
        # is NOT reclaimable while cached → UNBOUNDED distinct routings = OOM (every-step → step14;
        # N-block+reset only slowed it: reset leaves ~0.5GB/block residual → OOM ~step2400).
        # A fixed pool of K routings compiles EXACTLY K kernels over the first K distinct steps,
        # then every step REUSES one (idx = step % K) → zero new compiles forever → bounded GPU
        # mem (= K kernels), zero leak, while routing still changes EVERY step (cycles K patterns).
        # K = routing_pool_size; capped by GPU mem (each kernel + the ~26GB compile workspace during
        # build → ~12 fit on 80GB). Pool seeded per-member → identical on every SP rank.
        pool = getattr(self, "_routing_pool", None)
        if pool is None:
            K = max(1, int(getattr(self, "routing_pool_size", 12)))
            pool = []
            for k in range(K):
                g = torch.Generator(); g.manual_seed(0x5EED + k)
                perm = torch.randperm(len(base), generator=g).tolist()
                pool.append([base[i] for i in perm])
            self._routing_pool = pool
            # dynamo evicts compiled graphs past cache_size_limit (default 8) → cycled-out routings
            # would recompile & re-leak. Keep all K cached.
            try:
                import torch._dynamo as _dynamo
                if _dynamo.config.cache_size_limit < K + 4:
                    _dynamo.config.cache_size_limit = K + 4
            except Exception:
                pass
        step = int(self._routing_step)
        self._routing_step += 1
        idx = step % len(pool)
        self.local_attn_size = pool[idx]
        if step < len(pool) + 1:  # LIVE confirmation over the first full cycle (all ranks identical per step)
            try:
                import torch.distributed as _dist
                _rk = _dist.get_rank() if (_dist.is_available() and _dist.is_initialized()) else 0
            except Exception:
                _rk = 0
            _gh = [i for i, s in enumerate(self.local_attn_size) if s == -1]
            print(f"[randhead LIVE] rank={_rk} step={step} pool_idx={idx}/{len(pool)} global_heads={_gh}", flush=True)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=3, local_attn_size=-1
    ) -> BlockMask:

        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)
        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask


    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=1,
        local_attn_size: int | list[int] = -1,
        dropped_chunks_tensor: torch.Tensor | None = None,
    ) -> BlockMask:
        """
        Block-wise causal mask. Supports per-head local_attn_size when a list
        is provided (one entry per attention head).

        local_attn_size semantics:
        - -1: full causal history.
        -  N: only the latest N frames/chunks are visible.
        - -N: hide the latest N frames/chunks and keep older causal history visible.

        chunk_drop augmentation:
        - dropped_chunks_tensor: optional bool tensor of shape [num_chunks] where
          num_chunks = ceil(num_frames / num_frame_per_block). If provided, KV
          positions falling inside any chunk marked True are masked out (except
          for the diagonal q_idx == kv_idx, which stays for self-attention).
          When None (default), behaviour is identical to the original mask.
        - The closure captures dropped_chunks_tensor as a tensor; we keep
          _compile=True (dense materialisation OOMs at training scale).
        """
        # [BLOCKER-1 fix] MIXED per-head windows + chunk_drop: H=None below would
        # evaluate the BlockMask structure at h=0 only (per-head windows silently
        # collapsed to list[0]'s). Build one H=None mask PER DISTINCT window value
        # (scalar recursion → exact structure per group, OOM guarantee kept) and
        # return a grouped container; attention call sites run flex once per group.
        # Gate: uniform list / scalar → original single-mask path, byte-identical.
        if dropped_chunks_tensor is not None and has_mixed_per_head_windows(local_attn_size):
            groups = []
            for win, head_ids in partition_heads_by_window(local_attn_size):
                group_mask = CausalWanModel._prepare_blockwise_causal_attn_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=num_frame_per_block,
                    local_attn_size=win,  # scalar → legacy path, H=None
                    dropped_chunks_tensor=dropped_chunks_tensor,
                )
                groups.append((head_ids, group_mask))
            return PerHeadGroupedBlockMask(groups, num_heads=len(local_attn_size))

        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        block_size = frame_seqlen * num_frame_per_block

        per_head = isinstance(local_attn_size, (list, tuple))
        if per_head:
            local_attn_size_tensor = torch.tensor(
                local_attn_size, dtype=torch.int64, device=device
            )
        else:
            if local_attn_size != -1:
                local_window_frames_int = abs(local_attn_size)

        # chunk_drop: validate length matches actual chunk count
        has_chunk_drop = dropped_chunks_tensor is not None
        if has_chunk_drop:
            num_chunks_total = math.ceil(num_frames / num_frame_per_block)
            assert dropped_chunks_tensor.shape == (num_chunks_total,), (
                f"dropped_chunks_tensor shape {tuple(dropped_chunks_tensor.shape)} "
                f"!= expected ({num_chunks_total},)"
            )
            dropped_chunks_tensor = dropped_chunks_tensor.to(device=device, dtype=torch.bool)

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            current_block_end = ((q_idx // block_size) + 1) * block_size
            full_history_mask = is_real_q & (kv_idx < current_block_end)

            if per_head:
                size_h = local_attn_size_tensor[h]
                window_frames = torch.abs(size_h)
                left = current_block_end - window_frames * frame_seqlen
                local_history_mask = full_history_mask & (kv_idx >= left)
                inverse_local_mask = is_real_q & (kv_idx < left)
                clean_mask = torch.where(
                    size_h == -1,
                    full_history_mask,
                    torch.where(size_h < -1, inverse_local_mask, local_history_mask),
                )
            else:
                if local_attn_size == -1:
                    clean_mask = full_history_mask
                elif local_attn_size < -1:
                    left = current_block_end - local_window_frames_int * frame_seqlen
                    clean_mask = is_real_q & (kv_idx < left)
                else:
                    left = current_block_end - local_window_frames_int * frame_seqlen
                    clean_mask = full_history_mask & (kv_idx >= left)

            # chunk_drop: mask out KV positions inside dropped chunks.
            # Diagonal eye_mask below preserves self-attention so the dropped
            # token's own forward pass still has at least one key.
            if has_chunk_drop:
                kv_chunk = kv_idx // block_size
                # clamp to valid index (padded kv positions land at last chunk
                # but is_real_q already filters; for kv beyond total_length
                # we still get a valid index because dropped_chunks covers all
                # logical chunks)
                kv_chunk_clamped = torch.clamp(kv_chunk, max=num_chunks_total - 1)
                is_dropped = dropped_chunks_tensor[kv_chunk_clamped]
                clean_mask = clean_mask & (~is_dropped)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask

        # Key fixes:
        # (1) chunk_drop active + _compile=True -> torchinductor recompiles with exploding
        #     size_hints -> OOM. Use _compile=False when chunk_drop is active.
        # (2) chunk_drop active + H=num_heads (24) -> the mask fully expands along the head
        #     dim, 84480^2 x 24 x 1B ~= 159 GB -> runtime OOM. The chunk_drop pattern is
        #     identical across heads, so force H=None (mask shared across heads, 24x smaller).
        # (3) 720p: the dense (_compile=False) path materializes [S,S]; when S^2 > 2**31 the
        #     int32 indexing overflows -> "CUDA driver error: invalid argument". At 720p
        #     S~=84480 triggers this -> force _compile=True to use block-sparse masks.
        #     Smaller resolutions (S^2=5.3e8) are unchanged. See the matching fix in
        #     _prepare_teacher_forcing_mask.
        _S = total_length + padded_length
        _dense_int32_overflow = (_S * _S) > (2 ** 31 - 1)
        _use_compile = (not has_chunk_drop) or _dense_int32_overflow
        num_heads_for_mask = None if has_chunk_drop else (len(local_attn_size) if per_head else None)
        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=num_heads_for_mask,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=_use_compile,
            device=device,
        )

        import torch.distributed as dist
        if not has_chunk_drop and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)

        return block_mask


    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 31,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        local_attn_size: int | list[int] = -1,
        dropped_chunks_tensor: torch.Tensor | None = None,
    ):
        """
        Teacher-forcing block-wise mask. Sequence layout:
            [ clean (KV context, len=num_frames*frame_seqlen) |
              noisy (queries that denoise, same length) ]

        chunk_drop augmentation:
        - dropped_chunks_tensor: optional bool tensor of shape [num_chunks]
          (num_chunks = ceil(num_frames / num_frame_per_block)) indicating which
          chunks in the *clean* (KV) half are dropped. When chunk c is dropped,
          KV positions in clean range [c*block, (c+1)*block) are masked out for
          BOTH clean-to-clean and noisy-to-clean attention (except the diagonal
          eye_mask which preserves self-attention).
        - The noisy half itself is never dropped — noisy chunks still see their
          own C1 (same-block) tokens and other noisy positions are not part of
          the standard mask anyway.
        - When None (default), mask is identical to the original cached version.
        - dropped_chunks_tensor is closure-captured; we keep _compile=True
          because the dense materialisation path (_compile=False) OOMs at
          training sequence length. torch.compile may re-trace if it sees a new
          closure identity; in practice this means one trace at startup.
        """
        # [BLOCKER-1 fix] MIXED per-head windows + chunk_drop → 2-group scheme.
        # Same rationale as _prepare_blockwise_causal_attn_mask: H=None below
        # evaluates structure at h=0 only, clobbering per-head windows. Build one
        # H=None mask per distinct window value via scalar recursion (exact
        # structure per group; never an H=num_heads mask → OOM guarantee kept).
        # Gate: uniform list / scalar → original single-mask path, byte-identical.
        if dropped_chunks_tensor is not None and has_mixed_per_head_windows(local_attn_size):
            groups = []
            for win, head_ids in partition_heads_by_window(local_attn_size):
                group_mask = CausalWanModel._prepare_teacher_forcing_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=num_frame_per_block,
                    local_attn_size=win,  # scalar → legacy path, H=None
                    dropped_chunks_tensor=dropped_chunks_tensor,
                )
                groups.append((head_ids, group_mask))
            return PerHeadGroupedBlockMask(groups, num_heads=len(local_attn_size))

        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block

        # chunk_drop setup: validate length and move to device.
        # num_chunks_total is the number of chunks in the CLEAN half of the
        # sequence (which is also the same as the noisy half).
        has_chunk_drop = dropped_chunks_tensor is not None
        drop_is_2d = False
        if has_chunk_drop:
            num_chunks_total = math.ceil(num_frames / num_frame_per_block)
            assert dropped_chunks_tensor.shape in (
                (num_chunks_total,), (num_chunks_total, num_chunks_total)
            ), (
                f"dropped_chunks_tensor shape {tuple(dropped_chunks_tensor.shape)} "
                f"!= expected ({num_chunks_total},) [1D static] or "
                f"({num_chunks_total},{num_chunks_total}) [2D per-row consol]"
            )
            dropped_chunks_tensor = dropped_chunks_tensor.to(device=device, dtype=torch.bool)
            # 2D: row = query chunk → per-chunk history (consolidation novelty).
            # 1D: one drop-set for every query (static B1 / random_fixed).
            drop_is_2d = dropped_chunks_tensor.dim() == 2

        # local_attn_size: an int is shared by all heads; a list is indexed per head
        # (different heads get different local attention sizes).
        # -1 = full history; a positive value = most-recent window; a value < -1 masks the
        # most recent abs(size) window and keeps only the older history.
        # During compilation h is a tensor and cannot index a Python list, so lists are
        # converted to a tensor on the device first.
        per_head = isinstance(local_attn_size, (list, tuple))
        if per_head:
            local_attn_size_tensor = torch.tensor(
                local_attn_size, dtype=torch.int64, device=device
            )
            # represent "full history" as a sufficiently large chunk count to avoid a None in the closure (not traceable)
            max_chunk_window = (clean_ends // attention_block_size) + 1
            max_chunk_window_tensor = torch.tensor(
                max_chunk_window, dtype=torch.int64, device=device
            )
        else:
            # int: -1 means full history, otherwise convert to a chunk count
            if local_attn_size == -1:
                local_chunk_window_int = (clean_ends // attention_block_size) + 1
            else:
                local_chunk_window_int = max(1, abs(local_attn_size) // num_frame_per_block)

        # Core rule: use pure arithmetic on coordinates — never index into external tensors!
        def attention_mask(b, h, q_idx, kv_idx):
            # resolve local_chunk_window per head (scalar or tensor; always expressed as a window size, no None)
            if per_head:
                size_h = local_attn_size_tensor[h]
                local_chunk_window = torch.where(
                    size_h == -1,
                    max_chunk_window_tensor,
                    torch.clamp(torch.abs(size_h) // num_frame_per_block, min=1),
                )
                use_inverse_local = size_h < -1
                use_full_history = size_h == -1
            else:
                local_chunk_window = local_chunk_window_int
                use_inverse_local = local_attn_size < -1
                use_full_history = local_attn_size == -1

            # ==========================================
            # 1. mask logic for clean frames
            # ==========================================
            is_clean_q = q_idx < clean_ends

            # end position of the block containing the current token
            clean_block_idx = q_idx // attention_block_size
            current_clean_block_end = (clean_block_idx + 1) * attention_block_size

            # a positive window sees only the most recent local_chunk_window chunks; a negative window uses the same boundary to mask out the recent window.
            context_start_block = torch.clamp(
                clean_block_idx + 1 - local_chunk_window, min=0
            )
            clean_context_start = context_start_block * attention_block_size

            clean_full_mask = is_clean_q & (kv_idx < current_clean_block_end)
            clean_local_mask = (
                is_clean_q
                & (kv_idx < current_clean_block_end)
                & (kv_idx >= clean_context_start)
            )
            clean_inverse_mask = is_clean_q & (kv_idx < clean_context_start)
            if per_head:
                clean_mask = torch.where(
                    use_full_history,
                    clean_full_mask,
                    torch.where(use_inverse_local, clean_inverse_mask, clean_local_mask),
                )
            elif use_full_history:
                clean_mask = clean_full_mask
            elif use_inverse_local:
                clean_mask = clean_inverse_mask
            else:
                clean_mask = clean_local_mask

            # chunk_drop: precompute "is this kv inside a dropped clean chunk?".
            # Only applies when kv is in the clean half (kv_idx < clean_ends);
            # kv in the noisy half is never dropped.
            if has_chunk_drop:
                clean_kv_chunk = kv_idx // attention_block_size
                clean_kv_chunk_clamped = torch.clamp(
                    clean_kv_chunk, max=num_chunks_total - 1
                )
                kv_in_clean = kv_idx < clean_ends
                if drop_is_2d:
                    # query is a CLEAN chunk here → row = clean_block_idx.
                    q_clean_chunk = torch.clamp(clean_block_idx, max=num_chunks_total - 1)
                    clean_kv_is_dropped = (
                        kv_in_clean
                        & dropped_chunks_tensor[q_clean_chunk, clean_kv_chunk_clamped]
                    )
                else:
                    clean_kv_is_dropped = (
                        kv_in_clean & dropped_chunks_tensor[clean_kv_chunk_clamped]
                    )
                clean_mask = clean_mask & (~clean_kv_is_dropped)

            # ==========================================
            # 2. mask logic for noisy frames
            # ==========================================
            is_noisy_q = q_idx >= clean_ends

            noisy_rel_idx = q_idx - clean_ends
            block_index = noisy_rel_idx // attention_block_size

            # C1: noisy tokens within the same block
            noisy_block_start = clean_ends + (block_index * attention_block_size)
            noisy_block_end = noisy_block_start + attention_block_size
            C1 = (kv_idx >= noisy_block_start) & (kv_idx < noisy_block_end)

            # C2: clean (context) tokens from earlier blocks
            context_end_for_noisy = block_index * attention_block_size
            context_start_block_noise = torch.clamp(
                block_index - local_chunk_window, min=0
            )
            noise_context_start = context_start_block_noise * attention_block_size

            C2_full = kv_idx < context_end_for_noisy
            C2_local = (kv_idx < context_end_for_noisy) & (kv_idx >= noise_context_start)
            C2_inverse = kv_idx < noise_context_start
            if per_head:
                C2 = torch.where(
                    use_full_history,
                    C2_full,
                    torch.where(use_inverse_local, C2_inverse, C2_local),
                )
            elif use_full_history:
                C2 = C2_full
            elif use_inverse_local:
                C2 = C2_inverse
            else:
                C2 = C2_local
            # chunk_drop: also mask C2 (clean KV viewed from noisy queries).
            # C1 (noisy-to-noisy same block) is unaffected by design.
            if has_chunk_drop:
                if drop_is_2d:
                    # query is a NOISY chunk here → row = block_index. THIS is the
                    # loss-bearing denoising path, so it MUST use keep_i (per-chunk).
                    q_noisy_chunk = torch.clamp(block_index, max=num_chunks_total - 1)
                    noisy_kv_is_dropped = (
                        (kv_idx < clean_ends)
                        & dropped_chunks_tensor[q_noisy_chunk, clean_kv_chunk_clamped]
                    )
                    C2 = C2 & (~noisy_kv_is_dropped)
                else:
                    C2 = C2 & (~clean_kv_is_dropped)
            noise_mask = is_noisy_q & (C1 | C2)

            # ==========================================
            # 3. final combination
            # ==========================================
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        # Key fixes:
        # (1) chunk_drop active + _compile=True -> torchinductor recompiles with exploding
        #     size_hints -> OOM.
        # (2) chunk_drop active + H=num_heads (24) -> the mask fully expands along the head
        #     dim, 84480^2 x 24 x 1B ~= 159 GB -> OOM at step-0 runtime. chunk_drop is
        #     identical across heads, so force H=None for a 24x saving.
        # (3) 720p: the dense (_compile=False) path materializes an [S,S] bool tensor; when
        #     S^2 > 2**31 (the int32 indexing limit) the vmap kernel fails with
        #     "CUDA driver error: invalid argument". At 720p S~=84480 -> S^2~=7.1e9 > 2.1e9,
        #     an outright overflow. In that case the _compile=True Triton block-sparse path
        #     (never materializes dense; uses block/64-bit indexing) is REQUIRED.
        #     Smaller resolutions (S=23040, S^2=5.3e8 < 2**31) are unaffected and keep the
        #     historical _compile=False behavior (byte-identical).
        _S = total_length + padded_length
        _dense_int32_overflow = (_S * _S) > (2 ** 31 - 1)
        _use_compile = (not has_chunk_drop) or _dense_int32_overflow
        num_heads_for_mask = None if has_chunk_drop else (len(local_attn_size) if per_head else None)
        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=num_heads_for_mask,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=_use_compile,
            device=device,
        )

        import torch.distributed as dist
        if not has_chunk_drop and (not dist.is_initialized() or dist.get_rank() == 0):
            print(block_mask)

        return block_mask

    @staticmethod
    def _sample_dropped_chunks_tensor(
        num_chunks: int,
        prob: float,
        protect_sink: int,
        protect_recency: int,
        device: torch.device | str,
        sync_across_ranks: bool = True,
    ) -> torch.Tensor:
        """Sample a bool tensor of shape [num_chunks] indicating which chunks
        to drop from the K/V context for the current training step.

        Args:
            num_chunks: total number of chunks in the clean (KV) half.
            prob: per-chunk independent drop probability.
            protect_sink: first N chunks always kept (forced False).
            protect_recency: last N chunks always kept (forced False).
            device: device for the returned tensor.
            sync_across_ranks: when True and distributed is initialized, sample
                on rank 0 then broadcast — keeps mask identical across all ranks
                (required for SP > 1 and avoids inconsistent FSDP grads).

        Returns:
            Bool tensor [num_chunks]. True at index c == "drop chunk c".
        """
        import torch.distributed as dist
        protect_sink = max(0, int(protect_sink))
        protect_recency = max(0, int(protect_recency))

        if dist.is_initialized() and sync_across_ranks:
            # Rank 0 samples, broadcast to all ranks for SP/DP consistency.
            if dist.get_rank() == 0:
                drops = (torch.rand(num_chunks, device=device) < prob)
            else:
                drops = torch.zeros(num_chunks, device=device, dtype=torch.bool)
            # bool tensors do not broadcast cleanly on some NCCL versions; use uint8
            tmp = drops.to(torch.uint8).contiguous()
            dist.broadcast(tmp, src=0)
            drops = tmp.to(torch.bool)
        else:
            drops = (torch.rand(num_chunks, device=device) < prob)

        # Force protected chunks to False (kept).
        if protect_sink > 0:
            drops[:min(protect_sink, num_chunks)] = False
        if protect_recency > 0:
            drops[max(0, num_chunks - protect_recency):] = False
        return drops

    @staticmethod
    def _pose_distance_se3(c2w_a: torch.Tensor, c2w_b: torch.Tensor) -> torch.Tensor:
        """Pose distance between two SE3 (4,4) tensors.

        Matches ``longctx/inference/pose_aware_cache.py::pose_distance``:
            d = sqrt(|dxyz|^2 + (dyaw_deg / 60)^2)
        where ``yaw`` is extracted as ``atan2(R[0, 2], R[2, 2])`` (assumes roll ≈ 0,
        which holds for the WorldMem / Sekai world-model pose conventions used
        throughout the project).  Returns a 0-d tensor.

        Implemented in torch (no .item()/.cpu()) so a future vectorised pairwise
        variant can be JIT'd; the current pose-aware sampler is called once per
        step with ``num_chunks ~ 12``, so the inner loop cost is negligible.
        """
        dxyz = torch.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
        yaw_a = torch.atan2(c2w_a[0, 2], c2w_a[2, 2]) * (180.0 / math.pi)
        yaw_b = torch.atan2(c2w_b[0, 2], c2w_b[2, 2]) * (180.0 / math.pi)
        dyaw_raw = torch.remainder(torch.abs(yaw_a - yaw_b), 360.0)
        dyaw = torch.minimum(dyaw_raw, 360.0 - dyaw_raw) / 60.0
        return torch.sqrt(dxyz * dxyz + dyaw * dyaw)

    @staticmethod
    def _sample_pose_aware_dropped_chunks_tensor(
        num_chunks: int,
        poses_per_chunk: torch.Tensor,
        epsilon: float,
        protect_sink: int,
        protect_recency: int,
        device: torch.device | str,
        sync_across_ranks: bool = True,
    ) -> torch.Tensor:
        """Pose-aware chunk_drop sampler — drop pose-redundant chunks to mimic the
        ``sink + pose_cons`` KV compression used at inference.

        Algorithm (mirrors ``longctx/inference/pose_aware_cache.py``):

          1. Compute pairwise SE3 pose distance between all (chunk_i, chunk_j) pairs
             using the same metric as inference (``sqrt(dxyz^2 + (dyaw_deg/60)^2)``).
          2. Greedily cluster chunks where any pair with distance < epsilon is in
             the same cluster (union-find).
          3. From every cluster of size ≥ 2, randomly retain **exactly one** chunk
             and mark the others as dropped.
          4. Force protected chunks (sink prefix + recency suffix) to be kept.
             A protected chunk's cluster is processed *before* dropping: if any
             chunk in the cluster is protected, that protected chunk is kept and
             the rest of the cluster is dropped; if multiple chunks in the cluster
             are protected, all protected chunks survive.

        Args:
            num_chunks: total number of chunks in the clean (KV) half.
            poses_per_chunk: float tensor [num_chunks, 4, 4] of SE3 c2w matrices
                — typically the *mean* c2w over the ``num_frame_per_block`` latent
                frames inside the chunk (computed by the trainer).  Must live on
                ``device``.
            epsilon: pose-distance threshold (default 0.5, matching inference
                ``sink+pose_cons`` Pareto winner from longctx_axis_b).
            protect_sink: first N chunks always kept (forced False).
            protect_recency: last N chunks always kept (forced False).
            device: device for the returned tensor.
            sync_across_ranks: when True and distributed is initialized, sample
                on rank 0 then broadcast — keeps mask identical across all ranks.

        Returns:
            Bool tensor [num_chunks]. True at index c == "drop chunk c".
            Same dtype/shape contract as ``_sample_dropped_chunks_tensor`` so the
            downstream ``_prepare_*_mask`` paths consume it identically.
        """
        protect_sink = max(0, int(protect_sink))
        protect_recency = max(0, int(protect_recency))
        assert poses_per_chunk.shape == (num_chunks, 4, 4), (
            f"poses_per_chunk shape {tuple(poses_per_chunk.shape)} != "
            f"expected ({num_chunks}, 4, 4)"
        )

        def _sample_on_rank0() -> torch.Tensor:
            drops = torch.zeros(num_chunks, device=device, dtype=torch.bool)

            # ---- 1. pairwise distance + union-find clustering ----
            # parent[i] = root index; small num_chunks (≤ 24 typical) so O(N^2) loop is fine.
            parent = list(range(num_chunks))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # path compression
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            poses_cpu = poses_per_chunk.detach().to(torch.float32)
            for i in range(num_chunks):
                for j in range(i + 1, num_chunks):
                    d = CausalWanModel._pose_distance_se3(poses_cpu[i], poses_cpu[j])
                    if d.item() < epsilon:
                        union(i, j)

            # ---- 2. group by root ----
            clusters: dict[int, list[int]] = {}
            for i in range(num_chunks):
                r = find(i)
                clusters.setdefault(r, []).append(i)

            # ---- 3. compute protection mask (sink + recency) ----
            is_protected = torch.zeros(num_chunks, dtype=torch.bool, device=device)
            if protect_sink > 0:
                is_protected[:min(protect_sink, num_chunks)] = True
            if protect_recency > 0:
                is_protected[max(0, num_chunks - protect_recency):] = True

            # ---- 4. per-cluster: pick one to keep, drop the rest ----
            # If cluster has any protected chunk, ALL protected chunks survive and
            # all unprotected ones are dropped.  Otherwise pick a random survivor.
            for _root, members in clusters.items():
                if len(members) <= 1:
                    continue  # singleton — nothing to drop
                protected_members = [m for m in members if bool(is_protected[m].item())]
                unprotected_members = [m for m in members if not bool(is_protected[m].item())]
                if len(protected_members) > 0:
                    # All unprotected members get dropped
                    for m in unprotected_members:
                        drops[m] = True
                else:
                    # Random survivor (uniform among cluster members)
                    keep_pos = int(torch.randint(0, len(members), (1,)).item())
                    for k_idx, m in enumerate(members):
                        if k_idx != keep_pos:
                            drops[m] = True

            # ---- 5. final safety: re-zero any protected slot (idempotent) ----
            if protect_sink > 0:
                drops[:min(protect_sink, num_chunks)] = False
            if protect_recency > 0:
                drops[max(0, num_chunks - protect_recency):] = False
            return drops

        import torch.distributed as dist
        if dist.is_initialized() and sync_across_ranks:
            if dist.get_rank() == 0:
                drops = _sample_on_rank0()
            else:
                drops = torch.zeros(num_chunks, device=device, dtype=torch.bool)
            tmp = drops.to(torch.uint8).contiguous()
            dist.broadcast(tmp, src=0)
            drops = tmp.to(torch.bool)
        else:
            drops = _sample_on_rank0()
        return drops

    @staticmethod
    def _sample_landmark_kept_chunks_tensor(
        num_chunks: int,
        poses_per_chunk: torch.Tensor,
        epsilon: float,
        n_sink: int,
        recent_w: int,
        n_landmark: int,
        device: torch.device | str,
        sync_across_ranks: bool = True,
    ) -> torch.Tensor:
        """Landmark keep-set sampler — train-side mirror of the v1.5a inference cache
        (``longctx/inference/landmark_cache.py``).  Unlike the pose-aware *merge*
        sampler (union-find collapse, which FREEZES), this SELECTS a sparse keep-set:

            keep = sink[: n_sink]
                 ∪ recent[-recent_w :]
                 ∪ up to n_landmark pose-DISTINCT landmarks greedily ε-covered from
                   the OLD middle chunks (those neither sink nor recent).

        Everything else is dropped.  So each query attends a budget-sized
        {sink + recent + scattered pose-distinct landmarks} set — the exact pattern
        v1.5 builds at inference.  PRoPE bakes pose into K, so the kept set is
        pose-indexed (not count-indexed), which is why a short (12-chunk) training
        horizon can teach a skill that transfers to long (36-chunk) inference.

        Landmark selection (greedy ε-cover, mirrors landmark_cache._bank_insert):
          walk old chunks in order; keep one as a landmark iff it is ≥ epsilon (in
          ``pose_distance``) from EVERY already-kept pose (sink, recent, and earlier
          landmarks) — i.e. it covers a pose region nothing else holds.  Stop at
          n_landmark.  Deterministic given poses ⇒ already rank-consistent; we still
          broadcast for parity with the other samplers.

        Returns:
            Bool tensor [num_chunks]. True at index c == "drop chunk c".  Same
            contract as ``_sample_dropped_chunks_tensor``.
        """
        n_sink = max(0, int(n_sink))
        recent_w = max(0, int(recent_w))
        n_landmark = max(0, int(n_landmark))
        assert poses_per_chunk.shape == (num_chunks, 4, 4), (
            f"poses_per_chunk shape {tuple(poses_per_chunk.shape)} != "
            f"expected ({num_chunks}, 4, 4)"
        )

        def _sample_on_rank0() -> torch.Tensor:
            poses = poses_per_chunk.detach().to(torch.float32)
            sink_idx = list(range(min(n_sink, num_chunks)))
            recent_idx = list(range(max(0, num_chunks - recent_w), num_chunks))
            kept = set(sink_idx) | set(recent_idx)
            old_idx = [c for c in range(num_chunks) if c not in kept]
            # greedy ε-cover landmark selection among old chunks
            landmark_idx = []
            kept_poses = [poses[c] for c in kept]
            for c in old_idx:
                if len(landmark_idx) >= n_landmark:
                    break
                pc = poses[c]
                if all(CausalWanModel._pose_distance_se3(pc, kp).item() >= epsilon
                       for kp in kept_poses):
                    landmark_idx.append(c)
                    kept_poses.append(pc)
            kept |= set(landmark_idx)
            drops = torch.ones(num_chunks, device=device, dtype=torch.bool)
            for c in kept:
                drops[c] = False
            return drops

        import torch.distributed as dist
        if dist.is_initialized() and sync_across_ranks:
            if dist.get_rank() == 0:
                drops = _sample_on_rank0()
            else:
                drops = torch.zeros(num_chunks, device=device, dtype=torch.bool)
            tmp = drops.to(torch.uint8).contiguous()
            dist.broadcast(tmp, src=0)
            drops = tmp.to(torch.bool)
        else:
            drops = _sample_on_rank0()
        return drops

    @staticmethod
    def _sample_random_fixed_dropped(
        num_chunks: int,
        keep_total: int,
        n_sink: int,
        device: torch.device | str,
        sync_across_ranks: bool = True,
    ) -> torch.Tensor:
        """Budget-matched RANDOM baseline (the foil for landmark_consol).

        Keep a FIXED-size set per clip: sink[:n_sink] (always kept — the attention
        anchor / table-stakes, dropping it collapses any method) PLUS
        (keep_total - n_sink) chunks chosen UNIFORMLY AT RANDOM from the rest.
        Everything else dropped.  ONE keep-set for the WHOLE clip (static, 1D),
        applied identically to every query chunk → chunk5 and chunk8 see the SAME
        history.  It CANNOT mimic the per-chunk consolidation that inference does
        — that is exactly the point: same budget, random-vs-pose selection, no
        per-chunk variation.

        Returns 1D bool [num_chunks], True == drop.  RNG → rank-0 samples then
        broadcasts (uint8) for SP/DP consistency, same contract as the other
        random sampler.
        """
        n_sink = max(0, int(n_sink))
        keep_total = max(n_sink, int(keep_total))

        def _sample_on_rank0() -> torch.Tensor:
            drops = torch.ones(num_chunks, device=device, dtype=torch.bool)
            sink_idx = list(range(min(n_sink, num_chunks)))
            for c in sink_idx:
                drops[c] = False
            rest = [c for c in range(num_chunks) if c not in sink_idx]
            n_extra = min(max(0, keep_total - len(sink_idx)), len(rest))
            if n_extra > 0:
                perm = torch.randperm(len(rest), device=device)[:n_extra].tolist()
                for p in perm:
                    drops[rest[p]] = False
            return drops

        if dist.is_initialized() and sync_across_ranks:
            if dist.get_rank() == 0:
                drops = _sample_on_rank0()
            else:
                drops = torch.zeros(num_chunks, device=device, dtype=torch.bool)
            tmp = drops.to(torch.uint8).contiguous()
            dist.broadcast(tmp, src=0)
            drops = tmp.to(torch.bool)
        else:
            drops = _sample_on_rank0()
        return drops

    @staticmethod
    def _sample_landmark_consol_keepmatrix(
        num_chunks: int,
        poses_per_chunk: torch.Tensor,
        n_sink: int,
        recent_w: int,
        retrieve_k: int,
        K: int,
        eps: float,
        diverse: bool,
        device: torch.device | str,
        sync_across_ranks: bool = False,
        cache_mode: str = "a",
    ) -> torch.Tensor:
        """PER-CHUNK consolidation keep-MATRIX — the (a) novelty.

        cache_mode='a' (DEFAULT, v15a): bounded distinct-landmark bank, query attends
        the ENTIRE bank (no top-k) → keep_i = sink ∪ recent ∪ all-bank. Retrieval is
        left to PRoPE's pose-indexed attention; the bank cap (= retrieve_k arg here) +
        ε-cover eviction keep a DIVERSE coarse cover of everywhere visited.
        cache_mode='b' (v15b): explicit pose top-k retrieval (legacy; not used by default).

        Replays the SAME inference cache (``longctx/inference/landmark_cache.py``
        ``LandmarkCache``, mode 'b') over the GT poses, chunk by chunk, recording
        for EACH query chunk i the exact set ``keep_i`` it WOULD attend at
        inference.  Bakes all keep_i into a 2D ``[num_chunks, num_chunks]`` mask:
        row i = which past clean chunks query chunk i attends.  row5 != row8 →
        matches the consolidating inference (fixes the static B1 train/infer gap).

        Ordering mirrors the inference loop EXACTLY (pipeline
        ``_apply_landmark`` + ``LandmarkCache.add_chunk``):
          for each generated chunk c:  R = add_chunk(c, pose_c, pose_next=pose_c)
          → the resulting cache state (sink + recent + bank[R]) is the history the
          NEXT chunk (c+1) attends.  So ``keep_{c+1}`` = born indices of
          {sink ∪ recent ∪ bank[R]} captured right AFTER adding chunk c, with the
          retrieval target = pose_c (NOT pose_{c+1} — the pipeline approximates the
          next FOV by the current pose).  Chunk 0 has no history → row 0 attends
          only itself.

        Poses MUST already be in the inference convention (first-frame rotation +
        mean translation, == ``_compute_chunk_pose``); the caller builds them so.

        Deterministic given poses (no RNG) → sync_across_ranks=False (same
        rationale as landmark mode: SP group sees bit-identical prope_context so
        every rank builds the identical matrix; different DP groups correctly get
        different matrices).

        Returns 2D bool [num_chunks, num_chunks], True == drop (query i does NOT
        attend clean chunk j).
        """
        from longctx.inference.landmark_cache import LandmarkCache

        n_sink = max(0, int(n_sink))
        recent_w = max(0, int(recent_w))
        retrieve_k = max(0, int(retrieve_k))
        K = max(1, int(K))
        assert poses_per_chunk.shape == (num_chunks, 4, 4), (
            f"poses_per_chunk shape {tuple(poses_per_chunk.shape)} != "
            f"expected ({num_chunks}, 4, 4)"
        )

        def _build_on_rank0() -> torch.Tensor:
            poses = poses_per_chunk.detach().to(torch.float32).cpu().numpy()
            _mode = "a" if str(cache_mode).lower() in ("a", "consol", "consol_div") else "b"
            cache = LandmarkCache(
                chunk_tokens=1, n_sink=n_sink, recent_w=recent_w, K=K,
                retrieve_k=retrieve_k, eps=eps, mode=_mode, diverse=bool(diverse),
            )
            dummy = torch.zeros(1, 1, 1)
            dropped = torch.ones(num_chunks, num_chunks, device=device, dtype=torch.bool)
            keep_for_next = None  # keep-set chunk (i) uses = cache state after adding chunk i-1
            for i in range(num_chunks):
                if keep_for_next is not None:
                    for j in keep_for_next:
                        if 0 <= j < i:
                            dropped[i, j] = False
                dropped[i, i] = False  # own clean block (eye also covers the diagonal)
                # add chunk i; retrieval target = pose_i (== pipeline pose_next=chunk_pose)
                R = cache.add_chunk(dummy, dummy, poses[i], poses[i])
                keep_for_next = (
                    {lm.born for lm in cache.sink}
                    | {lm.born for lm in cache.recent}
                    | {cache.bank[r].born for r in R}
                )
            return dropped

        if dist.is_initialized() and sync_across_ranks:
            if dist.get_rank() == 0:
                dropped = _build_on_rank0()
            else:
                dropped = torch.zeros(num_chunks, num_chunks, device=device, dtype=torch.bool)
            tmp = dropped.to(torch.uint8).contiguous()
            dist.broadcast(tmp, src=0)
            dropped = tmp.to(torch.bool)
        else:
            dropped = _build_on_rank0()
        return dropped

    def _sample_chunk_drop_for_step(self, num_chunks_total, num_frame_per_block,
                                    prope_context, device):
        """Sample the per-step dropped_chunks bool tensor according to
        ``self.chunk_drop_config['mode']`` (random / pose_aware / landmark).

        SHARED by the non-SP (``_forward_train``) and SP
        (``sp_dit_causal_forward_train``) teacher-forcing paths so the two NEVER
        diverge — the SP path historically had NO chunk_drop at all, which made
        A1b's chunk_drop a silent no-op. Routing both through one helper makes the
        mask provably identical regardless of sequence_parallel_size.

        Emits a one-shot rank-0 confirmation print so the training log proves the
        mask is live (gate + the kept-set for landmark mode).

        Returns: (dropped_chunks_tensor [num_chunks_total] bool, mode str).
        """
        cfg = self.chunk_drop_config
        mode = str(cfg.get("mode", "")).lower()
        # clean user-facing aliases: "consol" == landmark_consol, "random_drop" == random_fixed
        mode = {"consol": "landmark_consol", "random_drop": "random_fixed"}.get(mode, mode)
        if not mode:
            mode = "pose_aware" if bool(cfg.get("pose_aware", False)) else "random"

        poses_per_chunk = None
        poses_consol = None
        if mode in ("pose_aware", "landmark", "landmark_consol"):
            if prope_context is None:
                raise RuntimeError(
                    f"chunk_drop mode={mode!r} needs poses but prope_context is None; "
                    "pose/landmark chunk_drop requires the trainer to pass prope_context."
                )
            pc = prope_context[0].to(device=device, dtype=torch.float32)
            T_latent = pc.shape[0]
            assert T_latent >= num_chunks_total * num_frame_per_block, (
                f"prope_context T_latent={T_latent} < num_chunks_total*block="
                f"{num_chunks_total * num_frame_per_block}"
            )
            pcv = pc[: num_chunks_total * num_frame_per_block].view(
                num_chunks_total, num_frame_per_block, 4, 4
            )
            poses_per_chunk = pcv.mean(dim=1)  # landmark / pose_aware (unchanged)
            if mode == "landmark_consol":
                # Match inference _compute_chunk_pose (pipeline/causal_diffusion_
                # inference_compressed.py): first-frame ROTATION + MEAN translation —
                # NOT the elementwise mean above — so the train replay's pose
                # distances are identical to what v15b sees at inference.
                poses_consol = pcv[:, 0].clone()
                poses_consol[:, :3, 3] = pcv[:, :, :3, 3].mean(dim=1)

        if mode == "landmark":
            # sync_across_ranks=False ON PURPOSE: landmark selection is a DETERMINISTIC
            # function of poses_per_chunk (greedy ε-cover, no RNG). Within an SP group
            # prope_context is broadcast bit-identical (_sync_batch_for_sequence_parallel),
            # so every rank computes the SAME mask within-group (consistent for the SP
            # all_to_all) while DIFFERENT DP groups (different clips/poses) correctly get
            # DIFFERENT landmark sets. A world broadcast (sync=True) would instead force
            # rank-0's clip's landmark indices onto every DP group's (different) poses —
            # a silent pose-mismatch. Deterministic + group-identical input ⇒ no collective
            # needed, no hang risk.
            dropped = self._sample_landmark_kept_chunks_tensor(
                num_chunks=num_chunks_total,
                poses_per_chunk=poses_per_chunk,
                epsilon=float(cfg.get("landmark_epsilon", 0.5)),
                n_sink=int(cfg.get("n_sink", 1)),
                recent_w=int(cfg.get("recent_w", 3)),
                n_landmark=int(cfg.get("n_landmark", 2)),
                device=device, sync_across_ranks=False,
            )
        elif mode == "pose_aware":
            dropped = self._sample_pose_aware_dropped_chunks_tensor(
                num_chunks=num_chunks_total,
                poses_per_chunk=poses_per_chunk,
                epsilon=float(cfg.get("pose_epsilon", 0.5)),
                protect_sink=int(cfg.get("protect_sink", 0)),
                protect_recency=int(cfg.get("protect_recency", 0)),
                device=device, sync_across_ranks=True,
            )
        elif mode == "landmark_consol":
            # (a) novelty — PER-CHUNK consolidation replay → 2D [C,C] keep-matrix.
            # Deterministic given poses → sync=False (same as landmark mode).
            dropped = self._sample_landmark_consol_keepmatrix(
                num_chunks=num_chunks_total,
                poses_per_chunk=poses_consol,
                n_sink=int(cfg.get("n_sink", 1)),
                recent_w=int(cfg.get("recent_w", 3)),
                retrieve_k=int(cfg.get("retrieve_k", 2)),
                K=int(cfg.get("bank_K", cfg.get("K", 30))),
                eps=float(cfg.get("landmark_epsilon", cfg.get("eps", 0.5))),
                diverse=bool(cfg.get("diverse", False)),
                cache_mode=str(cfg.get("cache_mode", "a")),
                device=device, sync_across_ranks=False,
            )
        elif mode == "random_fixed":
            # Budget-matched random baseline (the foil) → static 1D keep-set.
            dropped = self._sample_random_fixed_dropped(
                num_chunks=num_chunks_total,
                keep_total=int(cfg.get("keep_total", 6)),
                n_sink=int(cfg.get("n_sink", 1)),
                device=device, sync_across_ranks=True,
            )
        else:
            dropped = self._sample_dropped_chunks_tensor(
                num_chunks=num_chunks_total,
                prob=float(cfg.get("prob", 0.0)),
                protect_sink=int(cfg.get("protect_sink", 0)),
                protect_recency=int(cfg.get("protect_recency", 0)),
                device=device, sync_across_ranks=True,
            )

        if not getattr(self, "_cd_step_dbg", False):
            import torch.distributed as _d
            if (not _d.is_initialized()) or _d.get_rank() == 0:
                if dropped.dim() == 2:
                    per_row_keep = [int((~dropped[i]).sum()) for i in range(num_chunks_total)]
                    print(f"[chunk_drop LIVE] mode={mode} num_chunks={num_chunks_total} "
                          f"PER-ROW keep_counts={per_row_keep} "
                          f"(row i = #clean chunks query-chunk i attends; rows differ ⇒ per-chunk)",
                          flush=True)
                else:
                    kept = [int(c) for c in range(num_chunks_total) if not bool(dropped[c])]
                    print(f"[chunk_drop LIVE] mode={mode} num_chunks={num_chunks_total} "
                          f"kept={kept} dropped={int(dropped.sum())}", flush=True)
            self._cd_step_dbg = True
        return dropped, mode

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # apply the rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # perform the rolling shift
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # insert the new key/value
                    cache["k"][:, local_start_index:local_end_index] = new_k
                    cache["v"][:, local_start_index:local_end_index] = new_v
                    
                elif update_info["action"] == "direct_insert":
                    # insert directly
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # insert the new key/value
                    cache["k"][:, local_start_index:local_end_index] = new_k
                    cache["v"][:, local_start_index:local_end_index] = new_v
            
            # update the indices
            kv_cache[block_index]["global_end_index"].fill_(current_end)
            kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (880 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, F]
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

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
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
        # assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
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

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # ============================================================
        # Construct blockwise causal attn mask
        # ------------------------------------------------------------
        # chunk_drop training augmentation (mirrors action_model.py logic):
        # active only when grad enabled, config enabled, and clean_x present.
        # ============================================================
        frame_seqlen_compute = (
            x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        )

        chunk_drop_active = (
            torch.is_grad_enabled()
            and isinstance(self.chunk_drop_config, dict)
            and self.chunk_drop_config.get("enabled", False)
            and clean_x is not None
            and not self.independent_first_frame
        )

        if chunk_drop_active:
            num_chunks_total = math.ceil(x.shape[2] / self.num_frame_per_block)
            # pose-aware vs random: pose-aware requires poses_per_chunk to be set on
            # self by the caller (currently CausalWanModel base does not accept
            # prope_context; pose-aware path here is dormant — the active path is
            # CausalWanActionModel._forward_train which threads prope_context).
            pose_aware = bool(self.chunk_drop_config.get("pose_aware", False))
            poses_per_chunk = getattr(self, "_poses_per_chunk_for_drop", None)
            if pose_aware and poses_per_chunk is not None:
                dropped_chunks_tensor = self._sample_pose_aware_dropped_chunks_tensor(
                    num_chunks=num_chunks_total,
                    poses_per_chunk=poses_per_chunk,
                    epsilon=float(self.chunk_drop_config.get("pose_epsilon", 0.5)),
                    protect_sink=int(self.chunk_drop_config.get("protect_sink", 0)),
                    protect_recency=int(self.chunk_drop_config.get("protect_recency", 0)),
                    device=device,
                    sync_across_ranks=True,
                )
            else:
                dropped_chunks_tensor = self._sample_dropped_chunks_tensor(
                    num_chunks=num_chunks_total,
                    prob=float(self.chunk_drop_config.get("prob", 0.0)),
                    protect_sink=int(self.chunk_drop_config.get("protect_sink", 0)),
                    protect_recency=int(self.chunk_drop_config.get("protect_recency", 0)),
                    device=device,
                    sync_across_ranks=True,
                )
            block_mask_step = self._prepare_teacher_forcing_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=frame_seqlen_compute,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=getattr(self, "local_attn_size", -1),
                dropped_chunks_tensor=dropped_chunks_tensor,
            )
        elif self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=getattr(self, "local_attn_size", -1),
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        block_mask_for_step = block_mask_step if chunk_drop_active else self.block_mask

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
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
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
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
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask_for_step)  # chunk_drop-aware: see top of fn

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
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
