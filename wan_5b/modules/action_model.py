import math

from wan_5b.modules.causal_model import (
    CausalWanModel,
    CausalWanSelfAttention,
    MultiShotT2VCrossAttention,
    run_flex_attention_with_grouped_mask,
)
from wan_5b.modules.attention import attention
from torch.nn.attention.flex_attention import flex_attention
from wan_5b.modules.model import rope_apply
from wan_5b.modules.causal_model import causal_rope_apply
from wan.modules.model import (
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    sinusoidal_embedding_1d,
    WanModel,
    WanSelfAttention,
)
import torch.nn as nn
import torch
from einops import repeat
from diffusers.configuration_utils import register_to_config

flex_attention = torch.compile(flex_attention, dynamic=False)


# Module-level flag for inspect-mode attention capture.
# Pipeline sets it to True only during Step 3.3 (clean K/V rewrite per chunk),
# trainer's monkey-patched attention() reads it and only captures q/k when True.
# This skips the pure-waste syncs of Step 3.1: ~30 denoise x 2 cfg x 30 layers ~= 1800 calls/chunk.
_INSPECT_CAPTURE_NOW = False


# Pose descriptor dimension: vec(R)[9] + t[3] extracted from 4x4 SE3 matrices
_POSE_DIM = 12


def pm_rope_delta_apply(x, delta):
    """Apply additional phase rotation e^{iδ} on top of already-RoPE-encoded Q or K.

    x:     [B, L, n, d]      any float dtype
    delta: [B, L, 1, d//2]   phase offsets in radians, broadcast over heads
    """
    orig_dtype = x.dtype
    x_c = torch.view_as_complex(
        x.to(torch.float32).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    )  # [B, L, n, d//2] complex
    rot = torch.polar(torch.ones_like(delta.float()), delta.float())  # [B, L, 1, d//2]
    return torch.view_as_real(x_c * rot).flatten(-2).to(orig_dtype)


# [metric-prope, opt-in] Three input-transform switches, set once by
# CausalWanActionModel.__init__ from model_kwargs (single model per process).
# All of them only transform inputs — zero state-dict impact:
#   _PROPE_LOG_T:      sign(t)*log1p(|t|) at pose_mlp's translation input — compresses the
#                      raw metric range while preserving cross-clip scale ordering
#   _SE3_T_BOUND:      row-wise bounding t/(1+||t||) on the SE3 multiplicative path — the
#                      4x4 matrix product cannot take raw tens-of-meters values; per-row
#                      deterministic => consistent through streaming inference / KV cache,
#                      no sequence-level statistics
#   _PLUCKER_DM_SPLIT: the plucker head skips the joint LayerNorm(6); d (unit vector) is
#                      kept as-is, m gets a direction-preserving log scaling
_PROPE_LOG_T = False
_SE3_T_BOUND = False
_PLUCKER_DM_SPLIT = False


def _invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert a batch of SE3 matrices [..., 4, 4]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = t_inv
    T_inv[..., 3, :3] = 0
    T_inv[..., 3, 3] = 1
    return T_inv


def _se3_block_apply(x: torch.Tensor, prope_context: torch.Tensor,
                     mode: str, frame_seqlen: int) -> torch.Tensor:
    """Apply SE3 transform as a block-diagonal 4×4 matmul over head_dim.

    x:             [B, L, H, D]  (D must be divisible by 4)
    prope_context: [B, T_ctx, 4, 4]
    mode:          'P_inv' → apply P^{-1} (for V); 'P' → apply P (for O)
    Returns same shape as x.
    """
    B, L, H, D = x.shape
    assert D % 4 == 0, f"head_dim {D} must be divisible by 4 for SE3 block-apply"
    num_frames = L // frame_seqlen
    ctx = prope_context[:, -num_frames:].to(x.dtype)   # [B, T, 4, 4]
    if _SE3_T_BOUND:
        # bounded translation for the multiplicative path; applied BEFORE the
        # invert so P and P_inv stay exact inverses of the SAME modified matrix
        t = ctx[..., :3, 3]
        ctx = ctx.clone()
        ctx[..., :3, 3] = t / (1.0 + t.norm(dim=-1, keepdim=True))
    if mode == 'P_inv':
        ctx = _invert_se3(ctx)

    x = x.permute(0, 2, 1, 3)                          # [B, H, L, D]
    x = x.view(B, H, num_frames, frame_seqlen, D // 4, 4)
    # ctx: [B, T, 4, 4] → [B, 1, T, 1, 1, 4, 4]
    x = torch.matmul(
        ctx.unsqueeze(1).unsqueeze(3).unsqueeze(4),
        x.unsqueeze(-1)
    ).squeeze(-1)
    x = x.view(B, H, L, D)
    return x.permute(0, 2, 1, 3)                        # [B, L, H, D]


class CausalWanPropSelfAttention(CausalWanSelfAttention):
    """PM-RoPE: camera-conditioned phase offsets added to standard RoPE on Q/K,
    plus zero-initialized value and output residual projections."""

    def __init__(self, *args, **kwargs):
        # discard legacy split-head kwargs for backward compat
        kwargs.pop("use_prope_gate", None)
        kwargs.pop("prope_all_heads", None)
        super().__init__(*args, **kwargs)

        dim = self.num_heads * self.head_dim

        # h(·): pose descriptor → per-frame phase offset δ for QK  [last layer zero-init]
        self.pose_mlp = nn.Sequential(
            nn.Linear(_POSE_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, self.head_dim // 2),
        )
        nn.init.zeros_(self.pose_mlp[2].weight)
        nn.init.zeros_(self.pose_mlp[2].bias)

        # V-side: value_proj(P_inv · v)  [zero-init → identity at step 0]
        self.value_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        # O-side: prope_proj(P · attn_out)  [zero-init → identity at step 0]
        self.prope_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.prope_proj.weight)
        nn.init.zeros_(self.prope_proj.bias)

    def forward(
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

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()

        # ── CC-full context ───────────────────────────────────────────────
        if prope_context is not None:
            # extract 12-dim pose descriptor from SE3 [B, T_ctx, 4, 4]
            R_flat = prope_context[..., :3, :3].flatten(-2)   # [B, T_ctx, 9]
            t_vec  = prope_context[..., :3, 3]                # [B, T_ctx, 3]
            if _PROPE_LOG_T:
                t_vec = torch.sign(t_vec) * torch.log1p(t_vec.abs())
            c_all  = torch.cat([R_flat, t_vec], dim=-1)       # [B, T_ctx, 12]

            # teacher-forcing detection: clean+noisy are both present → s = 2·T·S
            is_tf  = (kv_cache is None) and (s == seq_lens[0].item() * 2)
            T_half = s // (2 * frame_seqlen) if is_tf else s // frame_seqlen

            c = c_all[:, -T_half:]  # [B, T_half, 12]

            # QK phase offsets δ: [B, T_half·S, 1, d//2]
            delta = self.pose_mlp(c)
            delta_tok = (delta.unsqueeze(2)
                              .expand(-1, -1, frame_seqlen, -1)
                              .reshape(b, T_half * frame_seqlen, d // 2)
                              .unsqueeze(2))
            if is_tf:
                delta_tok = torch.cat([delta_tok, delta_tok], dim=1)

            # V-side P_inv residual: ṽ = v + value_proj(P_inv · v)
            if is_tf:
                half = s // 2
                v_p = torch.cat([
                    _se3_block_apply(v[:, :half], prope_context, 'P_inv', frame_seqlen),
                    _se3_block_apply(v[:, half:], prope_context, 'P_inv', frame_seqlen),
                ], dim=1)
            else:
                v_p = _se3_block_apply(v, prope_context, 'P_inv', frame_seqlen)
            v = (v.flatten(2) + self.value_proj(v_p.flatten(2))).view(b, s, n, d)
        else:
            is_tf     = False
            delta_tok = None

        # ── RoPE + phase offset δ ─────────────────────────────────────────
        def apply_rope(q_in, k_in, delta_in, start_f=None, slice_grid=None):
            actual_grid = slice_grid if slice_grid is not None else grid_sizes
            if start_f is not None:
                rq = causal_rope_apply(q_in, actual_grid, freqs, start_frame=start_f).type_as(v)
                rk = causal_rope_apply(k_in, actual_grid, freqs, start_frame=start_f).type_as(v)
            else:
                rq = rope_apply(q_in, actual_grid, freqs).type_as(v)
                rk = rope_apply(k_in, actual_grid, freqs).type_as(v)
            if delta_in is not None:
                rq = pm_rope_delta_apply(rq, delta_in)
                rk = pm_rope_delta_apply(rk, delta_in)
            return rq, rk

        if kv_cache is None:
            # ── Training ──────────────────────────────────────────────────
            if is_tf:
                half = s // 2
                d0 = delta_tok[:, :half] if delta_tok is not None else None
                d1 = delta_tok[:, half:] if delta_tok is not None else None
                rq0, rk0 = apply_rope(q[:, :half], k[:, :half], d0)
                rq1, rk1 = apply_rope(q[:, half:], k[:, half:], d1)
                roped_q = torch.cat([rq0, rq1], dim=1)
                roped_k = torch.cat([rk0, rk1], dim=1)
            else:
                roped_q, roped_k = apply_rope(q, k, delta_tok)

            padded_l = math.ceil(s / 128) * 128 - s
            pad = lambda t: torch.cat([t, torch.zeros([b, padded_l, n, d], device=t.device, dtype=t.dtype)], dim=1) if padded_l > 0 else t

            # grouped-mask aware (BLOCKER-1): plain BlockMask → identical single
            # flex_attention call; PerHeadGroupedBlockMask (mixed per-head windows
            # + chunk_drop) → one call per head group, stitched back in head order.
            attn_out = run_flex_attention_with_grouped_mask(
                flex_attention,
                pad(roped_q).transpose(2, 1),
                pad(roped_k).transpose(2, 1),
                pad(v).transpose(2, 1),
                block_mask,
            )
            if padded_l > 0:
                attn_out = attn_out[:, :, :-padded_l]
            attn_out = attn_out.transpose(2, 1)

        else:
            # ── Inference ─────────────────────────────────────────────────
            num_new = q.shape[1]
            cur_end = current_start + num_new
            sink_tokens = self.sink_size * frame_seqlen
            if isinstance(self.local_attn_size, (list, tuple)):
                finite_sizes = [abs(sz) for sz in self.local_attn_size if sz != -1]
                cache_local_attn_size = max(finite_sizes) if finite_sizes else -1
            else:
                cache_local_attn_size = abs(self.local_attn_size) if self.local_attn_size != -1 else -1
            self.max_attention_size = None if cache_local_attn_size == -1 else int(cache_local_attn_size) * int(frame_seqlen)
            kv_cache_size = kv_cache["k"].shape[1]

            g_start_f = current_start // frame_seqlen
            # [rope-remap] with BANKV2_ROPE_REMAP=1 the pipeline provides a start frame on a
            # virtual time axis in kv_cache (StreamingLLM-style repositioning); if absent, the
            # real position is used and behavior is unchanged. Only the RoPE phase is affected;
            # the local/global_end_index write-pointer bookkeeping is untouched.
            _rsf = kv_cache.get("rope_start_frame") if isinstance(kv_cache, dict) else None
            if _rsf is not None:
                g_start_f = int(_rsf.item())
            new_grid   = grid_sizes.clone()
            new_grid[:, 0] = max(1, num_new // frame_seqlen)
            q_f, k_f = apply_rope(q, k, delta_tok, start_f=g_start_f, slice_grid=new_grid)

            # sliding window eviction
            if cache_local_attn_size != -1 and (cur_end > kv_cache["global_end_index"].item()) and (num_new + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens  = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index   = kv_cache["local_end_index"].item() + cur_end - kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new
            else:
                local_end_index   = kv_cache["local_end_index"].item() + cur_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new

            kv_cache["k"][:, local_start_index:local_end_index] = k_f
            kv_cache["v"][:, local_start_index:local_end_index] = v

            if self.max_attention_size is None:
                k_slice = kv_cache["k"][:, :local_end_index]
                v_slice = kv_cache["v"][:, :local_end_index]
            else:
                slice_start = max(0, local_end_index - self.max_attention_size)
                k_slice = kv_cache["k"][:, slice_start:local_end_index]
                v_slice = kv_cache["v"][:, slice_start:local_end_index]

            attn_out = attention(q_f, k_slice, v_slice)

            kv_cache["global_end_index"].fill_(cur_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # ── O-side P residual: out = o(y) + prope_proj(P · y) ────────────
        out = self.o(attn_out.flatten(2))
        if prope_context is not None:
            if is_tf:
                half = attn_out.shape[1] // 2
                x_p = torch.cat([
                    _se3_block_apply(attn_out[:, :half], prope_context, 'P', frame_seqlen),
                    _se3_block_apply(attn_out[:, half:], prope_context, 'P', frame_seqlen),
                ], dim=1)
            else:
                x_p = _se3_block_apply(attn_out, prope_context, 'P', frame_seqlen)
            out = out + self.prope_proj(x_p.flatten(2))
        return out
        
class PluckerHeadV2(nn.Module):
    """[plucker-v2, opt-in] Entry head for per-token plucker maps.

    Replaces MLPProj for the action path when action_head == "v2". Differences,
    both audit-driven: no leading LayerNorm(in_dim) — the joint
    6-channel normalization crushes the direction channels once |m| >> |d|
    (inputs are already O(1) under plucker_480p_v2's re-anchor + t_scale) —
    and no trailing LayerNorm(out_dim), so injection amplitude is governed by
    the zero-init block gate instead of being pinned at unit RMS.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class CausalWanActionAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 max_discrete_actions=6,
                 action_embedding_type="add",
                 use_prope=False, # <-- enable PRoPE
                 use_prope_gate=True, # <-- if False, the 25% PRoPE heads connect directly without a gate
                 prope_all_heads=False, # <-- if True, every head runs PRoPE
                 action_head="v1",           # [plucker-v2, opt-in] "v2" = PluckerHeadV2 (no LayerNorms)
                 action_gate_zero_init=False,  # [plucker-v2, opt-in] zero-init scalar gate on injection
                 action_inject=True,         # [plucker-v2, opt-in] False = this block has no action path
        ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.action_embedding_type = action_embedding_type
        self.use_prope = use_prope
        self.use_prope_gate = use_prope_gate
        self.prope_all_heads = prope_all_heads

        # =====================================================================
        # choose the self-attn engine based on the switch
        # =====================================================================
        self.norm1 = WanLayerNorm(dim, eps)

        if self.use_prope:
            # PRoPE enabled: use the attention variant with built-in head splitting
            self.self_attn = CausalWanPropSelfAttention(
                dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
                use_prope_gate=use_prope_gate,
                prope_all_heads=prope_all_heads,
            )
        else:
            # PRoPE disabled: use standard attention
            self.self_attn = CausalWanSelfAttention(
                dim, num_heads, local_attn_size, sink_size, qk_norm, eps
            )
        
        # Note: instantiating self.prope_attn is no longer needed here,
        # and neither is self.prope_gate.

        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = MultiShotT2VCrossAttention(dim,num_heads,(-1, -1),qk_norm,eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        if action_embedding_type not in ("add", "adain"):
            raise ValueError(f"action_embedding_type must be 'add' or 'adain', got '{action_embedding_type}'")
        action_out_dim = 2 * dim if action_embedding_type == "adain" else dim
        if not action_inject:
            # [plucker-v2, opt-in] block excluded from action injection: no
            # parameters at all (keeps FSDP participation symmetric — the
            # module simply doesn't exist here, on every rank alike).
            self.action_embedding = None
            self.action_gate = None
        else:
            if action_head == "v2":
                self.action_embedding = PluckerHeadV2(max_discrete_actions, action_out_dim)
            else:
                self.action_embedding = MLPProj(max_discrete_actions, action_out_dim)
            # Zero-init scalar gate (AdaLN-Zero style): injection contributes
            # nothing at step 0 and its amplitude is learned. None when off —
            # default construction stays byte-identical to before.
            self.action_gate = nn.Parameter(torch.zeros(1)) if action_gate_zero_init else None

    def _apply_action_head(self, amap):
        """[metric-prope, opt-in] plucker d/m split: skip the joint LayerNorm(6)
        (which crushes unit-norm d once |m|>>|d| in metric units); d untouched,
        m direction-preserving log-magnitude scaled. Reuses existing weights —
        the leading LN's params simply go idle (zero state-dict impact)."""
        # gate on ndim==5 (plucker per-token maps (B,F,H,W,6)); ndim==3 6-dim
        # inputs are delta_euler vectors — must NEVER be d/m split.
        if not (_PLUCKER_DM_SPLIT and amap.ndim == 5 and amap.shape[-1] == 6):
            return self.action_embedding(amap)
        d, m = amap[..., :3], amap[..., 3:]
        mn = m.norm(dim=-1, keepdim=True)
        m = m / (mn + 1e-6) * torch.log1p(mn)
        x = torch.cat([d, m], dim=-1)
        head = self.action_embedding
        if isinstance(head, MLPProj):
            return head.proj[1:](x)     # skip leading LayerNorm(6)
        return head(x)                  # PluckerHeadV2 has no leading LN

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
        cache_start=None,
        combined_action_ids=None,
        teacher_forcing=False,
        prope_context=None,
    ):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # extract the branch feature
        normed_x = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)

        # inject the action_embedding (unchanged)
        # [plucker-v2] extra `self.action_embedding is not None` guard: blocks
        # excluded via action_inject_layers skip injection; with the default
        # config every block has the module and this guard is always True.
        if combined_action_ids is not None and self.action_embedding is not None:
            if combined_action_ids.ndim == 3:
                action_embedding = self._apply_action_head(combined_action_ids)
                spatial_size = grid_sizes[0, 1] * grid_sizes[0, 2]
                action_embedding = torch.repeat_interleave(action_embedding, spatial_size.item(), dim=1)
            elif combined_action_ids.ndim == 5:
                # per-token action maps (e.g. Plücker (B,F,H_tok,W_tok,6)): H,W must be the
                # DiT TOKEN grid — a transposed/right-numel-wrong-order grid would silently
                # misalign spatially in the row-major view below, so assert loudly here.
                _H, _W = int(grid_sizes[0, 1]), int(grid_sizes[0, 2])
                assert combined_action_ids.shape[2] == _H and combined_action_ids.shape[3] == _W, (
                    f"5-dim action map grid {tuple(combined_action_ids.shape[2:4])} != token grid ({_H},{_W}); "
                    f"for plucker use target_action_mode plucker_<res> matching image_or_video_shape"
                )
                action_embedding = self._apply_action_head(combined_action_ids)
                B, F, H, W, _ = action_embedding.shape
                action_embedding = action_embedding.view(B, F * H * W, -1)
            else:
                raise ValueError(f"Unsupported action_ids dimension: {combined_action_ids.ndim}")

            # [plucker-v2, opt-in] zero-init learnable gate scales the whole
            # injection (both add and adain branches). None by default.
            if self.action_gate is not None:
                action_embedding = action_embedding * self.action_gate

            if teacher_forcing:
                half = normed_x.shape[1] // 2
                if self.action_embedding_type == "adain":
                    action_scale, action_shift = action_embedding.chunk(2, dim=-1)
                    padded_scale = torch.zeros_like(normed_x)
                    padded_shift = torch.zeros_like(normed_x)
                    padded_scale[:, half:, :] = action_scale
                    padded_shift[:, half:, :] = action_shift
                    normed_x = normed_x * (1 + padded_scale) + padded_shift
                else:
                    padded_action = torch.zeros_like(normed_x)
                    padded_action[:, half:, :] = action_embedding
                    normed_x = normed_x + padded_action
            else:
                if self.action_embedding_type == "adain":
                    action_scale, action_shift = action_embedding.chunk(2, dim=-1)
                    normed_x = normed_x * (1 + action_scale) + action_shift
                else:
                    normed_x = normed_x + action_embedding

        # =====================================================================
        # Simplification: only one self_attn call is needed.
        # =====================================================================
        # A vanilla attention module ignores kwargs it does not recognize;
        # PropSelfAttention receives prope_context and performs head splitting.
        
        attn_kwargs = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start
        )
        
        if self.use_prope:
            attn_kwargs["prope_context"] = prope_context

        # a single computation — no redundant residual paths fighting each other
        y = self.self_attn(normed_x, **attn_kwargs)
        
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # =====================================================================
        # Cross-attention & FFN function
        # =====================================================================
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x

class CausalWanActionModel(CausalWanModel):
    """
    Layer-wise action variant built on CausalWanModel:
    - inherits CausalWanModel and reuses most of its logic
    - replaces blocks with CausalWanActionAttentionBlock
    - passes combined_action_ids to every block in forward
    """

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size=(1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        local_attn_size=-1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        max_discrete_actions: int = 6,
        action_embedding_type: str = "add",
        use_prope: bool = False,
        use_prope_gate: bool = True,
        prope_all_heads: bool = False,
        inference_local_attn_size=None,
        rope_spatial_interp=False,
        rope_spatial_ref_grid=None,
        rope_spatial_cur_grid=None,
        action_head: str = "v1",          # [plucker-v2, opt-in] "v2" = PluckerHeadV2
        action_gate_zero_init: bool = False,  # [plucker-v2, opt-in] zero-init injection gate
        action_inject_layers=None,        # [plucker-v2, opt-in] list of block idx; None = all
        prope_log_t: bool = False,        # [metric-prope, opt-in] pose_mlp eats sign*log1p(t)
        prope_se3_t_bound: bool = False,  # [metric-prope, opt-in] SE3 path eats t/(1+||t||)
        plucker_dm_split: bool = False,   # [metric-prope, opt-in] plucker head: no joint LN(6)
    ):
        # module-level switches (input transforms only; single model per process)
        global _PROPE_LOG_T, _SE3_T_BOUND, _PLUCKER_DM_SPLIT
        _PROPE_LOG_T = bool(prope_log_t)
        _SE3_T_BOUND = bool(prope_se3_t_bound)
        _PLUCKER_DM_SPLIT = bool(plucker_dm_split)
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            rope_spatial_interp=rope_spatial_interp,
            rope_spatial_ref_grid=rope_spatial_ref_grid,
            rope_spatial_cur_grid=rope_spatial_cur_grid,
        )

        self.use_prope = use_prope
        self.use_prope_gate = use_prope_gate
        self.prope_all_heads = prope_all_heads

        # Resolve inference-time local_attn_size (scalar) for KV cache window.
        # Priority: explicit config > auto-derive from per-head list > use as-is
        if inference_local_attn_size is not None:
            _infer_las = inference_local_attn_size
        elif isinstance(self.local_attn_size, (list, tuple)):
            finite_sizes = [abs(s) for s in self.local_attn_size if s != -1]
            _infer_las = max(finite_sizes) if finite_sizes else -1
        else:
            _infer_las = abs(self.local_attn_size) if self.local_attn_size != -1 else -1

        # replace the original blocks with action-aware blocks
        # [plucker-v2] action_inject_layers: None keeps the historical "every
        # block injects" behavior; a list (e.g. [0..9]) builds the action path
        # only in those blocks (AC3D-style early-layer conditioning).
        _inject_set = None if action_inject_layers is None else set(int(i) for i in action_inject_layers)
        cross_attn_type = "t2v_cross_attn" if "t2v" in self.model_type else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanActionAttentionBlock(
                    cross_attn_type=cross_attn_type,
                    dim=self.dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    local_attn_size=_infer_las,
                    sink_size=self.config.sink_size if hasattr(self, "config") else 0,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    max_discrete_actions=self.max_discrete_actions,
                    action_embedding_type=action_embedding_type,
                    use_prope=use_prope,
                    use_prope_gate=use_prope_gate,
                    prope_all_heads=prope_all_heads,
                    action_head=action_head,
                    action_gate_zero_init=action_gate_zero_init,
                    action_inject=(_inject_set is None or _b in _inject_set),
                )
                for _b in range(self.num_layers)
            ]
        )

        # run a custom initialization on each block's action_embedding
        try:
            from utils.wan_wrapper import custom_init
        except ImportError:
            custom_init = None

        if custom_init is not None:
            for block in self.blocks:
                if hasattr(block, "action_embedding") and block.action_embedding is not None:
                    custom_init(block.action_embedding)

    # -------- The two functions below modify CausalWanModel to pass combined_action_ids into each block --------

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
        cache_start: int = 0,
        combined_action_ids=None,
        prope_context=None,
    ):
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
            dim=0, sizes=t.shape
        )

        # text context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            combined_action_ids=combined_action_ids,
        )
        if self.use_prope:
            kwargs["prope_context"] = prope_context

        def create_custom_forward(module):
            def custom_forward(*inputs, **inner_kwargs):
                return module(*inputs, **inner_kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
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
        combined_action_ids=None,
        prope_context=None,
    ):
        self._maybe_randomize_routing()   # [ablation] re-sample global/local heads per train step (no-op unless enabled)
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # ============================================================
        # build the block-wise causal mask (reusing the base class's static method)
        # ------------------------------------------------------------
        # chunk_drop training augmentation:
        # - Only active when self.chunk_drop_config['enabled'] AND we're under
        #   grad enabled (i.e. real training step, not eval / inference).
        # - In that case, sample a fresh dropped_chunks tensor each step and
        #   build a fresh BlockMask; do NOT store on self.block_mask so the
        #   cached no-drop mask stays intact for inference / eval forward.
        # - Otherwise, fall back to the original cached self.block_mask path.
        # ============================================================
        frame_seqlen_compute = (
            x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        )

        chunk_drop_active = (
            torch.is_grad_enabled()
            and isinstance(self.chunk_drop_config, dict)
            and self.chunk_drop_config.get("enabled", False)
            and clean_x is not None  # only meaningful for teacher_forcing
            and not self.independent_first_frame  # i2v path not supported
        )

        if chunk_drop_active:
            # non-SP path: this rank holds the full sequence, so num_frames = x.shape[2].
            num_chunks_total = math.ceil(x.shape[2] / self.num_frame_per_block)
            # Shared sampler (random / pose_aware / landmark) — SAME helper the SP
            # forward uses, so the two paths never diverge. Emits a one-shot
            # "[chunk_drop LIVE] ..." confirmation so the log proves the mask applies.
            dropped_chunks_tensor, _mode = self._sample_chunk_drop_for_step(
                num_chunks_total=num_chunks_total,
                num_frame_per_block=self.num_frame_per_block,
                prope_context=prope_context,
                device=device,
            )
            block_mask_step = self._prepare_teacher_forcing_mask(
                device,
                num_frames=x.shape[2],
                frame_seqlen=frame_seqlen_compute,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                dropped_chunks_tensor=dropped_chunks_tensor,
            )
        elif self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=frame_seqlen_compute,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )

        # Resolve which mask we hand to the attention blocks below.
        block_mask_for_step = block_mask_step if chunk_drop_active else self.block_mask

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1
                )
                for u in x
            ]
        )

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
            dim=0, sizes=t.shape
        )

        # text context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor(
                [u.size(1) for u in clean_x], dtype=torch.long
            )
            # assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [
                    torch.cat(
                        [u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))],
                        dim=1,
                    )
                    for u in clean_x
                ]
            )

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)
            ).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask_for_step,  # chunk_drop-aware: see top of fn
            combined_action_ids=combined_action_ids,
            teacher_forcing=(clean_x is not None),
        )
        if self.use_prope:
            kwargs["prope_context"] = prope_context

        def create_custom_forward(module):
            def custom_forward(*inputs, **inner_kwargs):
                return module(*inputs, **inner_kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
