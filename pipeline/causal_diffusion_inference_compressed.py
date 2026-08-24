"""Inference pipeline with sink + pose-consolidation KV-cache compression.

Extends ``CausalDiffusionInferencePipeline`` *non-invasively*:

  * Subclass — no changes to ``wan_5b/modules/{action_model,causal_model}.py``.
  * Pre-allocated KV cache layout is preserved.  After each chunk's Step 3.3
    (the clean-context KV write), we *rewrite* the front portion of the
    cache with the compressed slots and patch ``local_end_index`` so the
    next chunk's attention sees the compressed set.
  * cond and uncond branches share **slot decisions** (pose is invariant w.r.t.
    text prompt) but maintain **independent K/V running means** (per
    designs/16 §5.2).
  * PRoPE δ baking on K: kept as-is.  We running-mean *already-δ-rotated* K's
    across chunks that are within ``epsilon`` of each other in pose space.
    Since δ ≈ identical for poses within ``epsilon`` (``pose_mlp`` is
    Lipschitz in pose), the merged K is approximately ``mean(raw_K) ⊙
    e^{iδ_shared}`` — δ semantics are preserved.  (designs/16 §3.1.)

Algorithm (one chunk):

  1. Run Steps 3.1 + 3.2 + 3.3 of the base pipeline unchanged.  Step 3.3
     writes the post-δ K (call this ``K_chunk_pos``, ``K_chunk_neg``) and V
     into ``kv_cache_*[L]["k"][:, le_pre:le_post]`` for each layer L.
  2. Extract ``K_chunk_*`` / ``V_chunk_*`` from each layer's cache.
  3. Extract chunk-level pose (mean of the chunk's per-frame c2w from
     ``prope_ctx[:, -num_frame_per_block:]``).
  4. Ask the shared ``PoseDecisionRegistry`` for a ``SlotDecision``.
  5. Apply the decision to each layer/branch's ``PoseAwareLayerCache``
     (K/V running mean update).
  6. ``commit`` the decision in the registry (updates pose/visit bookkeeping).
  7. For each layer/branch, ``write_into_cache`` rewrites the front portion
     of ``kv_cache["k"], kv_cache["v"]`` with the compressed slots and
     returns the new ``local_end_index``.  Update both
     ``local_end_index`` and ``global_end_index`` on the kv_cache dicts.
     Also reset ``self.cache_start_frame`` semantics — see ``_align_frame_indices``.
  8. Update the log-bias map per slot (``visit_count`` → ``log(visit_count)``)
     so the attention forward on the *next* chunk treats merged slots as
     having an effective multiplicity equal to ``visit_count``.

Log-bias mechanism:

  The pipeline monkey-patches ``wan_5b.modules.action_model.attention`` (the
  inference branch's attention call site, line ~262 in action_model.py) for
  the duration of inference.  When ``_KVC_LOG_BIAS_TOKENS`` is set on the
  module, the wrapper falls back to SDPA with an explicit additive
  log-bias mask on K's positional axis.  When unset (None) the wrapper is a
  zero-overhead passthrough to the original ``attention()``.

  This is the same idiom the trainer uses for inspect-mode attention
  capture (see ``trainer/diffusion_action.py:858-891``).

Caveats (honest):

  * Flash-attn 2/3 does not accept additive bias on logits.  When log-bias is
    active we fall back to SDPA — slower but correct.  Most chunks have
    no merges in their first slots, so log-bias is often all-zero — the
    wrapper short-circuits back to flash in that case.
  * Pre-allocated cache must be sized for at least ``budget_chunks ×
    chunk_tokens`` tokens.  The base pipeline pre-allocates for the full
    rollout, which is always ≥ budget × chunk_tokens, so this is satisfied
    by construction.  We assert it explicitly on first chunk.
  * Cond/uncond slot-decision sharing assumes pose is identical between the
    two branches.  This is **always** true here because pose comes from
    ``prope_context`` which is fed identically to both branches.  Confirmed
    by inspection of ``causal_diffusion_inference.py:286-307``.
"""
from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from utils.wan_default_config import wan_default_config


# Make the algo manager importable.  ``longctx/`` is not a python package,
# so we add the fork root to sys.path here once; the import below resolves
# relative to that.
_FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FORK_ROOT not in sys.path:
    sys.path.insert(0, _FORK_ROOT)

from longctx.inference.pose_aware_cache import (  # noqa: E402
    PoseAwareLayerCache,
    PoseDecisionRegistry,
)


@dataclass
class KVCompressionConfig:
    """YAML-driven knobs for inference-time KV compression."""
    enabled: bool = False
    mode: str = "sink_pose_cons"
    epsilon: float = 0.5
    n_sink: int = 4
    budget_chunks: int = 20
    k_renorm: bool = True
    log_bias: bool = True
    log_path: Optional[str] = None  # if set, dump compression events JSON here
    # Pose-noise robustness probe.  Not consumed inside the pipeline (noise is
    # injected upstream in inference_action.py before set_sequence) — kept here
    # for traceability so the compression log reflects what σ was used.
    pose_noise_sigma: float = 0.0
    # v1 graduated/selection policies (alternative to the legacy pose "merge"):
    #   "merge"  → original sink+pose_cons running-mean (default, back-compat)
    #   "window" → sink + sliding recent window, evict old (forgetting baseline)
    #   "naive"  → sink + recent + POOLED old chunks (graduated low-res memory)
    #   "select" → sink + recent + top-M old TOKENS by key-distinctiveness
    #   "consol" → sink + recent + a bounded bank of pose-deduped DISTINCT landmarks (greedy
    #              ε-cover, diversity-preserving eviction); attend the WHOLE bank — retrieval is
    #              left to PRoPE's pose-indexed attention. THIS IS THE METHOD. ("consol_div" =
    #              same but diverse/MMR retrieval.) Legacy aliases: v15a==consol, v15b/v15c=top-k.
    policy: str = "merge"
    recent_w: int = 7            # recent full-res window (chunks); window arm uses budget-n_sink
    pool_ratio: int = 4          # naive: pool factor for aged-out chunk tokens
    h2o_keep_frac: float = 0.25  # select: fraction of tokens kept per aged-out chunk
    landmark_k: int = 30         # bank capacity (distinct landmarks)
    retrieve_k: int = 6          # consol: bounded bank cap (ALL attended); top-k variants: # retrieved
    landmark_eps: float = 0.5    # pose ε for "redundant landmark" de-dup

    @classmethod
    def from_args(cls, args) -> "KVCompressionConfig":
        """Construct from an OmegaConf object's ``kv_compression`` block (if any)."""
        raw = getattr(args, "kv_compression", None)
        if raw is None:
            return cls(enabled=False)
        # Allow both OmegaConf DictConfig and plain dict.
        def _get(key, default):
            if hasattr(raw, "get"):
                try:
                    return raw.get(key, default)
                except Exception:
                    pass
            return getattr(raw, key, default)
        return cls(
            enabled=bool(_get("enabled", False)),
            mode=str(_get("mode", "sink_pose_cons")),
            epsilon=float(_get("epsilon", 0.5)),
            n_sink=int(_get("n_sink", 4)),
            budget_chunks=int(_get("budget_chunks", 20)),
            k_renorm=bool(_get("k_renorm", True)),
            log_bias=bool(_get("log_bias", True)),
            log_path=_get("log_path", None),
            pose_noise_sigma=float(_get("pose_noise_sigma", 0.0)),
            policy=str(_get("policy", "merge")),
            recent_w=int(_get("recent_w", 7)),
            pool_ratio=int(_get("pool_ratio", 4)),
            h2o_keep_frac=float(_get("h2o_keep_frac", 0.25)),
            landmark_k=int(_get("landmark_k", 30)),
            retrieve_k=int(_get("retrieve_k", 6)),
            landmark_eps=float(_get("landmark_eps", 0.5)),
        )


# ============================================================================
# Log-bias monkey-patch for ``wan_5b.modules.action_model.attention``.
# ============================================================================


def _install_log_bias_attention_patch():
    """Wrap ``wan_5b.modules.action_model.attention`` so that when a thread-
    local ``_KVC_LOG_BIAS_TOKENS`` is set, attention falls back to SDPA with
    an additive logit bias.  When unset, it's a zero-overhead passthrough.

    Idempotent — safe to call multiple times in one process.
    """
    import wan_5b.modules.action_model as _am

    if getattr(_am, "_kvc_patch_installed", False):
        return
    if not hasattr(_am, "_orig_attention_for_kvc"):
        _am._orig_attention_for_kvc = _am.attention

    # Module-level slot for the per-key log-bias tensor.  Shape: (k_seq_len,).
    # Bf16 or float32 — applied as additive bias to attention logits.
    _am._KVC_LOG_BIAS_TOKENS = None  # type: ignore[attr-defined]

    _orig_fn = _am._orig_attention_for_kvc

    def _patched_attention(q, k, v, **kw):
        bias = getattr(_am, "_KVC_LOG_BIAS_TOKENS", None)
        if bias is None:
            return _orig_fn(q, k, v, **kw)
        # Bias only applies when the K seq length matches the bias length.
        # This is a defensive check — cross-attn calls have different K len
        # and must not see the bias.
        k_len = int(k.shape[1])
        if bias.shape[0] != k_len:
            return _orig_fn(q, k, v, **kw)

        # Fall back to SDPA with explicit attn_mask = additive bias.
        # q, k, v: [B, L, H, D] in this codebase (see attention.py).
        b, lq, h, d = q.shape
        # SDPA expects [B, H, L, D]
        q_ = q.transpose(1, 2).to(torch.bfloat16)
        k_ = k.transpose(1, 2).to(torch.bfloat16)
        v_ = v.transpose(1, 2).to(torch.bfloat16)
        # attn_mask shape [..., Lq, Lk] additive; broadcast across batch/heads.
        attn_bias = bias.to(device=q.device, dtype=torch.bfloat16).view(1, 1, 1, k_len)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=attn_bias, is_causal=False, dropout_p=0.0,
        )
        return out.transpose(1, 2).contiguous().to(q.dtype)

    _am.attention = _patched_attention
    _am._kvc_patch_installed = True


@contextmanager
def _kvc_log_bias_active(bias_tensor: Optional[torch.Tensor]):
    """Context manager that sets / clears the module-level log-bias tensor."""
    import wan_5b.modules.action_model as _am
    prev = getattr(_am, "_KVC_LOG_BIAS_TOKENS", None)
    _am._KVC_LOG_BIAS_TOKENS = bias_tensor
    try:
        yield
    finally:
        _am._KVC_LOG_BIAS_TOKENS = prev


# ============================================================================
# Compressed inference pipeline
# ============================================================================


class CausalDiffusionInferenceCompressedPipeline(CausalDiffusionInferencePipeline):
    """``CausalDiffusionInferencePipeline`` with post-chunk KV consolidation.

    Overrides ``inference()`` to insert the sink+pose_cons step after each
    chunk's clean KV write (Step 3.3).  All other behavior is identical to
    the base class — sample scheduler, cross-attn cache, VAE decode, etc.

    Public constructor signature matches the base class — no extra args.
    ``kv_compression`` config is read from ``args`` via
    ``KVCompressionConfig.from_args(args)``.
    """

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__(args=args, device=device, generator=generator,
                         text_encoder=text_encoder, vae=vae)
        self.kvc_cfg = KVCompressionConfig.from_args(args)
        if self.kvc_cfg.enabled:
            print(f"[KVCompression] enabled: mode={self.kvc_cfg.mode} "
                  f"budget={self.kvc_cfg.budget_chunks} n_sink={self.kvc_cfg.n_sink} "
                  f"epsilon={self.kvc_cfg.epsilon} k_renorm={self.kvc_cfg.k_renorm} "
                  f"log_bias={self.kvc_cfg.log_bias}")
            # Install the log-bias monkey-patch once; harmless when no bias active.
            _install_log_bias_attention_patch()
        # Per-rollout state; (re)built in ``_init_compression_state``.
        self._registry: Optional[PoseDecisionRegistry] = None
        self._layer_caches_pos: List[PoseAwareLayerCache] = []
        self._layer_caches_neg: List[PoseAwareLayerCache] = []

    # ------------------------------------------------------------------
    # Compression state (one rollout)
    # ------------------------------------------------------------------

    def _init_compression_state(self, chunk_tokens: int, num_heads: int, head_dim: int):
        if not self.kvc_cfg.enabled:
            return
        # Cap budget to the pre-allocated cache size so we never exceed
        # capacity at ``write_into_cache`` time.  Cache size was set by the
        # base ``_initialize_kv_cache`` to fit the full rollout, so capping
        # at floor(cache_size / chunk_tokens) is safe.
        if self.kv_cache_pos is not None:
            cache_capacity = int(self.kv_cache_pos[0]["k"].shape[1])
            max_slots_fits = cache_capacity // chunk_tokens
            effective_budget = min(self.kvc_cfg.budget_chunks, max_slots_fits)
            if effective_budget < self.kvc_cfg.budget_chunks:
                print(f"[KVCompression] WARNING: budget_chunks "
                      f"{self.kvc_cfg.budget_chunks} > cache capacity "
                      f"{max_slots_fits} chunks; capping to {effective_budget}.")
        else:
            effective_budget = self.kvc_cfg.budget_chunks
        # v1 policies (window/naive/select): per-layer CompressCache, no pose registry.
        if self.kvc_cfg.policy in ("window", "naive", "select"):
            from longctx.inference.compress_cache_v1 import CompressCache
            rw = ((effective_budget - self.kvc_cfg.n_sink)
                  if self.kvc_cfg.policy == "window" else self.kvc_cfg.recent_w)
            def _mk():
                return CompressCache(
                    chunk_tokens=chunk_tokens, budget_chunks=effective_budget,
                    n_sink=self.kvc_cfg.n_sink, recent_w=rw, policy=self.kvc_cfg.policy,
                    pool_ratio=self.kvc_cfg.pool_ratio, h2o_keep_frac=self.kvc_cfg.h2o_keep_frac)
            self._cc_pos = [_mk() for _ in range(self.num_transformer_blocks)]
            self._cc_neg = [_mk() for _ in range(self.num_transformer_blocks)]
            self._registry = None
            self._kvc_chunk_counter = 0
            print(f"[KVCompression] policy={self.kvc_cfg.policy} budget={effective_budget} "
                  f"n_sink={self.kvc_cfg.n_sink} recent_w={rw} "
                  f"pool_ratio={self.kvc_cfg.pool_ratio} keep_frac={self.kvc_cfg.h2o_keep_frac}",
                  flush=True)
            return
        if self.kvc_cfg.policy in ("consol", "consol_div", "v15a", "v15b", "v15c"):
            from longctx.inference.landmark_cache import LandmarkCache
            # "consol" (the method): attend the WHOLE bounded landmark bank, no top-k → mode 'a'.
            # legacy "v15a"==consol, "v15b"/"v15c"==explicit top-k retrieval (kept for back-compat).
            mode = "a" if self.kvc_cfg.policy in ("consol", "consol_div", "v15a") else "b"
            diverse = self.kvc_cfg.policy in ("consol_div", "v15c")   # *_div = diverse (MMR/farthest-point)
            # ===== BUDGET-SPLIT KNOB =====
            # You set budget_chunks + n_sink + recent_w (a CAP); the # of landmarks
            # AUTO-fills the remainder. Rationale: recent (local continuity) saturates
            # fast, so extra budget should buy MORE landmarks (revisit memory), not more
            # recent. recent_w is capped so ≥1 landmark slot always survives. This makes
            # `budget` the only number you tune — no hand-computing landmark count.
            #   e.g. budget12, sink1, recent_w5  -> recent5  + landmark6  (== old default)
            #        budget24, sink1, recent_w7  -> recent7  + landmark16
            n_sink = self.kvc_cfg.n_sink
            eff_recent = min(self.kvc_cfg.recent_w, max(1, effective_budget - n_sink - 1))
            n_landmark = max(1, effective_budget - n_sink - eff_recent)
            # v15a attends ALL of its bank (LandmarkCache sets bank cap K=retrieve_k for
            # mode 'a'); v15b/c retrieve n_landmark from a larger bank K=landmark_k (must
            # be ≥ n_landmark).
            bank_K = max(self.kvc_cfg.landmark_k, n_landmark) if mode == "b" else n_landmark
            def _mklm():
                return LandmarkCache(
                    chunk_tokens=chunk_tokens, n_sink=n_sink,
                    recent_w=eff_recent, K=bank_K,
                    retrieve_k=n_landmark, eps=self.kvc_cfg.landmark_eps,
                    mode=mode, diverse=diverse)
            self._lm_pos = [_mklm() for _ in range(self.num_transformer_blocks)]
            self._lm_neg = [_mklm() for _ in range(self.num_transformer_blocks)]
            self._registry = None
            self._kvc_chunk_counter = 0
            print(f"[KVCompression] policy={self.kvc_cfg.policy} mode={mode} budget={effective_budget}"
                  f" = sink {n_sink} + recent {eff_recent} + landmark {n_landmark}"
                  f"  (bank K={bank_K}, recent_w cap={self.kvc_cfg.recent_w}, eps={self.kvc_cfg.landmark_eps})",
                  flush=True)
            return
        self._registry = PoseDecisionRegistry(
            budget_chunks=effective_budget,
            n_sink=self.kvc_cfg.n_sink,
            epsilon=self.kvc_cfg.epsilon,
        )
        self._layer_caches_pos = [
            PoseAwareLayerCache(chunk_tokens=chunk_tokens, num_heads=num_heads,
                                head_dim=head_dim, k_renorm=self.kvc_cfg.k_renorm)
            for _ in range(self.num_transformer_blocks)
        ]
        self._layer_caches_neg = [
            PoseAwareLayerCache(chunk_tokens=chunk_tokens, num_heads=num_heads,
                                head_dim=head_dim, k_renorm=self.kvc_cfg.k_renorm)
            for _ in range(self.num_transformer_blocks)
        ]
        self._kvc_chunk_counter = 0

    def _compute_chunk_pose(
        self,
        prope_ctx: torch.Tensor,
        num_frame_per_block: int,
    ) -> np.ndarray:
        """Extract a single (4, 4) chunk-level pose from prope_context.

        ``prope_ctx`` shape: ``[B, T_seen, 4, 4]``.  Last
        ``num_frame_per_block`` entries are the current chunk's poses.
        We take the mean translation and the *first* frame's rotation (rotation
        mean would require quaternion handling — the first frame is a fine
        proxy because the per-chunk rotation is small in our setups).
        """
        if prope_ctx is None:
            raise RuntimeError(
                "_compute_chunk_pose: prope_context is None.  KV compression "
                "requires the model to be running with use_prope=true."
            )
        cur = prope_ctx[0, -num_frame_per_block:].detach().to(torch.float32).cpu().numpy()
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = cur[0, :3, :3]
        out[:3, 3] = cur[:, :3, 3].mean(axis=0)
        return out

    def _extract_chunk_kv_from_cache(
        self,
        kv_cache: List[Dict[str, torch.Tensor]],
        local_start_index: int,
        local_end_index: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pull K, V for the just-written chunk out of each layer's cache."""
        k_list, v_list = [], []
        for layer_idx in range(self.num_transformer_blocks):
            k_list.append(
                kv_cache[layer_idx]["k"][:, local_start_index:local_end_index].clone()
            )
            v_list.append(
                kv_cache[layer_idx]["v"][:, local_start_index:local_end_index].clone()
            )
        return k_list, v_list

    def _apply_compression_for_chunk(
        self,
        chunk_pose: np.ndarray,
        chunk_k_pos: List[torch.Tensor],
        chunk_v_pos: List[torch.Tensor],
        chunk_k_neg: List[torch.Tensor],
        chunk_v_neg: List[torch.Tensor],
    ) -> int:
        """Run decide + per-layer apply + commit + rewrite cache.

        Returns the new ``local_end_index`` (same for all layers and both
        branches, since chunk_tokens is constant in our setup).
        """
        decision = self._registry.decide(self._kvc_chunk_counter, chunk_pose)
        for layer_idx in range(self.num_transformer_blocks):
            self._layer_caches_pos[layer_idx].apply_decision(
                decision, chunk_k_pos[layer_idx], chunk_v_pos[layer_idx]
            )
            self._layer_caches_neg[layer_idx].apply_decision(
                decision, chunk_k_neg[layer_idx], chunk_v_neg[layer_idx]
            )
        self._registry.commit(decision)
        # Rewrite each layer's cache front-portion + update local_end_index.
        new_le_pos = 0
        new_le_neg = 0
        for layer_idx in range(self.num_transformer_blocks):
            le_p = self._layer_caches_pos[layer_idx].write_into_cache(
                self.kv_cache_pos[layer_idx]
            )
            le_n = self._layer_caches_neg[layer_idx].write_into_cache(
                self.kv_cache_neg[layer_idx]
            )
            if layer_idx == 0:
                new_le_pos = le_p
                new_le_neg = le_n
            else:
                assert le_p == new_le_pos, "layer KV cache desync (pos)"
                assert le_n == new_le_neg, "layer KV cache desync (neg)"
            self.kv_cache_pos[layer_idx]["local_end_index"].fill_(le_p)
            self.kv_cache_neg[layer_idx]["local_end_index"].fill_(le_n)
        self._kvc_chunk_counter += 1
        return new_le_pos

    def _build_log_bias_tensor(
        self,
        layer_cache: PoseAwareLayerCache,
        chunk_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build per-key log-bias from a layer's slot visit counts.

        Returns ``None`` if all visit counts are 1 (no bias needed — short-
        circuit the SDPA fallback).
        """
        if not self.kvc_cfg.log_bias:
            return None
        visits = layer_cache.per_slot_visit_counts()
        if len(visits) == 0 or all(v == 1 for v in visits):
            return None
        # One bias value per token, repeated chunk_tokens times per slot.
        bias_per_slot = torch.tensor(
            [math.log(v) for v in visits], device=device, dtype=dtype
        )
        bias = bias_per_slot.unsqueeze(1).expand(-1, chunk_tokens).reshape(-1)
        return bias

    # ------------------------------------------------------------------
    # Overridden inference
    # ------------------------------------------------------------------

    def inference(  # noqa: C901  (mirrors the base method's structure)
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        combined_action_ids: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        num_inference_steps: Optional[int] = 30,
        prope_context: Optional[torch.Tensor] = None,
        control_manager=None,
    ):
        # Fast path when compression is disabled: just call the base method.
        if not self.kvc_cfg.enabled:
            return super().inference(
                noise=noise, text_prompts=text_prompts,
                combined_action_ids=combined_action_ids,
                initial_latent=initial_latent, return_latents=return_latents,
                start_frame_index=start_frame_index,
                num_inference_steps=num_inference_steps,
                prope_context=prope_context, control_manager=control_manager,
            )

        # Compressed path — same structure as the base, with hooks at Step 3.3.
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.frame_seq_length = height * width // 4

        if (not self.independent_first_frame
                or (self.independent_first_frame and initial_latent is not None)):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype,
        )

        # Step 1: Initialize KV cache.
        # ===== BOUNDED CACHE (the memory fix) =====
        # THIS is where the bounded cache lives. The base _initialize_kv_cache sizes the buffer
        # to `num_frames × frame_seq` — for full-KV that's the whole rollout (O(T) memory,
        # OOMs at k72+). But with compression the active set written into the buffer is
        # ALWAYS ≤ budget_chunks (LandmarkCache.write_into_cache / CompressCache only fill
        # the first ≤budget slots; the bank is a SEPARATE bounded structure). So when
        # compression is enabled we allocate a FIXED (budget+1)-chunk buffer — the +1 is
        # headroom for the freshly-written chunk before it's compressed back to ≤budget.
        # ⇒ inference memory O(budget), CONSTANT in video length T. full-KV keeps the
        # full-rollout alloc (it is the unbounded upper bound by definition).
        if self.kv_cache_pos is None:
            if self.kvc_cfg.enabled:
                cache_chunks = int(self.kvc_cfg.budget_chunks) + 1
                cache_frames = cache_chunks * self.num_frame_per_block
                print(f"[KVCompression] BOUNDED kv_cache = {cache_chunks} chunks "
                      f"({cache_frames} latent frames), vs full-rollout {num_output_frames} "
                      f"frames — memory O(budget), constant in length.", flush=True)
            else:
                cache_frames = num_output_frames  # full-KV: unbounded upper bound
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device,
                num_frames=cache_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device,
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Initialize compression state for this rollout.
        # chunk_tokens = num_frame_per_block × frame_seq_length
        chunk_tokens = self.num_frame_per_block * self.frame_seq_length
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        self._init_compression_state(chunk_tokens=chunk_tokens,
                                     num_heads=num_heads, head_dim=head_dim)

        prope_ctx_for_caching = (
            control_manager.full_prope_context if control_manager is not None
            else prope_context
        )

        # ----- Step 2: Cache context feature (initial conditioning frames) -----
        # The initial latents (e.g., the start image) are pushed into the cache
        # as-is — they are *also* subject to compression decisions, because
        # they're chunks in the per-chunk loop.  But the base pipeline writes
        # them differently (frame-by-frame for independent-first-frame, then in
        # chunks for the rest).  We mirror the base behavior for the writes and
        # then sweep them into the compression state in one go.
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self._call_generator_step33(
                    initial_latent[:, :1], conditional_dict, unconditional_dict,
                    timestep * 0, current_start_frame, cache_start_frame,
                    prope_ctx_for_caching, combined_action_ids_input=None,
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :, cache_start_frame:cache_start_frame + self.num_frame_per_block
                ]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self._call_generator_step33(
                    current_ref_latents, conditional_dict, unconditional_dict,
                    timestep * 0, current_start_frame, cache_start_frame,
                    prope_ctx_for_caching, combined_action_ids_input=None,
                )
                # Compress this initial chunk too — pose is available from
                # prope_ctx_for_caching.
                self._compress_post_step33(prope_ctx_for_caching)
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        if control_manager is not None:
            control_manager.cache_initial_frames(num_input_frames)

        # ----- Step 3: Temporal denoising loop -----
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        _is_rank0 = (not torch.distributed.is_initialized()
                     or torch.distributed.get_rank() == 0)
        _total_chunks = len(all_num_frames)

        for _chunk_idx, current_num_frames in enumerate(all_num_frames):
            # Per-chunk control signals
            if control_manager is not None:
                ctrl = control_manager.next(current_num_frames)
                combined_action_ids_input = ctrl["combined_action_ids"]
                prope_ctx = ctrl["prope_context"]
            else:
                if combined_action_ids is not None:
                    combined_action_ids_input = combined_action_ids[
                        :, cache_start_frame - num_input_frames:cache_start_frame
                        + current_num_frames - num_input_frames]
                else:
                    combined_action_ids_input = None
                prope_ctx = prope_context

            noisy_input = noise[
                :, cache_start_frame - num_input_frames:
                cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # ----- Step 3.1: Spatial denoising loop -----
            sample_scheduler = self._initialize_sample_scheduler(
                noise, num_inference_steps=num_inference_steps)
            # Log-bias is applied during Step 3.1 attention reads (the
            # compressed cache from prior chunks is what they read).  For Step
            # 3.3 (the per-chunk clean KV write) we deliberately DO NOT apply
            # log-bias — the chunk's own K/V is what is being written to the
            # cache slot, the attention output of Step 3.3 is unused.
            with self._log_bias_for_current_cache(chunk_tokens=chunk_tokens,
                                                   device=noise.device,
                                                   dtype=noise.dtype):
                for _, t in enumerate(tqdm(sample_scheduler.timesteps,
                                            disable=(not _is_rank0))):
                    latent_model_input = latents
                    timestep = t * torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device, dtype=torch.float32,
                    )

                    flow_pred_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        combined_action_ids=combined_action_ids_input,
                        prope_context=prope_ctx,
                    )
                    if abs(self.args.guidance_scale - 1.0) < 1e-6:
                        # LoRA/DMD already distills the guidance -> CFG is useless; skipping the uncond branch halves memory/KV. flow_pred==cond.
                        flow_pred = flow_pred_cond
                    else:
                        flow_pred_uncond, _ = self.generator(
                            noisy_image_or_video=latent_model_input,
                            conditional_dict=unconditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=current_start_frame * self.frame_seq_length,
                            cache_start=cache_start_frame * self.frame_seq_length,
                            combined_action_ids=combined_action_ids_input,
                            prope_context=prope_ctx,
                        )
                        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                            flow_pred_cond - flow_pred_uncond)
                    latents = sample_scheduler.step(
                        flow_pred, t, latents, return_dict=False)[0]

            # Step 3.2
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: clean-KV write (no log-bias).
            self._call_generator_step33(
                latents, conditional_dict, unconditional_dict, timestep * 0,
                current_start_frame, cache_start_frame, prope_ctx,
                combined_action_ids_input=None,
            )

            # ----- Compression hook (after Step 3.3) -----
            self._compress_post_step33(prope_ctx)

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames
            if _is_rank0:
                le_pos = int(self.kv_cache_pos[0]["local_end_index"].item())
                le_neg = int(self.kv_cache_neg[0]["local_end_index"].item())
                if self._registry is not None:
                    slot_count = self._registry.num_slots()
                    last_action = (self._registry.events[-1].action
                                   if self._registry.events else 'n/a')
                else:
                    slot_count = le_pos // chunk_tokens if chunk_tokens else -1
                    last_action = self.kvc_cfg.policy
                print(
                    f"[infer-kvc] chunk {_chunk_idx+1}/{_total_chunks} "
                    f"frames={current_num_frames} slots={slot_count} "
                    f"kv_local_end pos={le_pos} neg={le_neg} "
                    f"last_action={last_action}",
                    flush=True,
                )

        # Optional: dump compression log
        if self.kvc_cfg.log_path is not None and _is_rank0 and self._registry is not None:
            self._dump_compression_log()

        # Step 4: Decode
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_generator_step33(
        self, latents, conditional_dict, unconditional_dict, timestep,
        current_start_frame, cache_start_frame, prope_ctx,
        combined_action_ids_input=None,
    ):
        """Step 3.3 of the base pipeline — runs the generator twice (cond and
        uncond) at ``timestep=0`` to push the *clean* K/V into the cache.

        Wrapped in the same ``_INSPECT_CAPTURE_NOW`` flag dance as the base
        method (so inspect-mode trainer monkey-patching, if active, still
        captures only this chunk's clean KV).
        """
        import wan_5b.modules.action_model as _am_for_inspect
        _am_for_inspect._INSPECT_CAPTURE_NOW = True
        try:
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                prope_context=prope_ctx,
                **({"combined_action_ids": combined_action_ids_input}
                   if combined_action_ids_input is not None else {}),
            )
            if abs(self.args.guidance_scale - 1.0) >= 1e-6:
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx,
                    **({"combined_action_ids": combined_action_ids_input}
                       if combined_action_ids_input is not None else {}),
                )
        finally:
            _am_for_inspect._INSPECT_CAPTURE_NOW = False

    def _apply_compress_v1(self, k_pos, v_pos, k_neg, v_neg):
        """window/naive/select via per-layer CompressCache.  Keep-decisions are SHARED
        across layers+branches (one importance from pos layer-0 K) so every cache stays
        aligned (identical local_end_index).  k_*[L] = [B,ct,H,D] with B=1 at inference."""
        from longctx.inference.compress_cache_v1 import key_distinctiveness
        imp = None
        if self.kvc_cfg.policy == "select":
            imp = key_distinctiveness(k_pos[0][0])      # [ct] from pos layer-0 K
        new_le = None
        for L in range(self.num_transformer_blocks):
            self._cc_pos[L].add_chunk(k_pos[L][0], v_pos[L][0], imp)
            le_p = self._cc_pos[L].write_into_cache(self.kv_cache_pos[L])
            if k_neg is not None:   # CFG on; guidance=1.0 (i2v) runs pos branch only
                self._cc_neg[L].add_chunk(k_neg[L][0], v_neg[L][0], imp)
                le_n = self._cc_neg[L].write_into_cache(self.kv_cache_neg[L])
                assert le_p == le_n, f"v1 compress branch desync L={L}: {le_p} vs {le_n}"
            if new_le is None:
                new_le = le_p
            else:
                assert le_p == new_le, f"v1 compress layer desync L={L}: {le_p} vs {new_le}"
            self.kv_cache_pos[L]["local_end_index"].fill_(le_p)
            if k_neg is not None:
                self.kv_cache_neg[L]["local_end_index"].fill_(le_n)
        self._kvc_chunk_counter += 1
        return new_le

    def _apply_landmark(self, k_pos, v_pos, k_neg, v_neg, chunk_pose):
        """v15a/v15b via per-layer LandmarkCache. Pose decisions are deterministic given
        the (shared) chunk_pose → every layer/branch makes identical bank/retrieval choices
        → caches stay aligned. (retrieval target = current chunk_pose, ~= next FOV.)"""
        new_le = None
        # [rope-remap] the RoPE start frame used when this chunk's K is written in Step 3.3:
        # remap on = the virtual frame computed during assembly (rope_start_frame);
        # off = None (not involved in any computation).
        import os as _os
        _remap = _os.environ.get("BANKV2_ROPE_REMAP", "0") == "1"
        for L in range(self.num_transformer_blocks):
            if _remap and not hasattr(self._lm_pos[L], "_freqs_t"):
                _f = self.generator.model.freqs      # [1024,64] complex
                self._lm_pos[L].set_rope_freqs_t(_f[:, :22].to(self.kv_cache_pos[L]["k"].device))
                if k_neg is not None:
                    self._lm_neg[L].set_rope_freqs_t(_f[:, :22].to(self.kv_cache_neg[L]["k"].device))
            _pf = None
            if _remap:
                rsf = self.kv_cache_pos[L].get("rope_start_frame")
                _pf = int(rsf.item()) if rsf is not None else int(
                    self.kv_cache_pos[L]["global_end_index"].item() // self.frame_seq_length
                    - k_pos[L][0].shape[0] // self.frame_seq_length)
            R_p = self._lm_pos[L].add_chunk(k_pos[L][0], v_pos[L][0], chunk_pose, chunk_pose,
                                            phase_f=_pf)
            le_p = self._lm_pos[L].write_into_cache(self.kv_cache_pos[L], R_p)
            if _remap:
                nxt = torch.tensor(self._lm_pos[L].last_rope_start_f, dtype=torch.long,
                                   device=self.kv_cache_pos[L]["global_end_index"].device)
                self.kv_cache_pos[L]["rope_start_frame"] = nxt
            if k_neg is not None:   # CFG on
                R_n = self._lm_neg[L].add_chunk(k_neg[L][0], v_neg[L][0], chunk_pose, chunk_pose,
                                                phase_f=_pf)
                le_n = self._lm_neg[L].write_into_cache(self.kv_cache_neg[L], R_n)
                if _remap:
                    self.kv_cache_neg[L]["rope_start_frame"] = nxt.clone()
                assert le_p == le_n, f"landmark branch desync L={L}: {le_p} vs {le_n}"
                self.kv_cache_neg[L]["local_end_index"].fill_(le_n)
            if new_le is None:
                new_le = le_p
            else:
                assert le_p == new_le, f"landmark layer desync L={L}: {le_p} vs {new_le}"
            self.kv_cache_pos[L]["local_end_index"].fill_(le_p)
        self._kvc_chunk_counter += 1
        return new_le

    def _compress_post_step33(self, prope_ctx: torch.Tensor) -> None:
        """Run the compression decision + apply for the chunk just written.

        Assumes the chunk's K/V was written into the cache at
        ``[local_end_index - num_frame_per_block * frame_seq_length :
          local_end_index]`` (the standard base-pipeline layout).
        """
        chunk_tokens = self.num_frame_per_block * self.frame_seq_length
        le_pos = int(self.kv_cache_pos[0]["local_end_index"].item())
        le_neg = int(self.kv_cache_neg[0]["local_end_index"].item())
        _cfg_off = abs(self.args.guidance_scale - 1.0) < 1e-6
        # Sanity: both branches must be in lockstep (skip when CFG off: neg branch unused)
        assert _cfg_off or le_pos == le_neg, (
            f"KV cache cond/uncond out of sync: pos={le_pos} neg={le_neg}"
        )
        chunk_start = le_pos - chunk_tokens
        if chunk_start < 0:
            raise RuntimeError(
                f"_compress_post_step33: chunk_start={chunk_start} < 0, "
                f"local_end_index={le_pos} chunk_tokens={chunk_tokens}.  "
                f"This means Step 3.3 didn't write a full chunk to the cache."
            )

        # Pull chunk K/V from each layer
        k_pos, v_pos = self._extract_chunk_kv_from_cache(
            self.kv_cache_pos, chunk_start, le_pos
        )
        if _cfg_off:
            k_neg, v_neg = None, None
        else:
            k_neg, v_neg = self._extract_chunk_kv_from_cache(
                self.kv_cache_neg, chunk_start, le_neg
            )

        # v1 policies (window/naive/select): no pose needed.
        if self.kvc_cfg.policy in ("window", "naive", "select"):
            self._apply_compress_v1(k_pos, v_pos, k_neg, v_neg)
            return

        # v1.5 landmark policies: pose-based bank + retrieval.
        if self.kvc_cfg.policy in ("consol", "consol_div", "v15a", "v15b", "v15c"):
            chunk_pose = self._compute_chunk_pose(prope_ctx, self.num_frame_per_block)
            self._apply_landmark(k_pos, v_pos, k_neg, v_neg, chunk_pose)
            return

        # Chunk-level pose (one (4,4) for the merge decision)
        chunk_pose = self._compute_chunk_pose(prope_ctx, self.num_frame_per_block)

        # Decide + apply + rewrite cache
        new_le = self._apply_compression_for_chunk(
            chunk_pose, k_pos, v_pos, k_neg, v_neg
        )
        # global_end_index keeps growing monotonically (it's the "true" stream
        # position, used by next chunk's RoPE start_frame).  We update it to
        # match the current_start frame logic — which the action_model
        # inference branch uses via ``cur_end = current_start + num_new``.
        # In the base pipeline, ``global_end_index`` ends up equal to
        # ``current_start + num_new`` after each chunk; we leave that
        # untouched and only patch ``local_end_index`` to reflect the
        # *physical* slot count in the cache.
        # (The base inference branch derives local_end_index from
        # ``kv_cache["local_end_index"].item() + cur_end - global_end_index``
        # — so on the next chunk it will see local_end_index ≈ compressed-size.)
        # We deliberately keep global_end_index in sync with the
        # uncompressed stream so the next-chunk RoPE/PRoPE math still uses
        # the correct absolute frame indices.
        _ = new_le

    @contextmanager
    def _log_bias_for_current_cache(
        self, chunk_tokens: int, device: torch.device, dtype: torch.dtype,
    ):
        """Activate the log-bias monkey-patch using the current cache's
        per-slot visit counts.  Cond and uncond have *identical* visit counts
        (slot decisions are shared), so a single bias tensor is correct for
        both branches.

        No-op when log_bias is disabled or all slots have visit_count == 1.
        """
        if not self.kvc_cfg.log_bias:
            yield
            return
        # Bias is per-layer; for the inference attention call site we apply
        # the same bias regardless of layer index, because slot visit counts
        # are identical across layers (the registry is shared).  Use layer 0
        # as representative.
        if not self._layer_caches_pos:
            yield
            return
        bias = self._build_log_bias_tensor(
            self._layer_caches_pos[0], chunk_tokens=chunk_tokens,
            device=device, dtype=dtype,
        )
        with _kvc_log_bias_active(bias):
            yield

    def _dump_compression_log(self) -> None:
        import json
        if self._registry is None:
            return
        out = {
            "config": {
                "enabled": self.kvc_cfg.enabled,
                "mode": self.kvc_cfg.mode,
                "epsilon": self.kvc_cfg.epsilon,
                "n_sink": self.kvc_cfg.n_sink,
                "budget_chunks": self.kvc_cfg.budget_chunks,
                "k_renorm": self.kvc_cfg.k_renorm,
                "log_bias": self.kvc_cfg.log_bias,
                "pose_noise_sigma": self.kvc_cfg.pose_noise_sigma,
            },
            "events": [
                {
                    "chunk_idx": e.chunk_idx,
                    "action": e.action,
                    "slot_idx": e.slot_idx,
                    "pose_distance": e.pose_distance_at_decision,
                    "num_slots_after": e.num_slots_after,
                }
                for e in self._registry.events
            ],
            "final_slots": self._registry.slot_summary(),
            "compression_ratio": (
                self._kvc_chunk_counter / max(1, self._registry.num_slots())
            ),
        }
        os.makedirs(os.path.dirname(self.kvc_cfg.log_path), exist_ok=True)
        with open(self.kvc_cfg.log_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[KVCompression] Dumped log → {self.kvc_cfg.log_path}")

    def clear_cache(self) -> None:
        super().clear_cache()
        self._registry = None
        self._layer_caches_pos = []
        self._layer_caches_neg = []
