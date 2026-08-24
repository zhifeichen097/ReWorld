"""PoseAwareKVCacheManager — sink + pose_cons KV compression for AR video inference.

Algorithm:

    On each new chunk N arriving with pose P_N and KV vectors (K_N, V_N):

      if chunk_index < n_sink:
        # Always retain the sink chunks
        allocate fresh slot, register pose

      else:
        # Look for redundancy with any non-sink slot
        d_min, slot_min = min over non-sink slots S: pose_distance(P_N, S.pose)
        if d_min < epsilon:
          # Merge new K,V into slot_min via running mean
          c = slot_min.visit_count
          slot_min.k = (c / (c+1)) * slot_min.k + (1 / (c+1)) * K_N
          slot_min.v = (c / (c+1)) * slot_min.v + (1 / (c+1)) * V_N
          slot_min.pose = (c / (c+1)) * slot_min.pose + (1 / (c+1)) * P_N
          slot_min.visit_count += 1
        else:
          if num_slots == budget:
            # Evict — drop the most-recent non-sink slot with lowest visit_count
            evict_slot()
          allocate fresh slot, register pose

This module is **standalone** — it does not modify the live inference pipeline.
It is intended for offline testing + later integration into
`pipeline/causal_diffusion_inference.py`.

For now, the manager assumes a **single transformer block** of cache. The real
integration will instantiate one manager per layer (or share the slot decisions
across layers, since pose is shared, but per-layer KV running-means).

Usage:
    mgr = PoseAwareKVCacheManager(
        num_blocks=30, num_heads=24, head_dim=128, frame_seq_len=1200,
        chunk_size=4, budget_chunks=20, n_sink=4, epsilon=0.5, device="cuda",
    )
    for chunk_idx, (c2w, k_chunk, v_chunk) in enumerate(stream):
        action, slot_idx = mgr.add_chunk(chunk_idx, c2w, k_chunk, v_chunk, layer=0)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np
import torch


def pose_distance(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    """Pose distance: sqrt(dxyz^2 + (dyaw/60)^2).

    c2w_a, c2w_b: (4, 4) numpy arrays
    """
    dxyz = float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))
    # Yaw from rotation matrix: extract from y-axis euler
    # Equivalent to scipy Rotation.from_matrix(...).as_euler("yxz", degrees=True)[0]
    # Hard-coded simple yaw extraction (works when roll≈0):
    yaw_a = math.degrees(math.atan2(c2w_a[0, 2], c2w_a[2, 2]))
    yaw_b = math.degrees(math.atan2(c2w_b[0, 2], c2w_b[2, 2]))
    dyaw_raw = abs(yaw_a - yaw_b) % 360.0
    dyaw = min(dyaw_raw, 360.0 - dyaw_raw) / 60.0
    return math.sqrt(dxyz ** 2 + dyaw ** 2)


@dataclass
class Slot:
    """One pose-aware cache slot, per-layer.

    is_sink: True if this slot is in the sink region (never evicted, never merged into).
    pose: (4, 4) centroid pose, updated on merge.
    visit_count: number of chunks merged into this slot.
    k: (batch, chunk_tokens, num_heads, head_dim) running-mean K.
    v: same shape.
    born_chunk: original chunk index (for debugging / eviction tie-break).
    last_seen_chunk: most-recent chunk index that contributed to this slot.
    """
    is_sink: bool
    pose: np.ndarray
    visit_count: int
    k: torch.Tensor
    v: torch.Tensor
    born_chunk: int
    last_seen_chunk: int


@dataclass
class CompressionEvent:
    """Recorded for analysis. action ∈ {sink_alloc, fresh_alloc, merge, evict_then_alloc}."""
    chunk_idx: int
    action: str
    slot_idx: int
    merged_into_pose: Optional[np.ndarray] = None
    evicted_slot_pose: Optional[np.ndarray] = None
    pose_distance_at_decision: Optional[float] = None
    num_slots_after: int = 0


@dataclass
class SlotDecision:
    """Pose-only decision describing what to do with the incoming chunk.

    Produced by ``PoseDecisionRegistry.decide(...)`` and applied by per-layer /
    per-branch ``PoseAwareLayerCache.apply_decision(...)``.  The decision is
    *pose-only*; the same decision is shared across (a) the 30 transformer
    layers and (b) the cond / uncond branches, because pose is invariant w.r.t.
    both.  Only the K/V running-mean update is per-(layer, branch).

    Fields
    ------
    action :
        One of ``"sink_alloc"`` (always for the first ``n_sink`` chunks),
        ``"fresh_alloc"`` (no pose-redundant slot found, room available),
        ``"merge"`` (merge into ``target_slot_idx``), or ``"evict_then_alloc"``
        (budget full, evict ``evict_slot_idx`` then allocate at end).
    target_slot_idx :
        For ``"merge"``: slot to merge into.
        For ``"sink_alloc"`` / ``"fresh_alloc"``: ``None`` — the new slot is
        appended at the tail and gets index ``len(slots)``.
        For ``"evict_then_alloc"``: ``None`` (new slot is appended after evict).
    evict_slot_idx :
        For ``"evict_then_alloc"``: slot to remove before allocating.
        ``None`` otherwise.
    target_visit_count_before :
        Visit count of ``target_slot_idx`` *before* this merge — needed by
        ``apply_decision`` to compute the running-mean weight
        ``c / (c+1) * old + 1 / (c+1) * new``.  ``None`` when not merging.
    """
    chunk_idx: int
    action: str
    target_slot_idx: Optional[int] = None
    evict_slot_idx: Optional[int] = None
    target_visit_count_before: Optional[int] = None
    pose_distance_at_decision: Optional[float] = None
    new_pose: Optional[np.ndarray] = None


class PoseAwareKVCacheManager:
    """Manages sink + pose_redundancy_consolidation slots for ONE transformer layer.

    For multi-layer cache, instantiate one manager per layer but share pose decisions
    by using the same `decide(c2w)` call across layers — see `add_chunk_multi_layer`.
    """

    def __init__(
        self,
        chunk_tokens: int,          # num_frame_per_block * frame_seq_len
        num_heads: int,
        head_dim: int,
        budget_chunks: int,
        n_sink: int = 4,
        epsilon: float = 0.5,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.chunk_tokens = chunk_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.budget_chunks = budget_chunks
        self.n_sink = n_sink
        self.epsilon = epsilon
        self.device = device
        self.dtype = dtype
        self.slots: List[Slot] = []
        self.events: List[CompressionEvent] = []
        self._chunk_counter = 0

    # ------------------ Slot allocation logic ------------------

    def _find_merge_target(self, pose: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Return (slot_idx, distance) of the nearest non-sink slot within epsilon, or (None, None).
        """
        best_idx = None
        best_d = None
        for i, s in enumerate(self.slots):
            if s.is_sink:
                continue
            d = pose_distance(pose, s.pose)
            if d < self.epsilon and (best_d is None or d < best_d):
                best_idx = i
                best_d = d
        return best_idx, best_d

    def _evict_one(self) -> Optional[int]:
        """Evict the non-sink slot with the lowest visit_count, tiebreak by lowest born_chunk
        (oldest slot wins eviction if tied — i.e., evict older lonely slots first; recency wins).

        Returns evicted slot index, or None if no eviction was possible (only sink slots).

        Design note: this is intentionally NOT FIFO. We want to preserve slots that have
        accumulated many visits (they encode a frequently-revisited location), and drop
        slots that were "alloc'd once and never returned to" — those are likely random walk
        excursions, not memorable locations.
        """
        # Search for lowest visit_count non-sink slot
        candidates = [(i, s) for i, s in enumerate(self.slots) if not s.is_sink]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1].visit_count, x[1].born_chunk))
        evict_idx = candidates[0][0]
        evicted_pose = self.slots[evict_idx].pose.copy()
        # Pop
        self.slots.pop(evict_idx)
        # Record (will be filled in by caller)
        return evict_idx

    # ------------------ Public single-layer API ------------------

    def add_chunk(
        self,
        chunk_idx: int,
        c2w_pose: np.ndarray,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> Tuple[str, int]:
        """Add a new chunk's KV to the cache. Returns (action, slot_idx).

        k_chunk, v_chunk shape: (batch, chunk_tokens, num_heads, head_dim)

        Single-layer interface. For multi-layer use, call decide() once and then
        apply_decision() per layer (so all layers share the same slot structure).
        """
        evt = CompressionEvent(chunk_idx=chunk_idx, action="?", slot_idx=-1)

        if chunk_idx < self.n_sink:
            # SINK: always allocate a new slot, mark as sink
            slot = Slot(
                is_sink=True, pose=c2w_pose.copy(), visit_count=1,
                k=k_chunk.detach().clone(), v=v_chunk.detach().clone(),
                born_chunk=chunk_idx, last_seen_chunk=chunk_idx,
            )
            self.slots.append(slot)
            evt.action = "sink_alloc"
            evt.slot_idx = len(self.slots) - 1
            evt.num_slots_after = len(self.slots)
            self.events.append(evt)
            return evt.action, evt.slot_idx

        # Non-sink: check for merge candidate
        merge_idx, merge_d = self._find_merge_target(c2w_pose)
        if merge_idx is not None:
            # Merge: running mean of K, V, pose
            # IMPORTANT: per experiment 06, naive running mean causes K magnitude
            # to decay as 1/sqrt(visit_count) for iid K's. We re-normalize K to the
            # mean of source norms to preserve retrieval signal magnitude.
            s = self.slots[merge_idx]
            c = s.visit_count
            k_old_norm = s.k.norm()
            k_new_norm = k_chunk.norm()
            target_norm = (c * k_old_norm + k_new_norm) / (c + 1)

            s.k = (c / (c + 1)) * s.k + (1 / (c + 1)) * k_chunk
            s.v = (c / (c + 1)) * s.v + (1 / (c + 1)) * v_chunk
            # Renormalize K to preserve attention dot-product magnitude
            cur_norm = s.k.norm()
            if cur_norm > 1e-6:
                s.k = s.k * (target_norm / cur_norm)

            s.pose = (c / (c + 1)) * s.pose + (1 / (c + 1)) * c2w_pose
            s.visit_count = c + 1
            s.last_seen_chunk = chunk_idx
            evt.action = "merge"
            evt.slot_idx = merge_idx
            evt.merged_into_pose = s.pose.copy()
            evt.pose_distance_at_decision = merge_d
            evt.num_slots_after = len(self.slots)
            self.events.append(evt)
            return evt.action, evt.slot_idx

        # Fresh allocation needed. If budget exceeded, evict first.
        if len(self.slots) >= self.budget_chunks:
            evict_idx = self._evict_one()
            if evict_idx is None:
                # All slots are sink — can't evict. Skip this chunk (no slot available).
                # In practice budget should be > n_sink so this never happens.
                evt.action = "skip_no_room"
                evt.num_slots_after = len(self.slots)
                self.events.append(evt)
                return evt.action, -1
            evt.action = "evict_then_alloc"
        else:
            evt.action = "fresh_alloc"

        slot = Slot(
            is_sink=False, pose=c2w_pose.copy(), visit_count=1,
            k=k_chunk.detach().clone(), v=v_chunk.detach().clone(),
            born_chunk=chunk_idx, last_seen_chunk=chunk_idx,
        )
        self.slots.append(slot)
        evt.slot_idx = len(self.slots) - 1
        evt.num_slots_after = len(self.slots)
        self.events.append(evt)
        return evt.action, evt.slot_idx

    # ------------------ Snapshots / queries ------------------

    def num_slots(self) -> int:
        return len(self.slots)

    def num_sink_slots(self) -> int:
        return sum(1 for s in self.slots if s.is_sink)

    def num_nonsink_slots(self) -> int:
        return sum(1 for s in self.slots if not s.is_sink)

    def slot_summary(self) -> List[dict]:
        return [
            dict(
                idx=i, is_sink=s.is_sink, visit_count=s.visit_count,
                born_chunk=s.born_chunk, last_seen_chunk=s.last_seen_chunk,
                xyz=s.pose[:3, 3].tolist(),
                yaw_deg=math.degrees(math.atan2(s.pose[0, 2], s.pose[2, 2])),
            )
            for i, s in enumerate(self.slots)
        ]

    def get_compact_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all retained slots into a single (batch, total_tokens, H, D) tensor pair.

        This is what the attention layer would receive as its KV cache after compression.
        """
        k_list = [s.k for s in self.slots]
        v_list = [s.v for s in self.slots]
        return torch.cat(k_list, dim=1), torch.cat(v_list, dim=1)


# ============================================================================
# Multi-layer / cond+uncond split: pose-only decision registry + per-layer
# K/V running-mean state.  Designed for direct integration into
# ``pipeline/causal_diffusion_inference_compressed.py``.
# ============================================================================


class PoseDecisionRegistry:
    """Pose-only slot bookkeeping shared across layers and cond/uncond.

    Stores **only** the slot pose history + visit counts — *no* K/V tensors.
    Every layer/branch attaches its own ``PoseAwareLayerCache`` and applies the
    same ``SlotDecision`` to its (pre-allocated) KV cache tensor.

    Rationale (designs/16 §5.2):

      * Cond and uncond have different K because the text prompt influences
        K via cross-attention residual and (indirectly) through self-attn
        dynamics.  We must therefore maintain two independent K/V running
        means.
      * However the *slot decision* (merge vs new vs evict) depends only on
        pose, which is identical for cond and uncond.  Sharing the decision
        avoids the two branches drifting into different slot layouts —
        which would be impossible to reconcile, since the same `local_end_index`
        is used by both halves of the pipeline downstream.

    Slot indexing:

      * Slots are kept in the order they were allocated.  Sink slots are
        always at indices ``[0, n_sink)``.  Non-sink slots follow.
      * Eviction removes a slot index in-place and *shifts higher indices
        down by one*.  Layer caches must mirror this shift (handled by
        ``apply_decision``).
    """

    def __init__(self, budget_chunks: int, n_sink: int, epsilon: float):
        if budget_chunks <= n_sink:
            raise ValueError(
                f"budget_chunks ({budget_chunks}) must be strictly > n_sink "
                f"({n_sink}), otherwise non-sink eviction is impossible."
            )
        self.budget_chunks = budget_chunks
        self.n_sink = n_sink
        self.epsilon = epsilon
        # Per-slot metadata.  Mirror layout of every ``PoseAwareLayerCache``.
        self._slot_poses: List[np.ndarray] = []      # (4, 4) centroid
        self._slot_visit: List[int] = []             # running merge count
        self._slot_is_sink: List[bool] = []
        self._slot_born: List[int] = []              # original chunk index
        self.events: List[CompressionEvent] = []

    # -------- queries --------

    def num_slots(self) -> int:
        return len(self._slot_poses)

    def slot_summary(self) -> List[dict]:
        return [
            dict(
                idx=i,
                is_sink=self._slot_is_sink[i],
                visit_count=self._slot_visit[i],
                born_chunk=self._slot_born[i],
                xyz=self._slot_poses[i][:3, 3].tolist(),
            )
            for i in range(len(self._slot_poses))
        ]

    # -------- decision --------

    def _find_merge_target(self, pose: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_idx, best_d = None, None
        for i in range(len(self._slot_poses)):
            if self._slot_is_sink[i]:
                continue
            d = pose_distance(pose, self._slot_poses[i])
            if d < self.epsilon and (best_d is None or d < best_d):
                best_idx, best_d = i, d
        return best_idx, best_d

    def _pick_evict(self) -> Optional[int]:
        # Lowest visit_count among non-sink, tiebreak by lowest born_chunk
        # (consistent with PoseAwareKVCacheManager._evict_one).
        cands = [
            (i, self._slot_visit[i], self._slot_born[i])
            for i in range(len(self._slot_poses))
            if not self._slot_is_sink[i]
        ]
        if not cands:
            return None
        cands.sort(key=lambda t: (t[1], t[2]))
        return cands[0][0]

    def decide(self, chunk_idx: int, c2w_pose: np.ndarray) -> SlotDecision:
        """Return what should happen for the incoming chunk.

        IMPORTANT: ``decide`` does *not* mutate registry state — only
        ``commit`` does.  The caller is expected to (a) call ``decide`` once,
        (b) apply the decision to each ``PoseAwareLayerCache``, then
        (c) call ``commit`` to finalize the pose / visit / slot list.
        Splitting decide + commit ensures all layers + branches see the
        same pre-state when applying the decision.
        """
        # SINK
        if chunk_idx < self.n_sink:
            return SlotDecision(
                chunk_idx=chunk_idx,
                action="sink_alloc",
                new_pose=c2w_pose.copy(),
            )

        # Try merge
        merge_idx, merge_d = self._find_merge_target(c2w_pose)
        if merge_idx is not None:
            return SlotDecision(
                chunk_idx=chunk_idx,
                action="merge",
                target_slot_idx=merge_idx,
                target_visit_count_before=self._slot_visit[merge_idx],
                pose_distance_at_decision=merge_d,
                new_pose=c2w_pose.copy(),
            )

        # Fresh alloc, possibly evicting first
        if len(self._slot_poses) >= self.budget_chunks:
            evict_idx = self._pick_evict()
            if evict_idx is None:
                # Should not happen since we enforce budget > n_sink in __init__
                return SlotDecision(chunk_idx=chunk_idx, action="skip_no_room")
            return SlotDecision(
                chunk_idx=chunk_idx,
                action="evict_then_alloc",
                evict_slot_idx=evict_idx,
                new_pose=c2w_pose.copy(),
            )
        return SlotDecision(
            chunk_idx=chunk_idx,
            action="fresh_alloc",
            new_pose=c2w_pose.copy(),
        )

    def commit(self, decision: SlotDecision) -> int:
        """Apply ``decision`` to registry state.  Returns the *final slot index*
        of the chunk (i.e., where its K/V now lives after this commit)."""
        a = decision.action
        if a == "sink_alloc":
            self._slot_poses.append(decision.new_pose.copy())
            self._slot_visit.append(1)
            self._slot_is_sink.append(True)
            self._slot_born.append(decision.chunk_idx)
            final_idx = len(self._slot_poses) - 1
            self.events.append(CompressionEvent(
                chunk_idx=decision.chunk_idx, action=a, slot_idx=final_idx,
                num_slots_after=len(self._slot_poses)))
            return final_idx
        if a == "merge":
            i = decision.target_slot_idx
            c = decision.target_visit_count_before
            self._slot_poses[i] = (c / (c + 1)) * self._slot_poses[i] + (1 / (c + 1)) * decision.new_pose
            self._slot_visit[i] = c + 1
            self.events.append(CompressionEvent(
                chunk_idx=decision.chunk_idx, action=a, slot_idx=i,
                pose_distance_at_decision=decision.pose_distance_at_decision,
                num_slots_after=len(self._slot_poses)))
            return i
        if a == "evict_then_alloc":
            i = decision.evict_slot_idx
            self._slot_poses.pop(i)
            self._slot_visit.pop(i)
            self._slot_is_sink.pop(i)
            self._slot_born.pop(i)
            # Then alloc at tail
            self._slot_poses.append(decision.new_pose.copy())
            self._slot_visit.append(1)
            self._slot_is_sink.append(False)
            self._slot_born.append(decision.chunk_idx)
            final_idx = len(self._slot_poses) - 1
            self.events.append(CompressionEvent(
                chunk_idx=decision.chunk_idx, action=a, slot_idx=final_idx,
                num_slots_after=len(self._slot_poses)))
            return final_idx
        if a == "fresh_alloc":
            self._slot_poses.append(decision.new_pose.copy())
            self._slot_visit.append(1)
            self._slot_is_sink.append(False)
            self._slot_born.append(decision.chunk_idx)
            final_idx = len(self._slot_poses) - 1
            self.events.append(CompressionEvent(
                chunk_idx=decision.chunk_idx, action=a, slot_idx=final_idx,
                num_slots_after=len(self._slot_poses)))
            return final_idx
        if a == "skip_no_room":
            self.events.append(CompressionEvent(
                chunk_idx=decision.chunk_idx, action=a, slot_idx=-1,
                num_slots_after=len(self._slot_poses)))
            return -1
        raise ValueError(f"unknown action: {a}")


class PoseAwareLayerCache:
    """Per-(layer, branch) K/V running-mean state, kept *in lockstep* with
    a ``PoseDecisionRegistry``.

    Internally tracks per-slot ``visit_count`` *for the K-norm renorm trick*
    (independent of the registry, although values match — kept duplicated to
    avoid coupling).  Also tracks how many tokens correspond to each slot —
    in our setup all slots have ``chunk_tokens`` (= ``num_frame_per_block ×
    frame_seq_len``) tokens, but we keep the bookkeeping explicit so a future
    "first chunk is independent" model variant doesn't silently break.

    Storage uses **Python lists of tensors** rather than a pre-allocated big
    tensor.  This is fine for inference (no autograd, no perf-critical
    inner loop here); the data is *copied* into the pre-allocated upstream
    ``kv_cache["k"], kv_cache["v"]`` tensor on every ``write_into_cache`` call
    so attention forward continues to see a single contiguous slab.
    """

    def __init__(
        self,
        chunk_tokens: int,
        num_heads: int,
        head_dim: int,
        k_renorm: bool = True,
    ):
        self.chunk_tokens = chunk_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_renorm = k_renorm
        # Parallel lists, one entry per slot.  Each entry shape:
        # K, V: (batch, chunk_tokens, num_heads, head_dim)
        # visit_count: int, K-running-mean count for renorm math
        self._slot_k: List[torch.Tensor] = []
        self._slot_v: List[torch.Tensor] = []
        self._slot_visit: List[int] = []

    def num_slots(self) -> int:
        return len(self._slot_k)

    def apply_decision(
        self,
        decision: SlotDecision,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Mirror the registry decision on this layer's K/V state.

        ``k_new``, ``v_new`` shape: ``(batch, chunk_tokens, num_heads, head_dim)``.
        For ``sink_alloc`` / ``fresh_alloc`` / ``evict_then_alloc`` we store
        a fresh detached clone (so the caller can free the upstream tensor).
        For ``merge`` we update the running mean of K and V on the target slot
        and re-normalize K to preserve attention dot-product magnitude
        (see ``PoseAwareKVCacheManager.add_chunk``).
        """
        a = decision.action
        if a in ("sink_alloc", "fresh_alloc"):
            self._slot_k.append(k_new.detach().clone())
            self._slot_v.append(v_new.detach().clone())
            self._slot_visit.append(1)
            return
        if a == "evict_then_alloc":
            i = decision.evict_slot_idx
            del self._slot_k[i]
            del self._slot_v[i]
            del self._slot_visit[i]
            self._slot_k.append(k_new.detach().clone())
            self._slot_v.append(v_new.detach().clone())
            self._slot_visit.append(1)
            return
        if a == "merge":
            i = decision.target_slot_idx
            c = decision.target_visit_count_before
            # V: plain running mean — preserves softmax-weighted sum semantics
            # when V's are similar (which they are for nearby poses).
            self._slot_v[i] = (c / (c + 1)) * self._slot_v[i] + (1 / (c + 1)) * v_new

            # K: running mean + renorm to preserve attention logit magnitude
            # so the merged slot is still "findable" by future queries.
            if self.k_renorm:
                k_old_norm = self._slot_k[i].norm()
                k_new_norm = k_new.norm()
                target_norm = (c * k_old_norm + k_new_norm) / (c + 1)
                self._slot_k[i] = (c / (c + 1)) * self._slot_k[i] + (1 / (c + 1)) * k_new
                cur_norm = self._slot_k[i].norm()
                if cur_norm.item() > 1e-6:
                    self._slot_k[i] = self._slot_k[i] * (target_norm / cur_norm)
            else:
                self._slot_k[i] = (c / (c + 1)) * self._slot_k[i] + (1 / (c + 1)) * k_new
            self._slot_visit[i] = c + 1
            return
        if a == "skip_no_room":
            return
        raise ValueError(f"unknown action: {a}")

    def write_into_cache(
        self,
        kv_cache: dict,
    ) -> int:
        """Copy *all* slot K/V into the upstream pre-allocated ``kv_cache``
        starting at index 0, contiguously, and return the new
        ``local_end_index`` (= ``num_slots * chunk_tokens``).

        Why rewrite the whole cache rather than incremental updates: when a
        ``merge`` happens at slot ``i``, the *existing* K/V at offset
        ``i * chunk_tokens`` in the upstream cache is now stale.  Rather than
        track sparse rewrites, we just rewrite the contiguous front portion
        on every chunk — this is one BF16 mem-copy of ``num_slots * chunk_tokens *
        num_heads * head_dim`` elements per layer per branch.  At 48 latent
        chunks × budget 20 × chunk_tokens 1760 × num_heads 24 × head_dim 128 ×
        bf16 (2B) ≈ 5 MB per layer × 30 layers × 2 branches = 300 MB/chunk
        total, which is negligible compared to the 30 denoise-step attention
        FLOPs.

        The caller is responsible for updating ``kv_cache["local_end_index"]``
        and ``kv_cache["global_end_index"]`` afterwards.
        """
        if len(self._slot_k) == 0:
            return 0
        # Sanity: cache must be big enough for all slots
        total_tokens = len(self._slot_k) * self.chunk_tokens
        cache_capacity = kv_cache["k"].shape[1]
        if total_tokens > cache_capacity:
            raise RuntimeError(
                f"PoseAwareLayerCache.write_into_cache: total_tokens={total_tokens} "
                f"exceeds upstream cache capacity={cache_capacity}.  "
                f"Increase image_or_video_shape[1] or reduce budget_chunks "
                f"({len(self._slot_k)} slots × {self.chunk_tokens} chunk_tokens)."
            )

        # Block-copy each slot into its dense position.  Use slot index × chunk_tokens.
        for i, (k_s, v_s) in enumerate(zip(self._slot_k, self._slot_v)):
            start = i * self.chunk_tokens
            end = start + self.chunk_tokens
            kv_cache["k"][:, start:end].copy_(k_s)
            kv_cache["v"][:, start:end].copy_(v_s)

        return total_tokens

    def per_slot_visit_counts(self) -> List[int]:
        """Used by log-bias: how many original chunks were merged into each slot."""
        return list(self._slot_visit)
