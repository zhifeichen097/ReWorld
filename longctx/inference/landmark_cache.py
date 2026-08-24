"""v1.5 landmark cache — keep DISTINCT real chunks (no merge/pool) + pose-retrieval.

Bounded, all-GPU-friendly, real-time compatible. Two modes:
  mode 'a': bank cap K = retrieve_k (e.g. 6); attend ALL bank (no per-step retrieval).
            active = sink + recent + bank(all)  →  fixed 12-cache, simplest. Tests
            "keep distinct real chunks" alone.
  mode 'b': bank cap K large (e.g. 30); each step RETRIEVE top retrieve_k by pose to a
            target FOV. active = sink + recent + retrieved (pose-retrieval mode).

Bank maintenance = GREEDY pose ε-cover (NOT union-find → no forward-walk chaining):
  a chunk that ages out of recent → if pose within ε of an existing landmark: DROP
  (redundant, bump its visit); else ADD; if bank full: evict the MOST-REDUNDANT
  landmark (member of the closest pair) → keeps coverage, graceful forget.

Pose decisions are content/layer-agnostic → in the real pipeline they're computed ONCE
(shared across layers + cond/uncond) so every layer's KV cache stays aligned. Here we
co-locate pose+KV for the standalone smoke; the pipeline split mirrors PoseDecisionRegistry.
"""
import math
import numpy as np
import torch


def pose_dist(a, b):
    """SE3 pose distance = sqrt(|dxyz|^2 + (dyaw_deg/60)^2). a,b: (4,4)."""
    dxyz = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    ya = math.degrees(math.atan2(a[0, 2], a[2, 2]))
    yb = math.degrees(math.atan2(b[0, 2], b[2, 2]))
    d = abs(ya - yb) % 360.0
    dyaw = min(d, 360.0 - d) / 60.0
    return math.sqrt(dxyz * dxyz + dyaw * dyaw)


class Landmark:
    __slots__ = ("k", "v", "pose", "visit", "born")
    def __init__(self, k, v, pose, born):
        self.k, self.v, self.pose, self.visit, self.born = k, v, pose.copy(), 1, born


class LandmarkCache:
    def __init__(self, chunk_tokens, n_sink=1, recent_w=5, K=30, retrieve_k=6,
                 eps=0.5, mode="b", diverse=False):
        self.ct = chunk_tokens; self.n_sink = n_sink; self.recent_w = recent_w
        self.eps = eps; self.mode = mode; self.diverse = diverse
        self.retrieve_k = retrieve_k if mode == "b" else K
        self.K = retrieve_k if mode == "a" else K
        self.sink = []                       # list[Landmark]
        self.recent = []                     # ring of Landmark (full-res)
        self.bank = []                       # ≤K distinct Landmark
        self._n = 0
        self.events = []                     # for inspection

    def _nearest_bank(self, pose):
        if not self.bank:
            return None, None
        ds = [pose_dist(pose, lm.pose) for lm in self.bank]
        j = int(np.argmin(ds)); return j, ds[j]

    def _most_redundant_bank(self):
        # member of the closest pair (smallest pairwise pose_dist) → evict it
        best = (None, 1e18)
        for i in range(len(self.bank)):
            for j in range(i + 1, len(self.bank)):
                d = pose_dist(self.bank[i].pose, self.bank[j].pose)
                if d < best[1]:
                    best = (i, d)            # evict the earlier-born of the pair → i
        return best[0]

    def _bank_insert(self, lm):
        j, d = self._nearest_bank(lm.pose)
        if j is not None and d < self.eps:
            self.bank[j].visit += 1
            self.events.append(("drop_redundant", lm.born, j, round(d, 3))); return
        if len(self.bank) >= self.K:
            ev = self._most_redundant_bank()
            self.events.append(("evict", self.bank[ev].born))
            self.bank.pop(ev)
        self.bank.append(lm)
        self.events.append(("add", lm.born, len(self.bank)))

    def add_chunk(self, k, v, pose, pose_next=None):
        """k,v: [ct,H,D]; pose: (4,4); pose_next: (4,4) for retrieval target (mode b)."""
        c = self._n; self._n += 1
        # clone: in the real pipeline k/v are slices of a kv_cache that gets overwritten
        # next chunk, so the bank/recent MUST own their copies.
        lm = Landmark(k.detach().clone(), v.detach().clone(), pose, c)
        if c < self.n_sink:
            self.sink.append(lm); return self._retrieve(pose_next if pose_next is not None else pose)
        self.recent.append(lm)
        while len(self.recent) > self.recent_w:
            old = self.recent.pop(0)
            self._bank_insert(old)
        return self._retrieve(pose_next if pose_next is not None else pose)

    def _retrieve(self, target_pose):
        """Pick which bank landmarks enter the active window. Returns list of bank indices."""
        if not self.bank:
            return []
        if self.mode == "a":
            return list(range(len(self.bank)))          # attend all
        ds = [pose_dist(target_pose, lm.pose) for lm in self.bank]
        if not getattr(self, "diverse", False):
            order = np.argsort(ds)[: self.retrieve_k]    # plain top-k nearest (clusters!)
            return sorted(int(i) for i in order)
        # diverse (MMR / farthest-point): nearest-1 (the match) + greedily add the landmark
        # FARTHEST from already-picked → relevance + broad coverage (fixes the clustering).
        n = len(self.bank)
        picked = [int(np.argmin(ds))]
        while len(picked) < min(self.retrieve_k, n):
            best, bestd = -1, -1.0
            for i in range(n):
                if i in picked:
                    continue
                d = min(pose_dist(self.bank[i].pose, self.bank[p].pose) for p in picked)
                if d > bestd:
                    bestd, best = d, i
            picked.append(best)
        return sorted(picked)

    def active_layout(self, R):
        """What attention sees this step: sink + recent + bank[R]. Returns count + chunks."""
        chunks = self.sink + self.recent + [self.bank[i] for i in R]
        return chunks

    def get_active_kv(self, R):
        chunks = self.active_layout(R)
        if not chunks:
            return None, None, 0
        k = torch.cat([c.k for c in chunks], 0); v = torch.cat([c.v for c in chunks], 0)
        return k, v, k.shape[0]

    def write_into_cache(self, kv_cache, R):
        """Assemble active = sink+recent+bank[R], copy into preallocated kv_cache
        ([B,tok,H,D]; we unsqueeze B=1), return local_end_index (tokens)."""
        k, v, tot = self.get_active_kv(R)
        if tot == 0:
            return 0
        kk = k.unsqueeze(0) if k.dim() == 3 else k
        vv = v.unsqueeze(0) if v.dim() == 3 else v
        if tot > kv_cache["k"].shape[1]:
            raise RuntimeError(f"landmark active {tot} > cache capacity {kv_cache['k'].shape[1]}")
        kv_cache["k"][:, :tot].copy_(kk)
        kv_cache["v"][:, :tot].copy_(vv)
        return tot

    def stats(self, R):
        return dict(sink=len(self.sink), recent=len(self.recent), bank=len(self.bank),
                    attended=len(self.sink) + len(self.recent) + len(R))
