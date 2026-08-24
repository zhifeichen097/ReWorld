"""v1 inference KV-compression cache — three deployable policies, ALL budget-bounded.

Shared structure (budget = budget_chunks * chunk_tokens TOKENS, never full-KV):
    [ sink: first n_sink chunks, FULL res ]
  + [ recent: last W chunks, FULL res ]
  + [ long-term: older chunks, COMPRESSED per policy ]
total attended tokens <= budget_tok at all times.

Policies (what happens to a chunk when it ages out of the recent window):
  - "window" : DROP it (forget). long-term stays empty → pure sink+sliding-window baseline.
  - "naive"  : POOL its tokens spatially (keep_tok = ct // pool_ratio) → low-res memory.
               graduated: older long-term tiers get pooled harder; evict oldest if still over.
  - "h2o"    : KEEP top-M tokens by importance (accumulated attention), drop the rest.
               evict globally-lowest-importance tokens if long-term over budget.

This module is the POLICY/bookkeeping core (token accounting + which tokens survive).
It is output-identical to a true compressed cache: the model attends to exactly the
surviving tokens.  Wiring into causal_diffusion_inference_compressed comes after this
logic is smoke-verified.  Pooling here = mean-pool over contiguous token groups (a
spatial-neighbour proxy; real grid pooling wires the H×W layout in at integration).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


def key_distinctiveness(k_chunk):
    """Per-token importance from K alone (no attention capture):
    a token is IMPORTANT if its key is DISTINCTIVE (low cosine sim to the chunk mean).
    k_chunk: [tok, H, D] or [B, tok, H, D] → returns [tok] importance (higher=keep).
    """
    k = k_chunk if k_chunk.dim() == 3 else k_chunk[0]
    f = k.reshape(k.shape[0], -1).float()
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
    anchor = f.mean(0, keepdim=True); anchor = anchor / (anchor.norm() + 1e-6)
    sim = (f @ anchor.T).squeeze(-1)          # cosine to chunk-mean
    return 1.0 - sim                          # distinctive (low sim) → high importance


@dataclass
class Slot:
    kind: str                 # 'sink' | 'recent' | 'long'
    k: torch.Tensor           # [tok, H, D]
    v: torch.Tensor
    born: int                 # chunk index
    imp: Optional[torch.Tensor] = None   # [tok] importance (h2o), else None


class CompressCache:
    def __init__(self, chunk_tokens, budget_chunks=12, n_sink=1, recent_w=7,
                 policy="naive", pool_ratio=4, h2o_keep_frac=0.25):
        self.ct = int(chunk_tokens)
        self.budget_tok = int(budget_chunks) * self.ct
        self.n_sink = int(n_sink)
        self.recent_w = int(recent_w)
        self.policy = policy
        self.pool_ratio = int(pool_ratio)
        self.h2o_keep = max(1, int(self.ct * h2o_keep_frac))
        self.long_budget = self.budget_tok - (self.n_sink + self.recent_w) * self.ct
        assert self.long_budget >= 0, "budget too small for sink+recent"
        self.sink: List[Slot] = []
        self.recent: List[Slot] = []
        self.long: List[Slot] = []
        self._n = 0

    # ---- helpers ----
    def _pool(self, k, v, ratio):
        """mean-pool contiguous groups of `ratio` tokens → tok//ratio tokens."""
        tok = k.shape[0]; keep = max(1, tok // ratio)
        k2 = k[: keep * ratio].reshape(keep, ratio, *k.shape[1:]).mean(1)
        v2 = v[: keep * ratio].reshape(keep, ratio, *v.shape[1:]).mean(1)
        return k2, v2

    def _topk_tokens(self, slot: Slot, m):
        idx = torch.topk(slot.imp, min(m, slot.imp.shape[0])).indices.sort().values
        return slot.k[idx], slot.v[idx], slot.imp[idx]

    def _long_tokens(self):
        return sum(s.k.shape[0] for s in self.long)

    # ---- main ----
    def add_chunk(self, k_chunk, v_chunk, importance=None):
        """k_chunk,v_chunk: [ct, H, D]; importance: [ct] (h2o) or None."""
        c = self._n; self._n += 1
        if c < self.n_sink:
            self.sink.append(Slot('sink', k_chunk, v_chunk, c)); return
        self.recent.append(Slot('recent', k_chunk, v_chunk, c, importance))
        # age out oldest recent if window full
        while len(self.recent) > self.recent_w:
            old = self.recent.pop(0)
            if self.policy == "window":
                pass  # drop → forget
            elif self.policy == "naive":
                k2, v2 = self._pool(old.k, old.v, self.pool_ratio)
                self.long.append(Slot('long', k2, v2, old.born))
                self._enforce_long_budget_naive()
            elif self.policy == "h2o":
                assert old.imp is not None, "h2o needs per-token importance"
                k2, v2, i2 = self._topk_tokens(old, self.h2o_keep)
                self.long.append(Slot('long', k2, v2, old.born, i2))
                self._enforce_long_budget_h2o()

    def _enforce_long_budget_naive(self):
        # pool oldest long slots harder, then evict oldest, until within budget
        while self._long_tokens() > self.long_budget and self.long:
            oldest = self.long[0]
            if oldest.k.shape[0] > 1:
                oldest.k, oldest.v = self._pool(oldest.k, oldest.v, 2)
            else:
                self.long.pop(0)

    def _enforce_long_budget_h2o(self):
        # evict globally-lowest-importance tokens until within budget
        while self._long_tokens() > self.long_budget and self.long:
            # find slot with the lowest single-token importance; drop its weakest token
            wi = min(range(len(self.long)), key=lambda i: float(self.long[i].imp.min()))
            s = self.long[wi]
            keep = torch.topk(s.imp, s.imp.shape[0] - 1).indices.sort().values if s.imp.shape[0] > 1 else None
            if keep is None:
                self.long.pop(wi)
            else:
                s.k, s.v, s.imp = s.k[keep], s.v[keep], s.imp[keep]

    def get_kv(self):
        slots = self.sink + self.long + self.recent
        if not slots:
            return None, None, 0
        k = torch.cat([s.k for s in slots], 0); v = torch.cat([s.v for s in slots], 0)
        return k, v, k.shape[0]

    def write_into_cache(self, kv_cache):
        """Copy the compressed K/V into a preallocated kv_cache dict ([B,tok,H,D]);
        return new local_end_index (total tokens). Caller squeezes B before add_chunk
        (B=1 inference), we unsqueeze here."""
        k, v, tot = self.get_kv()
        if tot == 0:
            return 0
        kk = k.unsqueeze(0) if k.dim() == 3 else k
        vv = v.unsqueeze(0) if v.dim() == 3 else v
        if tot > kv_cache["k"].shape[1]:
            raise RuntimeError(f"compressed tokens {tot} > cache capacity {kv_cache['k'].shape[1]}")
        kv_cache["k"][:, :tot].copy_(kk)
        kv_cache["v"][:, :tot].copy_(vv)
        return tot

    def layout(self):
        return dict(sink=len(self.sink), long=len(self.long), recent=len(self.recent),
                    long_tok=self._long_tokens(), total_tok=self.get_kv()[2])
