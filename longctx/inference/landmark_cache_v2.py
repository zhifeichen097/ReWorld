"""v2 landmark cache — always-admit bounded bank + protect-earliest eviction + tiered
host/GPU KV storage (pinned host master copy, small GPU hot set, async streams).

Drop-in replacement for LandmarkCache (v1.5, ./landmark_cache.py): same constructor
signature (legacy args accepted; `eps` / `diverse` are unused-compat no-ops, see
below) and the same methods: add_chunk(k, v, pose, pose_next) -> retrieved bank
indices, active_layout(R), get_active_kv(R), write_into_cache(kv_cache, R), stats(R).

Design notes
------------
1) Always-full bank, bounded on every tier.
   v1's eps admission gate silently dropped revisit providers: with eps=0.5 the far
   (gap>=17 chunks) provider coverage in our synthetic revisit simulation is ~64%, and EVERY
   always-admit variant lifts it to >=98.8%. So v2 admits unconditionally: a chunk
   aging out of `recent` always enters the bank; K recovers its true meaning = bank
   capacity (= host slot count). GPU-resident state stays fixed at
   sink + recent + hot set(<=hot_gpu) + retrieve_k staged members, independent of K,
   so VRAM is bounded for arbitrarily long rollouts and large banks.

2) Eviction rule = winner of the offline rule bake-off ("protect-earliest-2 +
   global-most-redundant", i.e. E_protect2 + C):
     - the first `protect_n`=2 members ever admitted to the bank are permanently
       exempt from eviction;
     - otherwise evict the globally most redundant member = the one with the
       smallest nearest-neighbour pose distance inside the bank; ties evict the
       later-born.
   Why: it is the only rule/retrieval combination that reaches 100% far-revisit
   coverage in BOTH the normalized and the metric pose space (alternatives sit at
   98.8-99.4% normalized / 99.4-100% metric). Every far miss of the other rules has
   a single shape — an early needle chunk (born ~2..6) evicted late in the
   trajectory — and the protect-2 exemption closes exactly that failure mode. The
   rule depends only on relative order and relative distances (no absolute scale),
   so it is immune to pose-normalization fixes. Retrieval stays plain
   top-k(=retrieve_k) nearest by pose: the MMR/diverse branch drags far coverage to
   74-76% (normalized) with misses of the form "provider sits in the bank but is
   not retrieved" — hence `diverse` is accepted but ignored.

3) Tiered storage & bandwidth budget.
   A bank member's KV is packed into ONE contiguous tensor ([2(k/v), ct, H, D] in
   this standalone; [n_layers, 2, ct, H, D] in-pipeline) and lives in a
   preallocated ring of K host slots (pinned when possible): insert = one copy_
   into a slot, evict = slot recycled through a free-list; steady state allocates
   nothing. The host copy is the MASTER (write-through at bank admission, async on
   a dedicated D2H stream), so GPU hot-set eviction never copies back — it just
   drops a reference / recycles a staging slot. The chunk-boundary critical path is
   therefore at most retrieve_k H2D copies, and even those are issued as prefetch
   on a dedicated H2D stream the moment add_chunk() decides R for the NEXT chunk,
   i.e. overlapped with the current chunk's denoise. Scale check: one 30-layer
   member is ~354 MB @480p / ~1.3 GB @720p; six members ~2.1 / 7.8 GB; at
   ~20 GB/s pinned-PCIe that is ~0.1 / 0.4 s, hidden under multi-second chunk
   generation. Host pool @K=30 @720p is ~39 GB pinned — auto-capped at <=50% of
   free RAM; pinned allocation failure degrades stepwise (halve slot count ->
   pageable pool + blocking copies). Without CUDA (or with CPU tensors) everything
   degrades to plain CPU tensors + blocking copies, so trajectory simulations can
   reuse the class unchanged.

Sync points (the four that matter, per the offload survey):
  (1) before compute reads a freshly fetched member: compute-stream wait_event on
      the fetch event (GPU-side wait, CPU never blocks);
  (2) before an H2D overwrites a GPU staging slot: wait_event on the slot's last
      consumer-read event;
  (3) before a D2H overwrites a host slot: the D2H stream waits the slot's last
      recorded event (covers in-flight H2D reads of that slot);
  (4) returning a slot to the free-list is LAZY — no synchronize; the next writer
      performs the wait. Producers additionally record_stream() their tensors on
      the copy stream so the caching allocator cannot recycle in-flight memory.

Pose decisions stay content/layer-agnostic exactly as in v1 (computed once, shared
across layers + cond/uncond); KV and pose are co-located here only for the
standalone smoke.
"""
import collections
import math
import os
import sys
import warnings

import numpy as np
import torch

try:  # package context: longctx.inference.landmark_cache_v2
    from .landmark_cache import LandmarkCache, pose_dist
except ImportError:  # script / flat sibling context
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from landmark_cache import LandmarkCache, pose_dist


def _bank_D(poses):
    """Vectorized pairwise pose distance matrix; same formula as pose_dist()."""
    P = np.stack([np.asarray(p, dtype=np.float64) for p in poses])
    t = P[:, :3, 3]
    yaw = np.degrees(np.arctan2(P[:, 0, 2], P[:, 2, 2]))
    dt = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    dy = np.abs(yaw[:, None] - yaw[None, :]) % 360.0
    dy = np.minimum(dy, 360.0 - dy) / 60.0
    return np.sqrt(dt * dt + dy * dy)


class _PackedChunk:
    """A resident chunk (sink/recent) or a staged active view of a bank member.
    Owns ONE packed tensor kv=[2, ct, H, D]; .k/.v are zero-copy views so the
    object is interface-compatible with v1's Landmark."""
    __slots__ = ("kv", "pose", "visit", "born", "phase_f")

    def __init__(self, kv, pose, born, phase_f=None):
        self.kv = kv
        self.pose = np.asarray(pose).copy()
        self.visit = 1
        self.born = born
        # [rope-remap] start frame of the K phase (the RoPE frame position at write time); None when remap is off.
        self.phase_f = phase_f

    @property
    def k(self):
        return self.kv[0]

    @property
    def v(self):
        return self.kv[1]


class _BankEntry:
    """Bank membership record. KV lives in the tiered store, keyed by member_id."""
    __slots__ = ("member_id", "pose", "born", "visit", "protected", "phase_f")

    def __init__(self, member_id, pose, born, phase_f=None):
        self.member_id = member_id
        self.pose = np.asarray(pose).copy()
        self.born = born
        self.visit = 1
        self.protected = False
        self.phase_f = phase_f  # [rope-remap] start frame of the stored K phase


class _TieredMemberStore:
    """K-slot host-master store for bank member KV + optional GPU hot set.

    Host tier: one preallocated tensor [n_slots, *member_shape] (pinned when the
    members are CUDA tensors and pinning succeeds), free-list of slot ids,
    member_id -> slot id. GPU tier: an LRU OrderedDict (hot set) whose values are
    either warm originals (tensor just admitted, zero-cost insert) or staging-ring
    slots filled by async H2D. Two dedicated streams (_h2d/_d2h) + one event per
    transfer; single-threaded by design — all asynchrony lives in CUDA streams.
    """

    def __init__(self, n_slots, hot_capacity, gpu_spare):
        self.req_slots = int(n_slots)
        self.hot_capacity = int(hot_capacity)
        # hot_capacity==0 => direct-write mode: no ring at all, spare may be 0 too
        self.gpu_spare = max(0 if self.hot_capacity <= 0 else 1, int(gpu_spare))
        self.cuda_path = False
        self.pinned = False
        self.n_slots = 0
        self.member_bytes = 0
        self._pool = None                       # host tier [n_slots, *shape]
        self._free = collections.deque()        # free host slot ids
        self._m2s = {}                          # member_id -> host slot id
        self._slot_ev = {}                      # host slot id -> last event touching it
        self._master_ev = {}                    # member_id -> D2H write-through event
        self._fetch_ev = {}                     # member_id -> pending H2D fetch event
        self._hot = collections.OrderedDict()   # member_id -> (tensor, gpu_slot|None)
        self._gpu_pool = None                   # staging ring [hot+spare, *shape]
        self._gpu_free = collections.deque()
        self._gpu_read_ev = {}                  # gpu slot id -> last consumer-read event
        self._h2d = None
        self._d2h = None
        self._dev = None
        self._alloc_count = 0                   # must stay <=2 (host pool + gpu ring)
        self.counters = dict(hits=0, misses=0, warm_inserts=0, hot_evictions=0,
                             bytes_h2d=0, bytes_d2h=0, waits_pending=0)

    # -- init / capacity ---------------------------------------------------
    def ensure_init(self, sample_kv):
        if self._pool is not None:
            return
        self.cuda_path = bool(sample_kv.is_cuda) and torch.cuda.is_available()
        shape, dtype = tuple(sample_kv.shape), sample_kv.dtype
        self.member_bytes = sample_kv.element_size() * int(np.prod(shape))
        n = self.req_slots
        avail = self._avail_ram()
        if avail is not None and self.member_bytes * n > 0.5 * avail:
            n = max(1, int((0.5 * avail) // max(self.member_bytes, 1)))
            warnings.warn(f"[tiered-store] host pool capped by free RAM: "
                          f"{self.req_slots} -> {n} slots")
        pin = self.cuda_path
        while True:
            try:
                self._pool = torch.empty((n,) + shape, dtype=dtype, pin_memory=pin)
                break
            except RuntimeError:
                if pin and n > 1:
                    n //= 2
                    warnings.warn(f"[tiered-store] pinned alloc failed, retry with {n} slots")
                elif pin:
                    pin = False  # last resort: pageable pool + blocking copies
                    warnings.warn("[tiered-store] pinned alloc impossible; pageable "
                                  "host memory + blocking copies")
                else:
                    raise
        self._alloc_count += 1
        self.pinned = pin
        self.n_slots = n
        self._free = collections.deque(range(n))
        if self.cuda_path:
            dev = sample_kv.device
            self._dev = dev
            n_gpu = self.hot_capacity + self.gpu_spare  # double-buffer spares
            if n_gpu > 0:
                self._gpu_pool = torch.empty((n_gpu,) + shape, dtype=dtype, device=dev)
                self._alloc_count += 1
                self._gpu_free = collections.deque(range(n_gpu))
            self._h2d = torch.cuda.Stream(device=dev)
            self._d2h = torch.cuda.Stream(device=dev)

    @staticmethod
    def _avail_ram():
        try:
            import psutil
            return psutil.virtual_memory().available
        except Exception:
            try:
                return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (ValueError, OSError, AttributeError):
                return None

    @property
    def slots_in_use(self):
        return (self.n_slots - len(self._free)) if self._pool is not None else 0

    # -- hot set (LRU, write-through => eviction is free) ------------------
    def _hot_insert(self, member_id, tensor, gpu_slot):
        if self.hot_capacity <= 0:      # direct-write mode keeps nothing on GPU
            return
        old = self._hot.pop(member_id, None)
        if old is not None and old[1] is not None:
            self._gpu_free.append(old[1])
        self._hot[member_id] = (tensor, gpu_slot)
        while len(self._hot) > self.hot_capacity:
            _, (t, s) = self._hot.popitem(last=False)  # LRU out; host is master
            self.counters["hot_evictions"] += 1
            if s is not None:
                self._gpu_free.append(s)  # never a copy-back

    def _acquire_gpu_slot(self):
        # sized hot_capacity+spare while hot uses <=hot_capacity slots => never empty
        assert self._gpu_free, "gpu staging ring exhausted (sizing bug)"
        return self._gpu_free.popleft()

    # -- writes ------------------------------------------------------------
    def put(self, member_id, kv):
        """Admit a member: host slot becomes the master copy (write-through).
        CUDA: async D2H on the dedicated stream + warm insert into the hot set."""
        self.ensure_init(kv)
        if not self._free:
            raise RuntimeError("tiered-store: no free host slot — evict a bank member first")
        slot = self._free.popleft()
        dst = self._pool[slot]
        if self.cuda_path:
            comp = torch.cuda.current_stream()
            self._d2h.wait_stream(comp)              # kv fully produced first
            prev = self._slot_ev.get(slot)
            if prev is not None:
                self._d2h.wait_event(prev)           # sync (3): slot may still be read
            with torch.cuda.stream(self._d2h):
                dst.copy_(kv, non_blocking=self.pinned)
                ev = torch.cuda.Event()
                ev.record()
            kv.record_stream(self._d2h)              # allocator must not recycle mid-flight
            self._slot_ev[slot] = ev
            self._master_ev[member_id] = ev
            self._hot_insert(member_id, kv, None)    # warm: original tensor, no ring slot
            self.counters["warm_inserts"] += 1
            self.counters["bytes_d2h"] += self.member_bytes
        else:
            dst.copy_(kv)
        self._m2s[member_id] = slot

    def free(self, member_id):
        """Bank eviction: recycle the host slot. sync (4): LAZY — no synchronize;
        the next writer of this slot waits its recorded event."""
        slot = self._m2s.pop(member_id)
        self._free.append(slot)
        self._master_ev.pop(member_id, None)
        self._fetch_ev.pop(member_id, None)
        old = self._hot.pop(member_id, None)
        if old is not None and old[1] is not None:
            self._gpu_free.append(old[1])

    # -- reads -------------------------------------------------------------
    def prefetch(self, member_ids):
        """Issue H2D for misses on the dedicated stream and return immediately.
        Called right after add_chunk() decides R for the next chunk, so the copies
        overlap the current chunk's denoise."""
        if self._pool is None or not self.cuda_path or self._gpu_pool is None:
            return
        for mid in member_ids:
            if mid in self._hot:
                self._hot.move_to_end(mid)
                self.counters["hits"] += 1
                continue
            hslot = self._m2s[mid]
            self.counters["misses"] += 1
            gslot = self._acquire_gpu_slot()
            master = self._master_ev.get(mid)
            if master is not None:
                self._h2d.wait_event(master)         # host master must be complete
            rd = self._gpu_read_ev.get(gslot)
            if rd is not None:
                self._h2d.wait_event(rd)             # sync (2): last reader done
            with torch.cuda.stream(self._h2d):
                self._gpu_pool[gslot].copy_(self._pool[hslot], non_blocking=self.pinned)
                ev = torch.cuda.Event()
                ev.record()
            self._slot_ev[hslot] = ev                # host slot reuse waits this read
            self._fetch_ev[mid] = ev
            self._hot_insert(mid, self._gpu_pool[gslot], gslot)
            self.counters["bytes_h2d"] += self.member_bytes

    def get(self, member_ids):
        """Members' packed KV on the compute device. sync (1): the COMPUTE STREAM
        waits pending fetch events; the CPU never blocks. CPU mode returns host
        slot views (valid until that member is evicted)."""
        if not self.cuda_path:
            return [self._pool[self._m2s[mid]] for mid in member_ids]
        if self._gpu_pool is None:                   # direct mode: transient device copies
            comp = torch.cuda.current_stream()
            out = []
            for mid in member_ids:
                mev = self._master_ev.get(mid)
                if mev is not None:
                    comp.wait_event(mev)
                slot = self._m2s[mid]
                t = self._pool[slot].to(self._dev, non_blocking=self.pinned)
                rd = torch.cuda.Event()
                rd.record(comp)
                self._slot_ev[slot] = rd
                self.counters["misses"] += 1
                self.counters["bytes_h2d"] += self.member_bytes
                out.append(t)
            return out
        self.prefetch(member_ids)                    # hits are free; misses staged
        comp = torch.cuda.current_stream()
        out = []
        for mid in member_ids:
            ev = self._fetch_ev.get(mid)
            if ev is not None:
                if not ev.query():
                    self.counters["waits_pending"] += 1
                comp.wait_event(ev)
            t, _ = self._hot[mid]
            self._hot.move_to_end(mid)
            out.append(t)
        return out

    @property
    def member_tokens(self):
        # pool layout [n_slots, 2, ct, H, D]
        return int(self._pool.shape[2]) if self._pool is not None else 0

    def read_pair_into(self, dst_k, dst_v, member_id):
        """Copy one member's K/V straight into preallocated device views on the
        CURRENT stream — zero staging memory (the direct-write path). A hot copy
        (ring or warm) is used D2D when present; otherwise host slot is the source."""
        ent = self._hot.get(member_id)
        comp = torch.cuda.current_stream() if self.cuda_path else None
        if ent is not None:
            fev = self._fetch_ev.get(member_id)
            if fev is not None:
                comp.wait_event(fev)                 # ring slot may still be filling
            t, gslot = ent
            dst_k.copy_(t[0])
            dst_v.copy_(t[1])
            self._hot.move_to_end(member_id)
            if gslot is not None:
                rd = torch.cuda.Event()
                rd.record(comp)
                self._gpu_read_ev[gslot] = rd
            self.counters["hits"] += 1
            return
        slot = self._m2s[member_id]
        src = self._pool[slot]
        if not self.cuda_path:
            dst_k.copy_(src[0])
            dst_v.copy_(src[1])
            return
        mev = self._master_ev.get(member_id)
        if mev is not None:
            comp.wait_event(mev)                     # write-through must land first
        dst_k.copy_(src[0], non_blocking=self.pinned)
        dst_v.copy_(src[1], non_blocking=self.pinned)
        rd = torch.cuda.Event()
        rd.record(comp)
        self._slot_ev[slot] = rd                     # host slot reuse waits this read
        self.counters["misses"] += 1
        self.counters["bytes_h2d"] += self.member_bytes

    def mark_consumed(self, member_ids):
        """Record a consumer-read event on the compute stream so staging slots are
        not overwritten while still being read. Call after the data left the slot
        (e.g. after the active-window cat/copy)."""
        if not self.cuda_path or not member_ids:
            return
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        for mid in member_ids:
            ent = self._hot.get(mid)
            if ent is not None and ent[1] is not None:
                self._gpu_read_ev[ent[1]] = ev


class LandmarkCacheV2:
    """Drop-in v2 of LandmarkCache: sink/recent unchanged; always-admit bank of
    true capacity K in tiered host storage; E_protect2+C eviction; top-k nearest
    retrieval. See module docstring."""

    def __init__(self, chunk_tokens, n_sink=1, recent_w=5, K=30, retrieve_k=6,
                 eps=0.5, mode="b", diverse=False,
                 protect_n=2, hot_gpu=None, gpu_spare=None):
        assert mode in ("a", "b")
        self.ct = chunk_tokens
        self.n_sink = n_sink
        self.recent_w = recent_w
        self.eps = eps          # unused-compat: v1 admission gate; v2 always admits
        self.diverse = diverse  # unused-compat: MMR retrieval loses far coverage
        self.mode = mode        # 'a' = attend the whole bank (compat); 'b' = top-k
        self.retrieve_k = retrieve_k
        self.K = K              # TRUE bank capacity (host slots); no mode-'a' swap
        self.protect_n = protect_n
        if protect_n >= K:
            warnings.warn(f"protect_n={protect_n} >= K={K}: eviction falls back to "
                          "the unprotected rule over all members")
        import os as _os
        if hot_gpu is None and _os.environ.get("BANKV2_HOT_GPU"):
            hot_gpu = int(_os.environ["BANKV2_HOT_GPU"])   # shrink the hot cache when 720p VRAM is tight
        if gpu_spare is None and _os.environ.get("BANKV2_GPU_SPARE"):
            gpu_spare = int(_os.environ["BANKV2_GPU_SPARE"])
        if hot_gpu is None:
            # Default: with no argument/env var set, use direct-write mode (the bank uses
            # zero GPU memory). The staged hot-cache ring (18 slots/layer ~= 744MiB/layer
            # x 30 layers ~= 23GB) reliably OOMs on a fully loaded 720p pipeline; to use
            # staged mode, explicitly pass hot_gpu>0 or set BANKV2_HOT_GPU>0.
            hot_gpu = 0
        if hot_gpu is not None and int(hot_gpu) <= 0:
            # direct-write mode (BANKV2_HOT_GPU=0): no staging ring, no warm set —
            # write_into_cache streams host->cache slices; the bank adds ZERO
            # steady GPU memory on top of the sink/recent window.
            hot, spare = 0, 0
        else:
            hot = max(hot_gpu if hot_gpu is not None else 2 * retrieve_k, retrieve_k)
            spare = gpu_spare if gpu_spare is not None else retrieve_k
        self.store = _TieredMemberStore(K, hot, spare)
        self.sink = []          # list[_PackedChunk], GPU-resident
        self.recent = []        # ring of _PackedChunk, GPU-resident
        self.bank = []          # <=K _BankEntry (KV in tiered store)
        self._n = 0
        self._admitted = 0
        self.events = []        # for inspection, v1-compatible kinds: add/evict
        self._step_ds = []      # consecutive-chunk pose distances (self-calibrated step size for the far gate)
        self._last_pose = None
        self._last_admit_pose = None   # last admitted pose for pose-stride admission

    # -- bank maintenance --------------------------------------------------
    def _evict_index(self):
        """Bake-off winner: protect the earliest `protect_n` admitted members;
        among the rest evict the globally most redundant = smallest
        nearest-neighbour pose distance (neighbours = whole bank); ties evict
        the later-born."""
        n = len(self.bank)
        if n == 1:
            return 0
        D = _bank_D([m.pose for m in self.bank])
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        cand = [i for i in range(n) if not self.bank[i].protected]
        if not cand:            # degenerate K <= protect_n
            cand = list(range(n))
        return min(cand, key=lambda i: (nn[i], -self.bank[i].born))

    def _bank_insert(self, lm):
        self.store.ensure_init(lm.kv)
        if self.store.n_slots < self.K:
            self.K = self.store.n_slots     # host pool was capped; keep invariant honest
        while len(self.bank) >= self.K:
            j = self._evict_index()
            victim = self.bank.pop(j)
            self.store.free(victim.member_id)
            self.events.append(("evict", victim.born))
        self.store.put(lm.born, lm.kv)      # member_id = born (unique)
        e = _BankEntry(lm.born, lm.pose, lm.born, phase_f=lm.phase_f)
        e.protected = self._admitted < self.protect_n
        self._admitted += 1
        self.bank.append(e)
        self.events.append(("add", lm.born, len(self.bank)))

    # -- v1-compatible interface -------------------------------------------
    def add_chunk(self, k, v, pose, pose_next=None, phase_f=None):
        """k,v: [ct,H,D]; pose: (4,4); pose_next: (4,4) retrieval target.
        phase_f: [rope-remap] the RoPE start frame (virtual or real) at which this chunk's
        K was written; pass None when remap is off (no effect).
        Returns retrieved bank indices for the NEXT chunk and prefetches them."""
        c = self._n
        self._n += 1
        if self._last_pose is not None:
            self._step_ds.append(pose_dist(pose, self._last_pose))
        self._last_pose = pose
        # one owning packed copy [2,ct,H,D] (the stack IS the clone; real-pipeline
        # k/v are cache slices that get overwritten next chunk)
        kv = torch.stack((k.detach(), v.detach()))
        lm = _PackedChunk(kv, pose, c, phase_f=phase_f)
        if c < self.n_sink:
            self.sink.append(lm)
        else:
            self.recent.append(lm)
            while len(self.recent) > self.recent_w:
                old = self.recent.pop(0)
                # BANKV2_ADMIT_STRIDE_STEPS=<a> (default 0 = off, byte-identical):
                # pose-stride admission — bank ~one snapshot per a*median-step of
                # TRAVEL, so K slots cover a*K steps of world instead of K.
                # Compares only against the LAST ADMITTED pose (temporal-local),
                # so a fresh revisit pass still gets banked — unlike v1's
                # whole-bank eps gate that starved revisit providers (64%
                # coverage lesson). Skipped chunks are dropped for good: they are
                # the densest, most redundant samples along a straight path.
                stride = float(os.environ.get("BANKV2_ADMIT_STRIDE_STEPS", "0") or 0)
                if (stride > 0 and self._last_admit_pose is not None
                        and len(self._step_ds) >= 3
                        and pose_dist(old.pose, self._last_admit_pose)
                        < stride * float(np.median(self._step_ds))):
                    continue                   # less than `a` steps of travel since the last admission; skip
                self._bank_insert(old)
                self._last_admit_pose = old.pose
        R = self._retrieve(pose_next if pose_next is not None else pose)
        self.store.prefetch([self.bank[i].member_id for i in R])
        return R

    def _retrieve(self, target_pose):
        """Plain top-k nearest by pose (bake-off: keep it; diverse/MMR hurts).

        BANKV2_RETRIEVE_MIN_AGE=<n> (default 0 = off, byte-identical to before):
        members born fewer than n chunks ago are invisible to retrieval this
        step (still banked / aging / evictable as usual). Rationale: with
        recent_w=5 the youngest bank members (age 5-6) duplicate content the
        recent window already shows in full, and on return legs they are the
        just-self-generated chunks — retrieving them feeds the model its own
        degenerate output back as "memory" (repeat-pattern lock). n = recent_w + 2 = 7
        blocks exactly those two youngest cohorts."""
        if not self.bank:
            return []
        if self.mode == "a":
            return list(range(len(self.bank)))       # attend the whole bank
        min_age = int(os.environ.get("BANKV2_RETRIEVE_MIN_AGE", "0") or 0)
        cur = self._n - 1                            # chunk being added right now
        ds = [pose_dist(target_pose, m.pose)
              if (cur - m.born) >= min_age else float("inf")
              for m in self.bank]
        # full distance order (inf sorts last, so [:k] below == old truncation)
        keep = [int(i) for i in np.argsort(ds) if np.isfinite(ds[i])]
        # BANKV2_RETRIEVE_FAR_GATE=<a> (default 0 = off, byte-identical to before):
        # "off-the-map" detector. tau = a * median consecutive-chunk step (self-
        # calibrating — sidesteps per-clip pose normalization). Members within tau
        # are real revisit providers: keep them (possibly fewer than retrieve_k —
        # no far filler). If NONE is within tau the camera is in never-observed
        # territory: keep only 2 nearest as style/identity anchors and leave the
        # rest of the budget empty so imagination isn't crowded by stale content
        # (observed in forward-shuttle test cases).
        far_gate = float(os.environ.get("BANKV2_RETRIEVE_FAR_GATE", "0") or 0)
        if far_gate > 0 and keep and len(self._step_ds) >= 3:
            tau = far_gate * float(np.median(self._step_ds))
            near = [i for i in keep if ds[i] <= tau]
            keep = near if near else keep[:2]
        # BANKV2_RETRIEVE_DIVERSE_SEP=<a> (default 0 = off): retrieval-time
        # diversity. Plain pose-top-k on a continuous trajectory returns
        # CONSECUTIVE chunks — near-duplicate views wasting the landmark budget
        # (confirmed in case replays). Greedy pick in
        # distance order, but skip a candidate closer than delta = a * median
        # chunk step to any already-picked member. Bank stays always-admit —
        # diversity lives in selection, not insertion (v1's insertion-eps
        # permanently lost content).
        div_sep = float(os.environ.get("BANKV2_RETRIEVE_DIVERSE_SEP", "0") or 0)
        if div_sep > 0 and len(keep) > 1 and len(self._step_ds) >= 3:
            delta = div_sep * float(np.median(self._step_ds))
            picked = []
            for i in keep:
                if all(pose_dist(self.bank[i].pose, self.bank[j].pose) >= delta
                       for j in picked):
                    picked.append(i)
                    if len(picked) == self.retrieve_k:
                        break
            keep = picked
        return sorted(keep[: self.retrieve_k])

    def active_layout(self, R):
        """What attention sees this step: sink + recent + staged bank[R].
        NOTE: external callers that keep the returned bank views past this step
        should call self.store.mark_consumed([...member_ids...]) once done reading
        (get_active_kv / write_into_cache do it internally)."""
        mids = [self.bank[i].member_id for i in R]
        kvs = self.store.get(mids)
        staged = [_PackedChunk(kv, self.bank[i].pose, self.bank[i].born,
                               phase_f=self.bank[i].phase_f)
                  for kv, i in zip(kvs, R)]
        # Layout sink|recent|bank, byte-for-byte the same order as v1 (a
        # sink|landmark|recent order was tried and reverted; the order has no semantic
        # effect on attention — position is baked into the RoPE phase).
        return self.sink + self.recent + staged

    def get_active_kv(self, R):
        chunks = self.active_layout(R)
        if not chunks:
            return None, None, 0
        k = torch.cat([c.k for c in chunks], 0)
        v = torch.cat([c.v for c in chunks], 0)
        self.store.mark_consumed([self.bank[i].member_id for i in R])
        return k, v, k.shape[0]

    def write_into_cache(self, kv_cache, R):
        """Assemble active = sink+recent+bank[R], copy into preallocated kv_cache
        ([B,tok,H,D]; we unsqueeze B=1), return local_end_index (tokens)."""
        if self.store.hot_capacity > 0:
            # staged path: members materialize in the ring/warm set first
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
        # direct-write path: window D2D + bank members host->cache slices;
        # no cat, no staging allocation — same byte layout as the staged path.
        kc, vc = kv_cache["k"], kv_cache["v"]
        window = self.sink + self.recent
        tot = sum(c.k.shape[0] for c in window) + len(R) * self.store.member_tokens
        if tot == 0:
            return 0
        if tot > kc.shape[1]:
            raise RuntimeError(f"landmark active {tot} > cache capacity {kc.shape[1]}")
        # [rope-remap] BANKV2_ROPE_REMAP=1 (default 0 = off, byte-equivalent):
        # StreamingLLM-style virtual continuous time axis. Virtual order =
        # sink -> bank[R] (birth order) -> recent, decoupled from the physical layout
        # (sink|recent|bank) — position is baked into the phase, physical order has no
        # semantics. Each chunk gets a corrective rotation delta_f = virtual start frame
        # - phase_f (the content's current phase frame). The PRoPE delta is a
        # position-independent phase factor and complex multiplication commutes, so it is
        # unaffected.
        remap = self._rope_remap_enabled()
        if remap:
            fs = self._frame_seq()
            vf_sink = 0
            vf_bank = vf_sink + sum(c.k.shape[0] for c in self.sink) // fs
            vf_recent = vf_bank + len(R) * (self.store.member_tokens // fs)
            vf = {"sink": vf_sink, "bank": vf_bank, "recent": vf_recent}
        off = 0
        for seg, c in [("sink", c) for c in self.sink] + [("recent", c) for c in self.recent]:
            n = c.k.shape[0]
            kc[0, off:off + n].copy_(c.k)
            vc[0, off:off + n].copy_(c.v)
            if remap:
                assert c.phase_f is not None, f"rope-remap is on but {seg} chunk born={c.born} has no phase_f (the pipeline must pass it in add_chunk)"
                self._rot_t(kc[0, off:off + n], vf[seg] - c.phase_f)
                vf[seg] += n // fs
            off += n
        for i in R:
            n = self.store.member_tokens
            self.store.read_pair_into(kc[0, off:off + n], vc[0, off:off + n],
                                      self.bank[i].member_id)
            if remap:
                e = self.bank[i]
                assert e.phase_f is not None, f"rope-remap is on but bank born={e.born} has no phase_f"
                self._rot_t(kc[0, off:off + n], vf["bank"] - e.phase_f)
                vf["bank"] += n // fs
            off += n
        if remap:
            # The current chunk's virtual start frame = the end of the time axis (= total
            # frames occupied). The pipeline reads it into kv_cache["rope_start_frame"]
            # and rotates Q / new K with it.
            self.last_rope_start_f = off // self._frame_seq()
        return off

    # ── [rope-remap] helpers ──────────────────────────────────────────────
    @staticmethod
    def _rope_remap_enabled():
        return os.environ.get("BANKV2_ROPE_REMAP", "0") == "1"

    def _frame_seq(self):
        return int(os.environ.get("BANKV2_FRAME_SEQ", "880"))

    def set_rope_freqs_t(self, freqs_t):
        """freqs_t: [1024, Ct] complex (the temporal section of the model freqs; Ct=22 at 720p). Injected once by the pipeline."""
        self._freqs_t = freqs_t

    def _rot_t(self, k_slice, delta_f):
        """Shift the temporal RoPE phase of a [n,H,D] K by delta_f frames. RoPE is a
        unit-modulus complex multiplication, so shifts compose; delta_f=0 is the identity.
        Go through fp32 to avoid amplifying bf16 double-rounding."""
        if delta_f == 0:
            return
        ft = getattr(self, "_freqs_t", None)
        assert ft is not None, "rope-remap is on but freqs_t was not injected (set_rope_freqs_t)"
        n, H, D = k_slice.shape
        ct = ft.shape[1]                       # temporal complex dimension (22)
        fs = self._frame_seq()
        phasor = ft[abs(delta_f)]              # [Ct] = e^{i·|Δ|·ω_t}
        if delta_f < 0:
            phasor = phasor.conj()
        x = k_slice[:, :, : 2 * ct].to(torch.float32).reshape(n, H, ct, 2)
        xc = torch.view_as_complex(x)
        xc = xc * phasor.to(xc.device).view(1, 1, ct)
        k_slice[:, :, : 2 * ct].copy_(
            torch.view_as_real(xc).reshape(n, H, 2 * ct).to(k_slice.dtype))

    def stats(self, R):
        s = dict(sink=len(self.sink), recent=len(self.recent), bank=len(self.bank),
                 attended=len(self.sink) + len(self.recent) + len(R))
        s.update(host_slots_used=self.store.slots_in_use,
                 host_slots_total=self.store.n_slots, pinned=self.store.pinned)
        s.update({f"io_{k}": v for k, v in self.store.counters.items()})
        return s


# ===========================================================================
# self-test (CPU-runnable; CUDA smoke runs additionally when a GPU is present)
# ===========================================================================

def _walk_poses(rng, n, step_t=0.35, step_yaw_deg=25.0):
    poses, t, yaw = [], np.zeros(3), 0.0
    for _ in range(n):
        t = t + rng.normal(0.0, step_t, 3)
        yaw = (yaw + rng.normal(0.0, step_yaw_deg)) % 360.0
        r = math.radians(yaw)
        p = np.eye(4, dtype=np.float32)
        c, s = math.cos(r), math.sin(r)
        p[0, 0] = c
        p[0, 2] = s
        p[2, 0] = -s
        p[2, 2] = c
        p[:3, 3] = t
        poses.append(p)
    return poses


def _bits(t):
    return t.detach().contiguous().cpu().view(torch.uint8)


def _same_bytes(a, b):
    return torch.equal(_bits(a), _bits(b))


def _selftest_equivalence(dtype):
    """(a) mechanical equivalence: identical composition (v1 eps=0.0 => always
    admit; short run => no eviction in either) + identical forced selections =>
    write_into_cache output must be byte-identical to v1."""
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    ct, H, D = 3, 2, 4
    N, K, rk, ns, rw = 10, 8, 4, 1, 3
    v1 = LandmarkCache(ct, n_sink=ns, recent_w=rw, K=K, retrieve_k=rk, eps=0.0, mode="b")
    v2 = LandmarkCacheV2(ct, n_sink=ns, recent_w=rw, K=K, retrieve_k=rk, eps=0.5, mode="b")
    poses = _walk_poses(rng, N + 1)
    for c in range(N):
        k = torch.randn(ct, H, D, generator=gen).to(dtype)
        v = torch.randn(ct, H, D, generator=gen).to(dtype)
        R1 = v1.add_chunk(k, v, poses[c], poses[c + 1])
        R2 = v2.add_chunk(k, v, poses[c], poses[c + 1])
        assert R1 == R2, f"retrieval diverged at chunk {c}: {R1} vs {R2}"
    assert [l.born for l in v1.bank] == [e.born for e in v2.bank]
    assert not any(e[0] == "evict" for e in v1.events + v2.events)
    nb = len(v1.bank)
    cap = (ns + rw + K) * ct + 8
    forced = [R1, list(range(min(3, nb))), [], [nb - 1], [0, nb - 1]]
    for R in forced:
        c1 = dict(k=torch.zeros(1, cap, H, D, dtype=dtype), v=torch.zeros(1, cap, H, D, dtype=dtype))
        c2 = dict(k=torch.zeros(1, cap, H, D, dtype=dtype), v=torch.zeros(1, cap, H, D, dtype=dtype))
        t1 = v1.write_into_cache(c1, R)
        t2 = v2.write_into_cache(c2, R)
        assert t1 == t2 == (ns + rw + len(R)) * ct
        assert _same_bytes(c1["k"], c2["k"]) and _same_bytes(c1["v"], c2["v"]), \
            f"byte mismatch for forced R={R}"
    print(f"[selftest a] equivalence {str(dtype):15s}: retrieval identical {N}/{N} steps; "
          f"write_into_cache byte-identical on {len(forced)} forced selections  OK")


def _selftest_bounded_always_full():
    """(b)+(c): 200-chunk rollout. Host slots <=K with zero re-allocation and full
    free-list conservation (b); occupancy == K for every chunk after warmup (c)."""
    rng = np.random.default_rng(1)
    gen = torch.Generator().manual_seed(1)
    ct, H, D = 2, 1, 2
    N, K, rk, ns, rw = 200, 30, 6, 1, 5
    v2 = LandmarkCacheV2(ct, n_sink=ns, recent_w=rw, K=K, retrieve_k=rk)
    poses = _walk_poses(rng, N + 1, step_t=0.25, step_yaw_deg=30.0)
    warmup = ns + rw + K - 1     # first chunk index at which the bank reaches K
    peak = 0
    for c in range(N):
        k = torch.randn(ct, H, D, generator=gen)
        v = torch.randn(ct, H, D, generator=gen)
        R = v2.add_chunk(k, v, poses[c], poses[c + 1])
        st = v2.store
        used = st.slots_in_use
        peak = max(peak, used)
        assert used == len(v2.bank) <= K, f"slot/bank invariant broken at {c}"
        if st._pool is not None:
            assert st.n_slots == K
            assert used + len(st._free) == st.n_slots, f"slot leak at {c}"
            assert len(st._m2s) == len(v2.bank), f"mapping leak at {c}"
            assert st._alloc_count <= 2, "host pool re-allocated (should be a one-time ring)"
        if c >= warmup:
            assert len(v2.bank) == K, f"bank not always-full at chunk {c}: {len(v2.bank)}"
        assert len(R) <= rk
        _, _, tot = v2.get_active_kv(R)
        assert tot <= (ns + rw + rk) * ct, "active window exceeded its budget"
    borns = {e.born for e in v2.bank}
    n_evict = sum(1 for e in v2.events if e[0] == "evict")
    assert {ns, ns + 1} <= borns, "protected earliest-2 members were evicted"
    assert n_evict == N - ns - rw - K, "eviction count mismatch"
    print(f"[selftest b] {N}-chunk rollout: host slots peak={peak} (K={K}), "
          f"free+used==n_slots at every step, allocs={v2.store._alloc_count} (one-time)  OK")
    print(f"[selftest c] always-full: bank==K for every chunk >= {warmup}; "
          f"protected borns {{{ns},{ns+1}}} survived {n_evict} evictions  OK")


def _selftest_cuda_smoke():
    """Extra (GPU only): exercise pinned pool, streams, events, hot-set LRU and
    slot reuse; verify every bank member's host master and fetched GPU copy are
    byte-identical to the ground-truth KV."""
    if not torch.cuda.is_available():
        print("[selftest cuda] skipped (no CUDA)")
        return
    dev = "cuda:0"
    rng = np.random.default_rng(2)
    gen = torch.Generator().manual_seed(2)
    ct, H, D = 4, 2, 8
    N, K, rk, ns, rw = 40, 8, 3, 1, 3
    # small hot set (=rk) forces H2D misses; small K forces slot reuse under events
    v2 = LandmarkCacheV2(ct, n_sink=ns, recent_w=rw, K=K, retrieve_k=rk, hot_gpu=rk)
    poses = _walk_poses(rng, N + 1, step_t=0.2, step_yaw_deg=40.0)
    truth = {}
    cap = (ns + rw + rk) * ct + 4
    kvc = dict(k=torch.zeros(1, cap, H, D, device=dev), v=torch.zeros(1, cap, H, D, device=dev))
    for c in range(N):
        k = torch.randn(ct, H, D, generator=gen)
        v = torch.randn(ct, H, D, generator=gen)
        truth[c] = torch.stack((k, v))
        R = v2.add_chunk(k.to(dev), v.to(dev), poses[c], poses[c + 1])
        v2.write_into_cache(kvc, R)
    torch.cuda.synchronize()
    st = v2.store
    assert st.pinned, "expected pinned host pool on the CUDA path"
    for e in v2.bank:
        hslot = st._m2s[e.member_id]
        assert _same_bytes(st._pool[hslot], truth[e.born]), "host master corrupted"
        (t,) = st.get([e.member_id])
        torch.cuda.synchronize()
        assert _same_bytes(t, truth[e.born]), "fetched GPU copy corrupted"
    cnt = st.counters
    assert cnt["misses"] > 0 and cnt["bytes_h2d"] > 0 and cnt["bytes_d2h"] > 0
    print(f"[selftest cuda] pinned={st.pinned} hot_hits={cnt['hits']} misses={cnt['misses']} "
          f"hot_evictions={cnt['hot_evictions']} bytes_h2d={cnt['bytes_h2d']} "
          f"bytes_d2h={cnt['bytes_d2h']} waits_pending={cnt['waits_pending']}; "
          f"all bank members byte-verified vs ground truth  OK")


if __name__ == "__main__":
    _selftest_equivalence(torch.float32)
    _selftest_equivalence(torch.bfloat16)
    _selftest_bounded_always_full()
    _selftest_cuda_smoke()
    print("ALL SELFTESTS PASSED")
