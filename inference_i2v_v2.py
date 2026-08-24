#!/usr/bin/env python3
"""Bank-v2 wiring shell for inference_i2v.py.

Uses the i2v path (which carries its own decode memory fix); the only change is
that LandmarkCache is swapped for LandmarkCacheV2 (always-full storage + protect2
eviction + top-k nearest retrieval + bank stored in CPU pinned memory) before the
pipeline is imported. The CLI is identical to inference_i2v.py.
Intended for single-process, single-GPU runs, so there is no multi-process
pinned-pool stacking issue.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import longctx.inference.landmark_cache as _bank_mod
from longctx.inference.landmark_cache_v2 import LandmarkCacheV2 as _V2

_bank_mod.LandmarkCache = _V2
if int(os.environ.get("RANK", "0")) == 0:
    print("[bank-v2] LandmarkCache -> LandmarkCacheV2 (i2v path)", flush=True)

import runpy
_target = os.path.join(_HERE, "inference_i2v.py")
sys.argv[0] = _target
runpy.run_path(_target, run_name="__main__")
