#!/usr/bin/env python3
"""Bank-v2 wiring shell for inference_action.py.

Before the inference pipeline is imported, the symbol
longctx.inference.landmark_cache.LandmarkCache is replaced with LandmarkCacheV2
(always-full storage + protect2/most-redundant eviction + top-k nearest retrieval
+ async transfers through a pinned host slot pool), then inference_action.py's
__main__ is executed unchanged. Apart from this swap, behavior is identical to
inference_action.py and the CLI arguments are exactly the same.
Usage: torchrun ... inference_action_v2.py <same arguments as inference_action.py>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# --- the bank swap: replace the class BEFORE the pipeline imports it ---
import longctx.inference.landmark_cache as _bank_mod
from longctx.inference.landmark_cache_v2 import LandmarkCacheV2 as _V2

_bank_mod.LandmarkCache = _V2
# --- 720p decode memory fix (same recipe as the i2v decode path) ---
import memfix_720p_decode
memfix_720p_decode.install()

if int(os.environ.get("RANK", "0")) == 0:
    print("[bank-v2] LandmarkCache -> LandmarkCacheV2 (always-full, protect2+most-redundant, "
          "top-k nearest, pinned-host tiered KV)", flush=True)

# --- run the real entrypoint unchanged ---
import runpy

_target = os.path.join(_HERE, "inference_action.py")
sys.argv[0] = _target
runpy.run_path(_target, run_name="__main__")
