# =============================================================================
# Self-Forcing Dataset Package
# =============================================================================
# Modular dataset classes for loading and processing video data for training
# autoregressive video diffusion models.
# =============================================================================

from dataset.base import (
    BaseVideoDataset,
)

from dataset.action_datasets import (
    BaseActionDataset,
    DL3DVDataset,
    MinecraftDataset,
    OmniWorldGameDataset,
    RealEstate10KDataset,
    SekaiGameDataset,
    SimulatedActionDataset,
)

from dataset.lmdb_datasets import (
    ODERegressionLMDBDataset,
    ShardingLMDBDataset,
)

from dataset.pair_datasets import (
    TextDataset,
    TextImagePairDataset,
    TextVideoPairDataset,
)

from dataset.samplers import DistributedWeightedSampler

from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Dataset

import importlib
import inspect
import pkgutil
from typing import Any, Dict, Iterable, Type


# ============================================================================
# Dataset registry utilities
# ============================================================================

_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def _normalize_dataset_key(name: str) -> str:
    """Normalize name variants (case / hyphens / spaces) into the registry key."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def register_dataset(cls: Type[Dataset], *names: str) -> Type[Dataset]:
    """Register a Dataset subclass in `_DATASET_REGISTRY`.

    Args
    ----
    cls:
        The `torch.utils.data.Dataset` subclass to register.
    names:
        Extra alias strings; the class name itself is always used as a key.
    """
    keys = (cls.__name__,) + tuple(names)
    for k in keys:
        if not k:
            continue
        _DATASET_REGISTRY[_normalize_dataset_key(k)] = cls
    return cls


def _iter_all_dataset_subclasses() -> Iterable[Type[Dataset]]:
    """Recursively iterate over all subclasses of `torch.utils.data.Dataset` (any depth)."""
    seen: set[Type[Dataset]] = set()
    stack = list(Dataset.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        yield cls
        stack.extend(cls.__subclasses__())


def _auto_register_datasets() -> None:
    """Auto-import all submodules under `dataset.*` and register the Dataset subclasses they define (any depth)."""
    if _DATASET_REGISTRY:
        return

    # First import every submodule so their Dataset subclasses get loaded.
    for m in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        mod_name = m.name
        try:
            importlib.import_module(mod_name)
        except Exception:
            # Best-effort: allow individual modules to fail to import when optional deps are missing.
            continue

    # Then register every currently imported Dataset subclass (any depth) in the global table.
    for cls in _iter_all_dataset_subclasses():
        register_dataset(cls)


def get_dataset_cls(name: str) -> Type[Dataset]:
    """Look up a Dataset class in the registry by (case-insensitive) name."""
    _auto_register_datasets()
    key = _normalize_dataset_key(name)
    if key not in _DATASET_REGISTRY:
        known = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset name {name!r}. Known: {known}")
    return _DATASET_REGISTRY[key]


def _filter_kwargs_for_cls(cls: Type[Dataset], kwargs: Dict[str, object]) -> Dict[str, object]:
    """Filter a kwargs dict against the target class's `__init__` signature.

    Rules:
    - If `__init__` accepts `**kwargs`, keep everything as-is.
    - Otherwise keep only parameters explicitly declared in the signature.
    - If the user passed unknown parameters, raise a clear error to aid config debugging.
    """
    init = getattr(cls, "__init__", None)
    if init is None:
        return dict(kwargs)

    try:
        sig = inspect.signature(init)
    except (TypeError, ValueError):
        return dict(kwargs)

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)

    accepted = {k for k in params.keys() if k != "self"}
    unknown = [k for k in kwargs.keys() if k not in accepted]
    if unknown:
        raise TypeError(
            f"{cls.__name__}.__init__() got unexpected config keys: {sorted(unknown)}"
        )
    return dict(kwargs)


def initial_cls_from_config(
    ds_cfg: Any,
    *,
    num_frames: int,
    video_size: Iterable[int],
) -> Dataset:
    """Build one or more dataset instances from the YAML dataset config.

    Typical usage:

    - training set:   `initial_cls_from_config(config.datasets, ...)`
    - validation set: `initial_cls_from_config(config.eval_datasets, ...)`

    Supported config shapes (`ds_cfg` usually comes from the YAML `datasets` /
    `eval_datasets` field):

    - `ds_cfg is None`:
        - Not allowed. Callers must configure `datasets` / `eval_datasets` explicitly.
    - `ds_cfg` is a `dict`:
        - A single dataset, e.g. `{"name": "MinecraftDataset", "params": {...}}`.
    - `ds_cfg` is a `list[dict]`:
        - Multiple dataset configs, instantiated in order and joined with `ConcatDataset`.
    """
    if ds_cfg is None:
        # Raise a clearer error to aid config debugging.
        raise ValueError(
            "ds_cfg must not be empty; configure datasets / eval_datasets in your config yaml, "
            "e.g.: datasets: [{name: MinecraftDataset, params: {...}}]"
        )

    ds_cfg_py = OmegaConf.to_container(ds_cfg, resolve=True)
    if isinstance(ds_cfg_py, dict):
        ds_items = [ds_cfg_py]
    elif isinstance(ds_cfg_py, list):
        ds_items = ds_cfg_py
    else:
        raise TypeError("ds_cfg must be a dict or a list[dict]")

    datasets = []
    for i, item in enumerate(ds_items):
        if not isinstance(item, dict):
            raise TypeError(f"datasets[{i}] must be a dict")

        name = (item.get("name") or item.get("type") or "").strip()
        if not name:
            raise ValueError(f"datasets[{i}] is missing the name/type field")

        cls = get_dataset_cls(name)

        params = dict(item.get("params") or {})
        # Inject common defaults only if the target class accepts them.
        # All paths and action_mode settings live in params; only the two generic
        # fields num_frames / video_size are injected here.
        injected = {
            "num_frames": num_frames,
            "video_size": video_size,
        }
        for k, v in injected.items():
            if k not in params:
                params[k] = v

        params = _filter_kwargs_for_cls(cls, params)
        datasets.append(cls(**params))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def cycle(dl):
    """Infinite iterator over a dataloader."""
    while True:
        for data in dl:
            yield data
            
__all__ = [
    # Base classes
    'BaseVideoDataset',
    # Action datasets
    'BaseActionDataset',
    'SekaiGameDataset',
    'MinecraftDataset',
    'DL3DVDataset',
    'RealEstate10KDataset',
    'OmniWorldGameDataset',
    'SimulatedActionDataset',
    'initial_cls_from_config',  
    'cycle',
    # LMDB datasets
    'ODERegressionLMDBDataset',
    'ShardingLMDBDataset',
    # Pair datasets
    'TextDataset',
    'TextImagePairDataset',
    'TextVideoPairDataset',
    # Samplers
    'DistributedWeightedSampler',
]
