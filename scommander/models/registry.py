"""Component registry + build_model resolver.

Every pluggable building block (neuron, encoder, attention, mlp, aggregator,
value_branch, long_range_branch, classifier) registers itself at import time
via `@register(kind, name)`. `build_model(cfg)` reads an OmegaConf / dict
config and assembles a SpikCommander instance by resolving registry keys.

Track C swap points are first-class: `long_range_branch.spiking_mamba` is a
reserved key populated in Phase 05.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

import torch.nn as nn

# Registry kinds mirror the paper's block taxonomy. Adding a new kind here is
# a deliberate architectural decision (touches build_model too).
REGISTRY: Dict[str, Dict[str, Type[nn.Module]]] = {
    "neuron": {},              # LIF, PLIF (Phase 08)
    "encoder": {},             # SEE
    "attention": {},           # STASA (SWA + LRA), MSW-STASA (Phase 03)
    "mlp": {},                 # SCR-MLP
    "aggregator": {},          # Sum-over-time, LearnedTemporalAgg (Phase 03)
    "value_branch": {},        # Depthwise conv on V-branch
    "long_range_branch": {},   # LRA (baseline) | SpikingMamba (Track C, Phase 05)
    "classifier": {},          # Linear head
    "model": {},               # Full trunk: spikcommander
}


def register(kind: str, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator: `@register('neuron', 'lif')` registers class under REGISTRY[kind][name]."""
    if kind not in REGISTRY:
        raise KeyError(f"Unknown registry kind: {kind!r}. Valid: {list(REGISTRY)}")

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in REGISTRY[kind]:
            raise KeyError(f"{kind}.{name!r} already registered by {REGISTRY[kind][name]}")
        REGISTRY[kind][name] = cls
        return cls

    return decorator


def resolve(kind: str, name: str) -> Type[nn.Module]:
    """Look up a registered class. Raises KeyError with actionable message if missing."""
    if kind not in REGISTRY:
        raise KeyError(f"Unknown registry kind: {kind!r}. Valid: {list(REGISTRY)}")
    if name not in REGISTRY[kind]:
        available = sorted(REGISTRY[kind])
        raise KeyError(
            f"{kind}.{name!r} not registered. Available {kind}: {available}. "
            f"Did you forget to import the module that declares it?"
        )
    return REGISTRY[kind][name]


# Per-dataset defaults from paper Tables 2/3 + best_config_*.py. Overridable via cfg.
_DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "shd": {"n_classes": 20, "in_features": 140, "dim": 128, "n_heads": 8, "depth": 1},
    "ssc": {"n_classes": 35, "in_features": 140, "dim": 256, "n_heads": 16, "depth": 1},
    "gsc": {"n_classes": 35, "in_features": 140, "dim": 256, "n_heads": 16, "depth": 2},
}


def build_model(cfg: Any) -> nn.Module:
    """Assemble a SpikCommander from config.

    Config contract (OmegaConf or dict):
        cfg.dataset.name = 'shd' | 'ssc' | 'gsc'
        cfg.model.arch = 'spikcommander'
        cfg.model.{depth, dim, n_heads, expansion, window_radius}  [all optional; dataset default if absent]
        cfg.neuron.{tau, v_threshold, v_reset, surrogate.name, surrogate.alpha, backend}
        cfg.training.dropout

    Returns trunk module ready for training. Track C swap is handled via
    `cfg.model.long_range_branch = 'spiking_mamba'` (reserved in Phase 05).
    """
    _cfg_get = _cfg_getter(cfg)

    dataset_name = _cfg_get("dataset.name")
    if dataset_name not in _DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Supported: {list(_DATASET_DEFAULTS)}")
    defaults = _DATASET_DEFAULTS[dataset_name]

    # Architecture hyperparameters (cfg overrides dataset defaults)
    dim = _cfg_get("model.dim", defaults["dim"])
    n_heads = _cfg_get("model.n_heads", defaults["n_heads"])
    depth = _cfg_get("model.depth", defaults["depth"])
    window_radius = _cfg_get("model.window_radius", 20)
    expansion = _cfg_get("model.expansion", 4.0)
    arch = _cfg_get("model.arch", "spikcommander")

    # Neuron config (passed to every LIF call-site)
    neuron_cfg: Dict[str, Any] = {
        "tau": _cfg_get("neuron.tau", 2.0),
        "v_threshold": _cfg_get("neuron.v_threshold", 1.0),
        "v_reset": _cfg_get("neuron.v_reset", 0.5),
        "surrogate": _cfg_get("neuron.surrogate.name", "atan"),
        "alpha": _cfg_get("neuron.surrogate.alpha", 5.0),
        "backend": _cfg_get("neuron.backend", "cupy"),
    }

    dropout_rate = float(_cfg_get("training.dropout", 0.0))

    # Long-range branch selector (Track C swap point)
    long_range_name = _cfg_get("model.long_range_branch", "lra")

    model_cls = resolve("model", arch)
    return model_cls(
        in_features=defaults["in_features"],
        num_classes=defaults["n_classes"],
        dim=dim,
        n_heads=n_heads,
        depth=depth,
        window_radius=window_radius,
        expansion=expansion,
        long_range_branch_name=long_range_name,
        neuron_cfg=neuron_cfg,
        dropout_rate=dropout_rate,
    )


def _cfg_getter(cfg: Any) -> Callable[..., Any]:
    """Return a helper that reads dotted paths from cfg (OmegaConf or dict)."""
    # OmegaConf DictConfig has .select; plain dict does not. Unify.
    try:
        from omegaconf import OmegaConf  # type: ignore

        def _get(path: str, default: Any = None) -> Any:
            v = OmegaConf.select(cfg, path, default=default)
            return default if v is None else v

        return _get
    except ImportError:
        def _get(path: str, default: Any = None) -> Any:
            node: Any = cfg
            for part in path.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif hasattr(node, part):
                    node = getattr(node, part)
                else:
                    return default
            return node

        return _get
