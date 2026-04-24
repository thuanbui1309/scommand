"""LIF spiking neuron wrapper on SpikingJelly.

Implements paper Eqs 1-3 (hard reset with `V_reset`):
    H[t] = V[t-1] - (1/tau) * (V[t-1] - V_reset) + X[t]         (Eq 1)
    S[t] = Theta(H[t] - V_th)                                    (Eq 2)
    V[t] = H[t] * (1 - S[t]) + V_reset * S[t]                    (Eq 3)

Thin wrapper around `spikingjelly.activation_based.neuron.LIFNode` — keeps
surrogate + backend + step_mode choices explicit and centralised so every
call-site in the model shares the same behaviour.

Backend:
    - `cupy` (default): SpikingJelly fused CUDA kernel, ~3x faster.
    - `torch`: pure-PyTorch fallback; used when CuPy unavailable or on CPU.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

from scommander.models.registry import register


_SURROGATE_MAP = {
    "atan": surrogate.ATan,
    "sigmoid": surrogate.Sigmoid,
    "piecewise_quadratic": surrogate.PiecewiseQuadratic,
}


def _make_surrogate(name: str, alpha: float) -> nn.Module:
    """Build a surrogate-gradient function by name."""
    key = name.lower()
    if key not in _SURROGATE_MAP:
        raise ValueError(f"Unknown surrogate {name!r}. Supported: {list(_SURROGATE_MAP)}")
    return _SURROGATE_MAP[key](alpha=alpha)


@register("neuron", "lif")
class LIFNode(nn.Module):
    """Configurable LIF neuron.

    Args:
        tau: Membrane time constant (paper default 2.0).
        v_threshold: Firing threshold (paper default 1.0).
        v_reset: Reset potential after spike; ``None`` → soft reset.
        surrogate_function: Name of surrogate for backprop.
        alpha: Surrogate steepness (ATan uses alpha=5.0 per paper).
        detach_reset: If True, reset is detached from autograd (standard).
        step_mode: ``'m'`` = multi-step (processes full (T,B,...) tensor).
        backend: ``'cupy'`` (fast) or ``'torch'`` (fallback).
        decay_input: If False, decay applied only to membrane V[t-1].
        store_v_seq: Keep full V trajectory (debug only; memory-heavy).

    Shape:
        Input/output: ``(T, B, ...)``. Step mode ``'m'`` expects leading T.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.5,
        surrogate_function: str = "atan",
        alpha: float = 5.0,
        detach_reset: bool = True,
        step_mode: str = "m",
        backend: str = "cupy",
        decay_input: bool = False,
        store_v_seq: bool = False,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_name = surrogate_function
        self.alpha = alpha
        self.step_mode = step_mode
        self.backend = backend

        self._neuron = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=_make_surrogate(surrogate_function, alpha),
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            decay_input=decay_input,
            store_v_seq=store_v_seq,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._neuron(x)

    def reset(self) -> None:
        """Clear per-sample membrane state. MUST call between independent sequences."""
        self._neuron.reset()

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, v_th={self.v_threshold}, v_reset={self.v_reset}, "
            f"surrogate={self.surrogate_name}(alpha={self.alpha}), "
            f"step_mode={self.step_mode}, backend={self.backend}"
        )


def make_lif(neuron_cfg: dict[str, Any] | None = None, **overrides: Any) -> LIFNode:
    """Factory: build LIFNode from a config dict (`build_model` style), honouring overrides."""
    cfg: dict[str, Any] = {
        "tau": 2.0,
        "v_threshold": 1.0,
        "v_reset": 0.5,
        "surrogate_function": "atan",
        "alpha": 5.0,
        "backend": "cupy",
    }
    if neuron_cfg:
        # Map the build_model keyspace (surrogate → surrogate_function) onto LIFNode's kwargs.
        mapped = {
            "tau": neuron_cfg.get("tau", cfg["tau"]),
            "v_threshold": neuron_cfg.get("v_threshold", cfg["v_threshold"]),
            "v_reset": neuron_cfg.get("v_reset", cfg["v_reset"]),
            "surrogate_function": neuron_cfg.get("surrogate", cfg["surrogate_function"]),
            "alpha": neuron_cfg.get("alpha", cfg["alpha"]),
            "backend": neuron_cfg.get("backend", cfg["backend"]),
        }
        cfg.update(mapped)
    cfg.update(overrides)
    return LIFNode(**cfg)
