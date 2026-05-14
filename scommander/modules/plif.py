"""Parametric LIF (PLIF) spiking neuron — Phase 05 C2.

Thin wrapper around ``spikingjelly.activation_based.neuron.ParametricLIFNode``:
the membrane time constant ``tau`` becomes a learnable scalar per neuron
module (one ``tau`` per call-site, not per channel). Matches the C2 spec in
``plans/260422-0220-spikcommander-improvement/phase-05-track-c-hybrid.md``
("LOW RISK, ships regardless of C1 outcome").

Drop-in for ``LIFNode`` — same forward shape contract ``(T, B, ...)`` →
``(T, B, ...)``, same reset semantics. Selected via ``neuron_cfg.type='plif'``
through the ``make_lif`` factory dispatcher.

Param overhead: 1 learnable scalar (``tau``) per neuron call-site. Across the
SHD baseline (~8 LIF call-sites) that's 8 extra params vs the LIF baseline's
190,664 — < 0.005% overhead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron

from scommander.models.registry import register
from scommander.modules.lif import _make_surrogate


@register("neuron", "plif")
class PLIFNode(nn.Module):
    """Configurable PLIF neuron with learnable membrane time constant.

    Args mirror ``LIFNode`` with one rename: ``tau`` → ``init_tau`` (starting
    value of the learnable parameter; SpikingJelly's internal representation
    is ``w = log(init_tau - 1)`` so the optimised quantity stays unbounded).

    Args:
        init_tau: Initial membrane time constant (gets learned during training).
        v_threshold: Firing threshold (paper default 1.0).
        v_reset: Reset potential after spike; ``None`` → soft reset.
        surrogate_function: Name of surrogate for backprop.
        alpha: Surrogate steepness.
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
        init_tau: float = 2.0,
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
        self.init_tau = init_tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_name = surrogate_function
        self.alpha = alpha
        self.step_mode = step_mode
        self.backend = backend

        self._neuron = neuron.ParametricLIFNode(
            init_tau=init_tau,
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
        self._neuron.reset()

    @property
    def tau(self) -> torch.Tensor:
        """Current learned tau value (read-only view)."""
        # SpikingJelly stores w; effective tau = 1 + sigmoid(w) * (init_tau - 1) on some
        # versions, or tau = 1 / sigmoid(w) on others. Use the public attribute when
        # available; fallback to the parameter itself.
        if hasattr(self._neuron, "tau"):
            t = self._neuron.tau
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)
        return self._neuron.w.detach()

    def extra_repr(self) -> str:
        return (
            f"init_tau={self.init_tau}, v_th={self.v_threshold}, v_reset={self.v_reset}, "
            f"surrogate={self.surrogate_name}(alpha={self.alpha}), "
            f"step_mode={self.step_mode}, backend={self.backend}"
        )
