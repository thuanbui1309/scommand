"""Firing-rate sparsity regularizer (Phase 06)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scommander.modules.lif import LIFNode


# PLIFNode dropped during 2026-05-15 SeMoE pivot; LIF is the only
# spiking neuron the FR collector needs to hook.
_NEURON_TYPES = (LIFNode,)


class FiringRateCollector:
    """Context manager: forward-hooks every LIF/PLIF, records per-layer FR scalars.

    Usage:
        with FiringRateCollector(model) as fr:
            logits = model(x)
        rates = fr.spike_rates        # {layer_name: scalar tensor}
        penalty = fr.mean_fr()        # scalar tensor

    FR per layer = mean(output) over all dims — spikes are 0/1 so this equals
    the firing rate fraction (∈ [0, 1]).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.spike_rates: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "FiringRateCollector":
        self.spike_rates = {}
        for name, m in self.model.named_modules():
            if isinstance(m, _NEURON_TYPES):
                h = m.register_forward_hook(self._make_hook(name))
                self.handles.append(h)
        return self

    def __exit__(self, *args: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.spike_rates[name] = output.float().mean()
        return hook

    def mean_fr(self) -> torch.Tensor:
        if not self.spike_rates:
            return torch.tensor(0.0)
        return torch.stack(list(self.spike_rates.values())).mean()


class FiringRatePenalty(nn.Module):
    """L_sparsity = lam * penalty(per-layer FRs).

    Two modes:
      - hinge (when ``target_fr`` set): penalize only FR > target via ReLU.
      - mean (default): penalize the mean FR directly.
    """

    def __init__(self, lam: float, target_fr: Optional[float] = None) -> None:
        super().__init__()
        self.lam = lam
        self.target_fr = target_fr

    def forward(self, spike_rates: dict[str, torch.Tensor]) -> torch.Tensor:
        if not spike_rates:
            return torch.tensor(0.0)
        rates = torch.stack(list(spike_rates.values()))
        if self.target_fr is None:
            penalty = rates.mean()
        else:
            penalty = F.relu(rates - self.target_fr).mean()
        return self.lam * penalty
