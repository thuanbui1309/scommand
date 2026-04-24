"""Linear classifier head.

Matches reference (``spikcommder.py:111``): a bias-free Linear applied per
timestep over the trunk output. ``nn.Linear`` naturally broadcasts over any
number of leading dims since it acts on the last one, so (T, B, D) -> (T, B, C)
works with no reshape.

Time aggregation is intentionally *outside* the model — loss receives
per-timestep logits and sums ``Softmax(logits, dim=-1)`` over T, matching
``calc_loss(config, output, y)`` with ``config.loss='sum'``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scommander.models.registry import register


@register("classifier", "linear_head")
class ClassifierHead(nn.Module):
    """Bias-free linear projection over the last (feature) dim.

    Args:
        in_features: Model dim D.
        num_classes: Output class count C.
        bias: Default False to match reference ``layer.Linear(..., bias=False)``.

    Shape:
        Input:  ``(..., D)`` — typically ``(T, B, D)`` from the trunk.
        Output: ``(..., C)`` — leading dims preserved; last dim becomes C.
    """

    def __init__(self, in_features: int, num_classes: int, bias: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
