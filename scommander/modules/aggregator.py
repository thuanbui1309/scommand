"""Temporal aggregators: collapse the T axis of (T, B, D) spike trains to (B, D).

Sum-over-time is the baseline; ``LearnedTemporalAgg`` (Track A, Phase 03)
plugs in here via the same registry key ``aggregator``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scommander.models.registry import register


@register("aggregator", "sum_over_time")
class SumAggregator(nn.Module):
    """Pool the temporal axis by summation.

    Shape:
        Input:  ``(T, B, D)``
        Output: ``(B, D)``
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=0)
