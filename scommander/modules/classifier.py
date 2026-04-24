"""Linear classifier heads.

The baseline head is a simple ``nn.Linear(D, C)`` applied to the aggregator
output ``(B, D) -> (B, C)`` producing pre-softmax logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scommander.models.registry import register


@register("classifier", "linear_head")
class ClassifierHead(nn.Module):
    """Single linear layer producing class logits.

    Shape:
        Input:  ``(B, D)``
        Output: ``(B, C)``
    """

    def __init__(self, in_features: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
