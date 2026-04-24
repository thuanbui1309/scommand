"""Time-aggregated cross-entropy loss.

Ports reference ``calc_loss(config, output, y)`` with ``config.loss='sum'``
(``reference/SpikCommander/SCommander/main_former_v2_*.py:26-45``):

    m = torch.sum(Softmax(output, dim=-1), dim=0)      # (T,B,C) -> (B,C)
    loss = nn.CrossEntropyLoss()(m, y_onehot)          # y pre-one-hot

and the matching accuracy metric (lines 57-72).

Targets ``y`` are **one-hot floats**, matching the reference's
``F.one_hot(y, n_classes).float()`` pre-loss conversion. The trainer handles
that conversion so loss call sites stay symmetric with the reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SumSoftmaxCE(nn.Module):
    """Sum-over-time Softmax + CE against one-hot targets.

    Matches reference ``config.loss='sum'`` + ``config.loss_fn='CEloss'`` path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            logits:  ``(T, B, C)`` per-timestep pre-softmax logits.
            y_onehot: ``(B, C)`` float one-hot targets.

        Returns:
            Scalar loss.
        """
        m = torch.sum(F.softmax(logits, dim=-1), dim=0)   # (B, C)
        return self.ce(m, y_onehot)


def accuracy_from_logits(logits: torch.Tensor, y_onehot: torch.Tensor) -> float:
    """Batch-mean top-1 accuracy matching reference ``calc_metric`` (line 57-72).

    Args:
        logits:  ``(T, B, C)`` per-timestep pre-softmax logits.
        y_onehot: ``(B, C)`` float one-hot targets.

    Returns:
        Python float in ``[0, 1]``.
    """
    m = torch.sum(F.softmax(logits, dim=-1), dim=0)       # (B, C)
    pred = m.argmax(dim=-1)
    target = y_onehot.argmax(dim=-1)
    return (pred == target).float().mean().item()


def to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convenience wrapper for ``F.one_hot(y, num_classes).float()``."""
    return F.one_hot(y, num_classes).float()
