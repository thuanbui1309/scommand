"""Losses: CE (Phase 02), sparsity (Phase 06), KD (Phase 05 C3)."""

from scommander.losses.ce import SumSoftmaxCE, accuracy_from_logits, to_one_hot
from scommander.losses.kd import LogitKDLoss
from scommander.losses.sparsity import FiringRateCollector, FiringRatePenalty

__all__ = [
    "FiringRateCollector",
    "FiringRatePenalty",
    "LogitKDLoss",
    "SumSoftmaxCE",
    "accuracy_from_logits",
    "to_one_hot",
]
