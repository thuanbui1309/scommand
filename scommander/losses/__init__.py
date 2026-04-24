"""Losses: time-aggregated cross-entropy (Phase 02 baseline).

Firing-rate sparsity + KD losses land in later phases (06 sparsity, 05 C3 distill).
"""

from scommander.losses.ce import SumSoftmaxCE, accuracy_from_logits, to_one_hot

__all__ = ["SumSoftmaxCE", "accuracy_from_logits", "to_one_hot"]
