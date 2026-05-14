"""Knowledge distillation loss (Phase 05 C3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitKDLoss(nn.Module):
    """KL(student/T || teacher/T) * T^2 + alpha * CE-equivalent weighting.

    Operates on time-aggregated logits matching ``SumSoftmaxCE`` semantics:
    sum(Softmax(logits, dim=-1), dim=0) → (B, C). KL is computed between the
    student and teacher aggregated distributions.
    """

    def __init__(self, weight: float = 1.0, temperature: float = 4.0) -> None:
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            student_logits: (T, B, C) pre-softmax.
            teacher_logits: (T, B, C) pre-softmax (no grad expected).
        """
        T = self.temperature
        s = torch.sum(F.softmax(student_logits / T, dim=-1), dim=0)
        t = torch.sum(F.softmax(teacher_logits / T, dim=-1), dim=0)
        s_log = torch.log(s.clamp_min(1e-9))
        kl = F.kl_div(s_log, t, reduction="batchmean")
        return self.weight * kl * (T * T)
