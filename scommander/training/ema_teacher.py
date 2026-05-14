"""EMA teacher for self-distillation (Phase 05 C3)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EmaTeacher(nn.Module):
    """Exponential moving average of student weights.

    Teacher forward = student-architecture forward with EMA-weighted params.
    No gradient flow through teacher.
    """

    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        for t_p, s_p in zip(self.teacher.parameters(), student.parameters()):
            t_p.mul_(self.decay).add_(s_p.detach(), alpha=1.0 - self.decay)
        for t_b, s_b in zip(self.teacher.buffers(), student.buffers()):
            t_b.copy_(s_b)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.teacher(*args, **kwargs)

    def reset(self) -> None:
        if hasattr(self.teacher, "reset"):
            self.teacher.reset()
