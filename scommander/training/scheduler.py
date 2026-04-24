"""LR scheduler factory.

Ports the ``CosineAnnealingLR`` setup from
``reference/SpikCommander/SCommander/main_former_v2_*.py:321``.

The reference ``n_warmup`` field exists in configs but is **never used** in
any training script — warmup is skipped in the baseline. This module wraps
only what the reference actually uses.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def build_scheduler(optimizer: Optimizer, cfg) -> LRScheduler:
    """Return a CosineAnnealingLR scheduler.

    Args:
        optimizer: The optimizer to wrap.
        cfg: OmegaConf config. Reads ``cfg.training.scheduler.t_max``.
            Reference default: 40.

    Returns:
        ``torch.optim.lr_scheduler.CosineAnnealingLR`` instance.
    """
    t_max = int(cfg.training.scheduler.t_max)
    return CosineAnnealingLR(optimizer, T_max=t_max)
