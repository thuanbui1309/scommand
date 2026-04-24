"""Minimal stub of the ref repo's utils module.

Provides only what main_former_v2_*.py imports: init_logger, build_optimizer,
count_parameters. Real implementation wasn't vendored; we stub with equivalent
behavior so ref entry points can run for ground-truth baseline comparison.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import torch


def init_logger(config, prefix: str = "training") -> logging.Logger:
    """Return a stdout + file logger. Matches ref usage: init_logger(config, 'training')."""
    logger = logging.getLogger("spikcommander_ref")
    if logger.handlers:
        # Already initialized (second call) — reuse.
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = os.path.join("log_files", getattr(config, "dataset", "run").upper())
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    seed = getattr(config, "seed", 0)
    log_path = os.path.join(log_dir, f"{prefix}-{ts}-seed{seed}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


def build_optimizer(config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with config.lr_w and config.weight_decay (ref convention)."""
    lr = float(getattr(config, "lr_w", 1e-3))
    wd = float(getattr(config, "weight_decay", 0.0))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
