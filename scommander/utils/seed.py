"""Deterministic seed control across Python, NumPy, PyTorch, CUDA, CuPy."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all RNG seeds and enforce determinism.

    Args:
        seed: integer seed value.
        deterministic: if True, enable cuDNN deterministic mode (may slow training).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace for deterministic matmul (PyTorch >= 1.11)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    # CuPy (if available)
    try:
        import cupy as cp

        cp.random.seed(seed)
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeder. Call with worker_init_fn=worker_init_fn."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
