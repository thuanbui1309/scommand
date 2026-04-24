"""Structured JSON-line logger with optional TensorBoard/WandB hooks.

Row schema includes: amp, gc, mamba_backend fields for experimental reproducibility.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMetadata:
    """Fields carried in every log row for repro auditability."""

    run_id: str
    seed: int
    dataset: str
    variant: str
    amp: bool = False
    gc: bool = False  # gradient checkpointing
    mamba_backend: str = "none"  # none | mamba-ssm | pure-pytorch-fallback
    extra: dict[str, Any] = field(default_factory=dict)


class JsonLineLogger:
    """Append-only JSON-line event log.

    Usage:
        logger = JsonLineLogger("logs/exp1.jsonl", metadata=RunMetadata(...))
        logger.log({"epoch": 0, "train_loss": 1.23, "val_acc": 0.67})
    """

    def __init__(self, path: str | Path, metadata: RunMetadata) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self._fh = self.path.open("a", buffering=1)  # line-buffered

    def log(self, event: dict[str, Any]) -> None:
        row = {"ts": time.time(), **asdict(self.metadata), **event}
        # Flatten extra dict at top level for easier querying
        extra = row.pop("extra", {}) or {}
        row.update(extra)
        self._fh.write(json.dumps(row, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonLineLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def get_stdout_logger(name: str = "scommander", level: int = logging.INFO) -> logging.Logger:
    """Simple stdout logger for human-readable progress."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
