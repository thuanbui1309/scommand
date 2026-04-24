"""SSC (Spiking Speech Commands) dataset loader.

SpikingJelly 0.0.0.0.14 dropped ``SpikingSpeechCommands`` — we load SSC via
``tonic.datasets.SSC`` directly (tonic is what spikingjelly wraps internally).
Semantics match reference ``BinnedSpikingSpeechCommands`` from
``reference/SpikCommander/SCommander/datasets.py:337-390``:
  events -> (T, 700) time-binned frame -> (T, 140) neuron-binned frame.

Dataset shape: (T=100, B=256, N=140). 35 classes. Real train/valid/test splits.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import tonic  # tonic.datasets.SSC
from spikingjelly.datasets import pad_sequence_collate

from scommander.augmentations.masking import TimeNeuronMask
from scommander.utils.seed import set_seed


class _Augs:
    """Thin wrapper applying a list of (x, y) -> (x, y) augmentations."""

    def __init__(self, aug: TimeNeuronMask) -> None:
        self._aug = aug

    def __call__(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        return self._aug(x, y)

    def __repr__(self) -> str:
        return f"Augs([{self._aug.__class__.__name__}])"


# SSC raw sensor: 700 input neurons (Heidelberg cochlear model).
_SSC_N_RAW = 700
# 1 s clip @ duration_ms time-bin => T = 1000 / duration_ms.
_SSC_CLIP_MS = 1000


class BinnedSpikingSpeechCommands(Dataset):
    """SSC dataset returning per-sample ``(T, N=140)`` binned frame.

    Wraps ``tonic.datasets.SSC`` (event-based) and performs the same two-stage
    binning as the reference's SpikingJelly-cached frames:
      1. Time-bin events into (T_raw, 700) frame where T_raw = clip_ms / duration_ms.
      2. Neuron-bin 700 -> 700 // n_bins = 140 by contiguous sum-pooling.

    Args:
        root: Dataset cache/download root.
        n_bins: Neuron-axis bin width. Reference: 5 (700//5 = 140).
        split: ``'train'`` | ``'valid'`` | ``'test'``.
        duration: Time-bin width in ms. Reference: 10 (-> T=100).
        transform: Optional ``(frame, label) -> (frame, label)`` post-bin hook
            (used for ``TimeNeuronMask`` augmentation).
    """

    def __init__(
        self,
        root: str,
        n_bins: int,
        split: str = "train",
        duration: int = 10,
        transform: Optional[Callable] = None,
    ) -> None:
        self.n_bins = n_bins
        self.duration_ms = duration
        self.duration_us = duration * 1000
        self.T = _SSC_CLIP_MS // duration  # 100 for duration=10ms
        self.transform = transform

        # tonic.datasets.SSC: split in {'train', 'valid', 'test'}.
        # Each item: (events_np_struct, label_int). events dtype has 't' (us) + 'x' (int).
        self.ssc = tonic.datasets.SSC(save_to=root, split=split)

    def __len__(self) -> int:
        return len(self.ssc)

    def __getitem__(self, i: int):
        events, label = self.ssc[i]
        # events is a structured numpy array with fields 't' (uint64 us) and 'x' (uint32).
        t = np.asarray(events["t"], dtype=np.int64)
        x = np.asarray(events["x"], dtype=np.int64)

        # Stage 1: time-bin into (T, 700).
        frame = np.zeros((self.T, _SSC_N_RAW), dtype=np.float32)
        t_bin = np.clip(t // self.duration_us, 0, self.T - 1)
        # Drop any out-of-range neuron indices defensively.
        valid = (x >= 0) & (x < _SSC_N_RAW)
        np.add.at(frame, (t_bin[valid], x[valid]), 1.0)

        # Stage 2: neuron-bin 700 -> 140 via contiguous sum-pool (n_bins=5).
        binned_len = _SSC_N_RAW // self.n_bins
        binned = frame[:, : binned_len * self.n_bins].reshape(self.T, binned_len, self.n_bins).sum(axis=2)

        if self.transform is not None:
            binned, label = self.transform(binned, label)

        return binned, int(label)


def make_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build SSC train, valid, test DataLoaders.

    Args:
        cfg: OmegaConf config. See docstrings in ``scommander.datasets`` for fields.

    Returns:
        ``(train_loader, valid_loader, test_loader)``.
    """
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    root = str(cfg.dataset.root)
    post_bins = int(cfg.dataset.n_bins)       # 140
    n_bins = _SSC_N_RAW // post_bins           # = 5
    duration = int(cfg.dataset.time_steps)    # ms; 10 -> T=100
    batch_size = int(cfg.training.batch_size)
    aug_enabled = bool(cfg.augmentation.enabled)

    transform = None
    if aug_enabled:
        proba = float(cfg.augmentation.eventdrop.get("drop_prob", 0.5))
        time_prop = float(cfg.augmentation.eventdrop.time_drop_size_pct)
        neuron_size = int(cfg.augmentation.eventdrop.neuron_drop_size)
        aug = TimeNeuronMask(
            proba=proba,
            time_mask_proportion=time_prop,
            neuron_mask_size=neuron_size,
        )
        transform = _Augs(aug)

    train_dataset = BinnedSpikingSpeechCommands(
        root=root, n_bins=n_bins, split="train",
        duration=duration, transform=transform,
    )
    valid_dataset = BinnedSpikingSpeechCommands(
        root=root, n_bins=n_bins, split="valid",
        duration=duration, transform=None,
    )
    test_dataset = BinnedSpikingSpeechCommands(
        root=root, n_bins=n_bins, split="test",
        duration=duration, transform=None,
    )

    train_loader = DataLoader(
        train_dataset, collate_fn=pad_sequence_collate,
        batch_size=batch_size, shuffle=True, num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset, collate_fn=pad_sequence_collate,
        batch_size=batch_size, num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, collate_fn=pad_sequence_collate,
        batch_size=batch_size, num_workers=4,
    )

    return train_loader, valid_loader, test_loader
