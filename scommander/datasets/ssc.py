"""SSC (Spiking Speech Commands) dataset loader.

Ports ``SSC_dataloaders`` + ``BinnedSpikingSpeechCommands`` from
``reference/SpikCommander/SCommander/datasets.py:244-390``.

Dataset shape: (T=100, B=256, N=140). Same binning as SHD (700->140).
SSC has real train/valid/test splits — returns 3 loaders.
35 classes.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from torch.utils.data import DataLoader

from spikingjelly.datasets.shd import SpikingSpeechCommands
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


class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    """SSC with neuron binning: 700 raw channels -> n_bins per group -> 140 features.

    Inherits SpikingJelly's SpikingSpeechCommands for .h5 loading.
    Overrides ``__getitem__`` to bin along the neuron axis via sum-pooling.

    Args:
        root: Path to dataset root.
        n_bins: Bin width for neuron axis. Reference: 5 (700//5 = 140 features).
        split: ``'train'``, ``'valid'``, or ``'test'``.
        data_type: Must be ``'frame'`` for binned operation.
        duration: Frame duration in ms (reference: 10 -> T=100).
        transform: Optional ``(frame, label) -> (frame, label)`` callable.
        target_transform: Optional label transform (unused in baseline).
    """

    def __init__(
        self,
        root: str,
        n_bins: int,
        split: str = "train",
        data_type: str = "frame",
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root, split, data_type, frames_number, split_by, duration,
            custom_integrate_function, custom_integrated_frames_dir_name,
            transform, target_transform,
        )
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == "frame":
            frames = np.load(self.frames_path[i], allow_pickle=True)["frames"].astype(np.float32)
            label = self.frames_label[i]

            # SpikingJelly frame layout: (T, N_raw) e.g. (100, 700).
            # Bin along neuron axis (axis=1): sum groups of n_bins neurons.
            # binned_len = 700 // 5 = 140.  Output shape: (T, 140).
            binned_len = frames.shape[1] // self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
            for k in range(binned_len):
                binned_frames[:, k] = frames[:, self.n_bins * k: self.n_bins * (k + 1)].sum(axis=1)

            if self.transform is not None:
                binned_frames, label = self.transform(binned_frames, label)

            return binned_frames, label

        # Fallback for event mode (unused in baseline)
        events = {"t": self.h5_file["spikes"]["times"][i], "x": self.h5_file["spikes"]["units"][i]}
        label = self.h5_file["labels"][i]
        if self.transform is not None:
            events, label = self.transform(events, label)
        return events, label


def make_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build SSC train, valid, and test DataLoaders.

    Args:
        cfg: OmegaConf config. Uses:
            - ``cfg.experiment.seed`` — seeded before dataset init.
            - ``cfg.dataset.root`` — filesystem path.
            - ``cfg.dataset.n_bins`` — post-binning neuron count (140).
            - ``cfg.dataset.time_steps`` — frame duration ms (10 -> T=100).
            - ``cfg.training.batch_size``.
            - ``cfg.augmentation.enabled``.
            - ``cfg.augmentation.eventdrop.*``.

    Returns:
        ``(train_loader, valid_loader, test_loader)``.
    """
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    root = str(cfg.dataset.root)
    post_bins = int(cfg.dataset.n_bins)   # 140
    raw_neurons = 700
    n_bins = raw_neurons // post_bins      # = 5

    duration = int(cfg.dataset.time_steps)
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
        data_type="frame", duration=duration, transform=transform,
    )
    valid_dataset = BinnedSpikingSpeechCommands(
        root=root, n_bins=n_bins, split="valid",
        data_type="frame", duration=duration, transform=None,
    )
    test_dataset = BinnedSpikingSpeechCommands(
        root=root, n_bins=n_bins, split="test",
        data_type="frame", duration=duration, transform=None,
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
