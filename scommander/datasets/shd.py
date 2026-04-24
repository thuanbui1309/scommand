"""SHD (Spiking Heidelberg Digits) dataset loader.

Ports ``SHD_dataloaders`` + ``BinnedSpikingHeidelbergDigits`` from
``reference/SpikCommander/SCommander/datasets.py:212-334``.

Dataset shape: (T=100, B=256, N=140). Binning: 700 neurons / 5 bins = 140.
The SpikingJelly .h5 cache is seed-sensitive — ``set_seed`` is called before
``__init__`` to ensure deterministic cache-key generation (see port-map §9).

SHD has no separate validation split — test set is used for validation.
``make_loaders`` returns (train_loader, test_loader).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from torch.utils.data import DataLoader

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
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


class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    """SHD with neuron binning: 700 raw channels -> n_bins per group -> N features.

    Inherits SpikingJelly's SpikingHeidelbergDigits for .h5 loading and
    cache management. Overrides ``__getitem__`` to bin along the neuron axis
    via sum-pooling.

    Args:
        root: Path to dataset root (contains SHD .h5 files).
        n_bins: Bin width for neuron axis. Reference: 5 (700//5 = 140 features).
        train: True for training split, False for test.
        data_type: Must be ``'frame'`` for binned operation.
        duration: Frame duration in ms. Reference: 10 (-> T=100 for 1s clips).
        transform: Optional ``(frame, label) -> (frame, label)`` callable.
            Applied after binning.
        target_transform: Optional label transform (unused in baseline).
    """

    def __init__(
        self,
        root: str,
        n_bins: int,
        train: bool = True,
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
            root, train, data_type, frames_number, split_by, duration,
            custom_integrate_function, custom_integrated_frames_dir_name,
            transform, target_transform,
        )
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == "frame":
            frames = np.load(self.frames_path[i], allow_pickle=True)["frames"].astype(np.float32)
            label = self.frames_label[i]

            # Sum-pool along neuron axis: (N_raw, T) -> (N_raw//n_bins, T)
            # then transpose to (T, N) for the model's (B, T, N) convention
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


def make_loaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Build SHD train and test DataLoaders.

    Args:
        cfg: OmegaConf config. Uses:
            - ``cfg.experiment.seed`` — seeded before dataset init (cache determinism).
            - ``cfg.dataset.root`` — filesystem path to SHD data.
            - ``cfg.dataset.n_bins`` — neuron bin width (reference: 140 post-binning
              means n_bins=5 applied to 700 raw neurons).
            - ``cfg.dataset.time_steps`` — frame duration in ms (reference: 10 -> T=100).
            - ``cfg.training.batch_size``.
            - ``cfg.augmentation.enabled``.
            - ``cfg.augmentation.eventdrop.*`` — time/neuron mask params.

    Returns:
        ``(train_loader, test_loader)`` — SHD test set doubles as validation.
    """
    seed = int(cfg.experiment.seed)
    # Critical: seed before dataset __init__ for Tonic cache-key determinism
    set_seed(seed)

    root = str(cfg.dataset.root)
    # SpikingJelly mkdir's subdirs but not the root — ensure it exists first.
    os.makedirs(root, exist_ok=True)
    # Reference uses n_bins=5; dataset.n_bins in yaml stores post-binning width=140.
    # Derive raw bin_width: 700 // post_bin_width = 5
    post_bins = int(cfg.dataset.n_bins)      # 140 post-binning neurons
    raw_neurons = 700
    n_bins = raw_neurons // post_bins         # = 5 bin-width

    # SpikingJelly's `duration` is ms PER FRAME. Our yaml stores the target T in
    # `time_steps` (=100) and the per-frame width in `bin_width_ms` (=10). The
    # spikingjelly arg we need is bin_width_ms, NOT time_steps.
    duration = int(cfg.dataset.bin_width_ms)  # 10ms per frame -> T=1000/10=100
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

    train_dataset = BinnedSpikingHeidelbergDigits(
        root=root,
        n_bins=n_bins,
        train=True,
        data_type="frame",
        duration=duration,
        transform=transform,
    )
    test_dataset = BinnedSpikingHeidelbergDigits(
        root=root,
        n_bins=n_bins,
        train=False,
        data_type="frame",
        duration=duration,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=batch_size,
        num_workers=4,
    )

    return train_loader, test_loader
