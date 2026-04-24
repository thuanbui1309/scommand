"""SHD (Spiking Heidelberg Digits) dataset loader.

Reads events directly from SpikingJelly's cached h5 files and bins to frames
with equal-time windows. Bypasses ``spikingjelly.datasets.shd`` framing entirely:

- spikingjelly 0.0.0.0.14+ interprets ``duration`` in the h5's t units (seconds
  for SHD), so ``duration=10`` gives T=1 bin instead of T=100. Verified on
  server: both our original port and the reference author's code both produce
  (1, 700) frames on this stack.
- ``frames_number=100, split_by='time'`` crashes with IndexError on empty bins.
- ``split_by='number'`` works but produces equal-event bins (not equal-time),
  breaking paper semantics.

Our custom loader:
  - Open h5 once per worker; lazy-read events per sample.
  - Equal-time binning over fixed 1.0s clip: ``bin_width = clip_duration_s / T``.
  - Neuron sum-pool: 700 -> 700 // n_bins = 140.
  - Returns (T, N) float32 tensors — all samples same shape, no pad needed.

Reference: paper + author code ``reference/SpikCommander/SCommander/datasets.py:
283-334``; diagnosed at
``plans/reports/decision-260424-2230-shd-custom-time-binning.md``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# We still inherit spikingjelly's SpikingHeidelbergDigits indirectly via
# the download/extract side-effect: we call it once just to populate
# ``root/extract/shd_{train,test}.h5``. After that we read h5 ourselves.
from spikingjelly.datasets.shd import SpikingHeidelbergDigits

from scommander.augmentations.masking import TimeNeuronMask
from scommander.utils.seed import set_seed


# SHD fixed clip length (audio is 1 second of Heidelberg digit utterances).
_SHD_CLIP_DURATION_S = 1.0
_SHD_N_SENSOR = 700


class _Augs:
    """Thin wrapper applying a list of (x, y) -> (x, y) augmentations."""

    def __init__(self, aug: TimeNeuronMask) -> None:
        self._aug = aug

    def __call__(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        return self._aug(x, y)

    def __repr__(self) -> str:
        return f"Augs([{self._aug.__class__.__name__}])"


class BinnedSpikingHeidelbergDigits(Dataset):
    """SHD dataset returning ``(T, N=140)`` equal-time-binned frames.

    Args:
        root: Dataset root. Must already contain
            ``root/extract/shd_{train,test}.h5`` (populated by SpikingJelly
            on first access).
        train: Train split (True) or test split (False).
        n_bins: Neuron-axis bin width. Reference: 5 (700//5 = 140).
        time_steps: Target T frame count. Reference: 100 for 10ms bins.
        clip_duration_s: Clip length in seconds. SHD default 1.0.
        transform: Optional ``(frame, label) -> (frame, label)`` hook.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        n_bins: int = 5,
        time_steps: int = 100,
        clip_duration_s: float = _SHD_CLIP_DURATION_S,
        transform: Optional[Callable] = None,
    ) -> None:
        self._h5_path = os.path.join(
            root, "extract", f"shd_{'train' if train else 'test'}.h5"
        )
        if not os.path.exists(self._h5_path):
            raise FileNotFoundError(
                f"SHD h5 missing at {self._h5_path}. Run SpikingHeidelbergDigits "
                f"once to populate via download/extract."
            )

        # Cache labels (small: int per sample). Events loaded lazily per item.
        with h5py.File(self._h5_path, "r") as f:
            self._labels = np.asarray(f["labels"])

        self.n_bins = n_bins
        self.time_steps = time_steps
        self.clip_duration_s = clip_duration_s
        self.transform = transform
        self._bin_width = clip_duration_s / time_steps
        self._binned_n = _SHD_N_SENSOR // n_bins  # 140

        # Opened per-worker on first __getitem__ (h5py and DataLoader workers
        # don't mix across fork boundaries cleanly).
        self._h5_file: Optional[h5py.File] = None

    def _ensure_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self._h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        h5 = self._ensure_h5()
        t = np.asarray(h5["spikes"]["times"][i])
        x = np.asarray(h5["spikes"]["units"][i], dtype=np.int64)
        label = int(self._labels[i])

        # Equal-time binning: (T, 700).
        t_idx = np.clip((t / self._bin_width).astype(np.int64), 0, self.time_steps - 1)
        frame = np.zeros((self.time_steps, _SHD_N_SENSOR), dtype=np.float32)
        valid = (x >= 0) & (x < _SHD_N_SENSOR)
        np.add.at(frame, (t_idx[valid], x[valid]), 1.0)

        # Neuron sum-pool: 700 -> 140.
        binned = frame[:, : self._binned_n * self.n_bins].reshape(
            self.time_steps, self._binned_n, self.n_bins
        ).sum(axis=2)

        if self.transform is not None:
            binned, label = self.transform(binned, label)
        return binned, label


def _equal_len_collate(batch):
    """Collate fn for equal-length (T, N) samples.

    Returns ``(x_batch, y_batch, x_len)`` matching trainer's unpack signature.
    All samples share T, so x_len = [T] * B (trainer's padded_sequence_mask
    produces an all-True attention mask under this condition — equivalent
    to no masking, matching paper which has no padding for SHD).
    """
    xs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x, _ in batch])
    ys = torch.as_tensor([y for _, y in batch], dtype=torch.long)
    x_len = torch.full((xs.size(0),), xs.size(1), dtype=torch.long)
    return xs, ys, x_len


def _populate_h5_cache(root: str) -> None:
    """Trigger SpikingJelly's download/extract side-effect once so h5 files exist.

    We use data_type='event' which skips the buggy duration/frames_number framing;
    all we need is ``root/extract/shd_{train,test}.h5`` on disk.
    """
    if os.path.exists(os.path.join(root, "extract", "shd_train.h5")) and os.path.exists(
        os.path.join(root, "extract", "shd_test.h5")
    ):
        return
    # Touch both splits to kick off download + extract.
    SpikingHeidelbergDigits(root, train=True, data_type="event")
    SpikingHeidelbergDigits(root, train=False, data_type="event")


def make_loaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Build SHD train and test DataLoaders.

    Args:
        cfg: OmegaConf config. Uses:
            - ``cfg.experiment.seed`` — seeded before dataset init.
            - ``cfg.dataset.root`` — filesystem path to SHD data.
            - ``cfg.dataset.n_bins`` — post-binning neuron count (140 ->
              bin_width=5 over 700 raw sensors).
            - ``cfg.dataset.time_steps`` — target T frame count.
            - ``cfg.training.batch_size``.
            - ``cfg.augmentation.enabled`` and ``.eventdrop.*``.

    Returns:
        ``(train_loader, test_loader)`` — SHD test set doubles as validation.
    """
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    root = str(cfg.dataset.root)
    os.makedirs(root, exist_ok=True)
    _populate_h5_cache(root)

    post_bins = int(cfg.dataset.n_bins)          # 140 post-binning
    n_bins = _SHD_N_SENSOR // post_bins           # = 5 bin-width
    time_steps = int(cfg.dataset.time_steps)     # 100
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
        root=root, train=True, n_bins=n_bins,
        time_steps=time_steps, transform=transform,
    )
    test_dataset = BinnedSpikingHeidelbergDigits(
        root=root, train=False, n_bins=n_bins,
        time_steps=time_steps, transform=None,
    )

    train_loader = DataLoader(
        train_dataset, collate_fn=_equal_len_collate,
        batch_size=batch_size, shuffle=True, num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, collate_fn=_equal_len_collate,
        batch_size=batch_size, num_workers=4,
    )

    return train_loader, test_loader
