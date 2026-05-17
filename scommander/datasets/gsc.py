"""GSC (Google Speech Commands V2) dataset loader.

Ports ``GSC_dataloaders`` + ``build_transform`` + ``GSpeechCommands`` from
``reference/SpikCommander/SCommander/datasets.py:267-457``.

Input is real-valued Mel spectrogram (not spikes). Shape: (B, T=100, F=140).

Transform pipeline (applied in dataset __getitem__):
    PadOrTruncate(16000) -> Resample(16k->8k) -> Spectrogram(n_fft=256, hop=80)
    -> MelScale(140 bins) -> AmplitudeToDB

SpecAugment is applied at batch level in the trainer (not here), matching
the reference ``augs(x, x_len)`` call in ``main_former_v2_gsc*.py:134-136``.

35 classes. Dataset returns (waveform_tensor, label_int, valid_T) per sample,
where valid_T is the number of non-masked time steps (used by SpecAugment).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, Resample
import torch.nn.functional as F

from scommander.utils.seed import set_seed


def _load_wav_no_torchcodec(path: str) -> torch.Tensor:
    """Load a PCM wav → float32 tensor (channels, samples), matching the
    shape/semantics of ``torchaudio.load``.

    torchaudio ≥2.8 routes ``torchaudio.load`` through TorchCodec, which is an
    extra dependency the server env lacks. GSC wavs are plain PCM, so we read
    them directly with soundfile (preferred) or scipy (fallback) — no codec.
    """
    try:
        import soundfile as sf
        data, _sr = sf.read(path, dtype="float32")          # (samples,) or (samples, ch)
    except Exception:
        from scipy.io import wavfile
        _sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]                                 # (1, samples)
    else:
        data = data.T                                        # (ch, samples)
    return torch.from_numpy(np.ascontiguousarray(data))


# 35-class label list copied verbatim from reference datasets.py:423
_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]

_TARGET_TRANSFORM = lambda word: torch.tensor(_LABELS.index(word))


class _PadOrTruncate:
    """Pad or truncate 1-D audio tensor to fixed length.

    Ports ``PadOrTruncate`` from ``reference/augmentations.py:340-350``.
    Reference passes ``audio_length=sample_rate=16000`` (1 second at 16kHz).
    """

    def __init__(self, audio_length: int) -> None:
        self.audio_length = audio_length

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.size(-1) <= self.audio_length:
            return F.pad(sample, (0, self.audio_length - sample.size(-1)))
        return sample[..., : self.audio_length]

    def __repr__(self) -> str:
        return f"PadOrTruncate(audio_length={self.audio_length})"


def _build_transform(cfg) -> torch.nn.Sequential:
    """Build deterministic Mel spectrogram transform (no training augmentation here).

    Reference ``build_transform(config, is_transform=False)`` skips
    SpeedPerturbation/RandomRoll. Baseline always uses ``is_transform=False``.

    Pipeline:
        PadOrTruncate(16000) -> Resample(16k->8k) ->
        Spectrogram(n_fft=256, hop=80, power=2) ->
        MelScale(140, sr=8000, f_min=50, f_max=14000) ->
        AmplitudeToDB()
    """
    sample_rate = 16000
    window_size = int(cfg.dataset.window_size)    # 256
    hop_length = int(cfg.dataset.hop_length)       # 80
    n_mels = int(cfg.dataset.n_bins)               # 140
    f_min = 50
    f_max = 14000
    downsampled_sr = sample_rate // 2              # 8000

    # Use a minimal Compose to avoid torchvision dependency.
    # Each step is a callable: _PadOrTruncate is a plain object;
    # torchaudio transforms are nn.Modules with __call__ == forward.
    steps = [
        _PadOrTruncate(sample_rate),
        Resample(orig_freq=sample_rate, new_freq=downsampled_sr),
        Spectrogram(n_fft=window_size, hop_length=hop_length, power=2),
        MelScale(
            n_mels=n_mels,
            sample_rate=downsampled_sr,
            f_min=f_min,
            f_max=f_max,
            n_stft=window_size // 2 + 1,
        ),
        AmplitudeToDB(),
    ]

    def transform(x: torch.Tensor) -> torch.Tensor:
        for step in steps:
            x = step(x)
        return x

    return transform


class GSpeechCommands(Dataset):
    """Google Speech Commands wrapper returning (mel_tensor, label, valid_T).

    Ports ``GSpeechCommands`` from ``datasets.py:427-457``.

    ``__getitem__`` returns:
        - ``waveform``: ``(T, F)`` Mel spectrogram after transform + squeeze + transpose.
        - ``target``: integer class index (via ``target_transform``).
        - ``valid_T``: number of non-masked time steps (count of rows where not all -100).

    Args:
        root: Dataset root directory.
        split_name: ``'training'``, ``'validation'``, or ``'testing'``.
        transform: Audio transform applied to raw waveform.
        target_transform: Label transform (default: word->index).
        download: Whether to download if missing.
    """

    def __init__(
        self,
        root: str,
        split_name: str,
        transform=None,
        target_transform=None,
        download: bool = True,
    ) -> None:
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform if target_transform is not None else _TARGET_TRANSFORM
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=download, subset=split_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Avoid self.dataset[index] — its internal torchaudio.load routes to
        # TorchCodec (missing dep). Read the wav ourselves from metadata.
        relpath, _sr, label, _spk, _utt = self.dataset.get_metadata(index)
        wav_path = os.path.join(self.dataset._archive, relpath)
        waveform = _load_wav_no_torchcodec(wav_path)

        if self.transform is not None:
            # transform: (1, samples) -> (F, T) -> squeeze -> (T, F) via .t()
            waveform = self.transform(waveform).squeeze().t()

        target = self.target_transform(label)

        # Reference (datasets.py:452-456) returns ``number = len(valid_rows)``
        # — i.e. the total timestep count T, NOT the count of all-valid rows.
        # GSC is fixed-length so every sample uses its full T (no real
        # padding mask). Using .sum() here was a port bug: the f_max(14000) >
        # Nyquist(4000) mel filters are always -100 dB, so .all(dim=1) is
        # uniformly False and .sum() collapsed valid_T to 0.
        valid_T = waveform.shape[0]

        return waveform, target, valid_T


def make_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build GSC train, valid, and test DataLoaders.

    Args:
        cfg: OmegaConf config. Uses:
            - ``cfg.experiment.seed``.
            - ``cfg.dataset.root`` — path for SPEECHCOMMANDS download/cache.
            - ``cfg.dataset.{n_bins, window_size, hop_length}``.
            - ``cfg.training.batch_size``.
            - ``cfg.augmentation.enabled`` — SpecAugment applied in trainer, not here.

    Returns:
        ``(train_loader, valid_loader, test_loader)``.
    """
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    root = str(cfg.dataset.root)
    # torchaudio SPEECHCOMMANDS doesn't auto-create the cache root — ensure it exists.
    os.makedirs(root, exist_ok=True)
    batch_size = int(cfg.training.batch_size)
    transform = _build_transform(cfg)

    train_dataset = GSpeechCommands(root, "training", transform=transform)
    valid_dataset = GSpeechCommands(root, "validation", transform=transform)
    test_dataset = GSpeechCommands(root, "testing", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4,
    )

    return train_loader, valid_loader, test_loader
