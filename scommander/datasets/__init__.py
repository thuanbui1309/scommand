"""Dataset loaders: SHD, SSC (via SpikingJelly), GSC (via torchaudio).

Public API
----------
make_loaders(dataset_name, cfg)
    Dispatcher returning DataLoaders for the named dataset.

padded_sequence_mask(lens, T_max)
    Re-exported from spikingjelly.datasets for trainer convenience.
    Signature: (lens: Tensor) -> (T, B) bool mask (True=valid, False=pad).
    Trainer must transpose to (B, T) before passing to model as attention_mask.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

# padded_sequence_mask: (lens: Tensor[B]) -> (T_max, B) bool mask
# True = valid timestep, False = padding. Defined in spikingjelly.datasets.
# Trainer transposes -> (B, T) before model forward.
from spikingjelly.datasets import padded_sequence_mask  # noqa: F401  (re-export)

from scommander.datasets import shd as _shd
from scommander.datasets import ssc as _ssc
from scommander.datasets import gsc as _gsc

_LOADERS = {
    "shd": _shd.make_loaders,
    "ssc": _ssc.make_loaders,
    "gsc": _gsc.make_loaders,
}


def make_loaders(dataset_name: str, cfg) -> tuple[DataLoader, ...]:
    """Dispatch to the appropriate per-dataset loader factory.

    Args:
        dataset_name: One of ``'shd'``, ``'ssc'``, ``'gsc'``.
        cfg: OmegaConf config (merged base + dataset + variant).

    Returns:
        SHD: ``(train_loader, test_loader)``
        SSC: ``(train_loader, valid_loader, test_loader)``
        GSC: ``(train_loader, valid_loader, test_loader)``

    Raises:
        ValueError: If ``dataset_name`` is not recognised.
    """
    if dataset_name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Supported: {sorted(_LOADERS)}"
        )
    return _LOADERS[dataset_name](cfg)


__all__ = ["make_loaders", "padded_sequence_mask"]
