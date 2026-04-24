"""Augmentations: TimeNeuronMask for spike (SHD/SSC), SpecAugment for Mel spectrograms (GSC)."""

from scommander.augmentations.masking import TimeNeuronMask
from scommander.augmentations.spec_aug import SpecAugment

__all__ = ["TimeNeuronMask", "SpecAugment"]
