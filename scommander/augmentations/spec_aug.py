"""SpecAugment for Mel spectrogram inputs (GSC).

Ports ``SpecAugment`` nn.Module from
``reference/SpikCommander/SCommander/datasets.py:81-122``.

The reference applies SpecAugment at **batch level** inside the training loop
(not as a dataset transform). This module is instantiated in ``trainer.py``
and called with ``(x, x_len)`` before the forward pass.

Input tensor shape: ``(B, T, F)`` (B=batch, T=time, F=frequency bins).
Reference transposes to ``(B, F, T)`` before applying torchaudio masking,
then transposes back.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class SpecAugment(nn.Module):
    """Adaptive SpecAugment with per-sample time mask sizing.

    Matches reference ``SpecAugment`` exactly:
    - Frequency masking: ``n_freq_masks`` masks each up to ``freq_mask_size`` bins.
    - Time masking: per-sample T clipped to ``x_len[b]``; mask size = ``int(time_mask_pct * x_len[b])``.

    Args:
        n_freq_masks: Number of frequency masks (mF in reference). GSC: 1.
        freq_mask_size: Max frequency mask width in bins (F in reference). GSC: 10.
        n_time_masks: Number of time masks per sample (mT in reference). GSC: 1.
        time_mask_pct: Adaptive time mask fraction of valid length (pS). GSC: 0.25.
    """

    def __init__(
        self,
        n_freq_masks: int,
        freq_mask_size: int,
        n_time_masks: int,
        time_mask_pct: float,
    ) -> None:
        super().__init__()
        self.n_freq_masks = n_freq_masks
        self.freq_mask_size = freq_mask_size
        self.n_time_masks = n_time_masks
        self.time_mask_pct = time_mask_pct

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment in-place (training only).

        Args:
            x: ``(B, T, F)`` mel spectrogram tensor.
            x_len: ``(B,)`` integer tensor of valid time lengths per sample.

        Returns:
            ``(B, T, F)`` augmented tensor (same device, same dtype).
        """
        # Torchaudio masking operates on (B, F, T) — transpose and back
        x = x.transpose(1, 2)   # (B, T, F) -> (B, F, T)

        # Frequency masking — batch-level (iid_masks=False = same mask per batch)
        for _ in range(self.n_freq_masks):
            x = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self.freq_mask_size, iid_masks=False
            ).forward(x)

        # Time masking — per-sample, clipped to valid length
        for b in range(x.size(0)):
            T_valid = int(x_len[b].item())
            time_mask_param = int(self.time_mask_pct * T_valid)
            for _ in range(self.n_time_masks):
                x[b:b + 1, :, :T_valid] = torchaudio.transforms.TimeMasking(
                    time_mask_param=time_mask_param
                ).forward(x[b:b + 1, :, :T_valid])

        x = x.transpose(1, 2)   # (B, F, T) -> (B, T, F)
        return x
