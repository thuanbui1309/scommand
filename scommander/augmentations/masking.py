"""Time-Neuron mask augmentation for binned spike datasets (SHD/SSC).

Ports ``TimeNeurons_mask_aug`` from
``reference/SpikCommander/SCommander/datasets.py:173-191``.

RNG: uses ``np.random.uniform`` / ``np.random.randint`` exactly as reference —
do NOT replace with torch RNG (reproduces reference drift sources for bisect).
"""

from __future__ import annotations

import numpy as np


class TimeNeuronMask:
    """Randomly zero a contiguous time window and a random neuron stripe.

    Applies two independent masks with probability ``proba`` each:

    1. **Time mask**: zeros ``floor(time_mask_proportion * T)`` consecutive
       time steps starting at a uniformly-sampled offset.
    2. **Neuron mask**: zeros a random-width stripe of up to
       ``neuron_mask_size`` neurons at a uniformly-sampled offset.

    Args:
        proba: Probability of applying each mask independently.
            Reference uses 0.5 for both SHD and SSC.
        time_mask_proportion: Fraction of T to zero. Reference: 0.2 (SHD),
            0.1 (SSC).
        neuron_mask_size: Maximum neuron stripe width. Reference: 20 (SHD),
            10 (SSC).
    """

    def __init__(
        self,
        proba: float,
        time_mask_proportion: float,
        neuron_mask_size: int,
    ) -> None:
        self.proba = proba
        self.time_mask_proportion = time_mask_proportion
        self.neuron_mask_size = neuron_mask_size

    def __call__(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        """Apply time and neuron masks in-place.

        Args:
            x: ``(T, N)`` float32 binned spike frame (numpy array).
            y: Integer class label (passed through unchanged).

        Returns:
            ``(x, y)`` — x may be modified in-place.
        """
        # Time mask: zero a contiguous time window
        if np.random.uniform() < self.proba:
            mask_size = int(self.time_mask_proportion * x.shape[0])
            ind = np.random.randint(0, x.shape[0] - mask_size)
            x[ind:ind + mask_size, :] = 0

        # Neuron mask: zero a random-width neuron stripe
        if np.random.uniform() < self.proba:
            mask_size = np.random.randint(0, self.neuron_mask_size)
            ind = np.random.randint(0, x.shape[1] - self.neuron_mask_size)
            x[:, ind:ind + mask_size] = 0

        return x, y
