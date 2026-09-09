# -*- coding: utf-8 -*-
"""
RST — Data Augmentation

Implements augmentation techniques for training:
1. SpecAugment: masks random frequency and time bands
2. Mixup: blends two spectrograms (and their labels) with a random weight
"""

import torch
import numpy as np
from typing import Tuple


def spec_augment(
    spectrogram: torch.Tensor,
    freq_mask_param: int = 32,
    time_mask_param: int = 8,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> torch.Tensor:
    """
    Apply SpecAugment: mask random bands in the spectrogram.
    Improves generalization by masking frequency and time bands.
    """
    spec = spectrogram.clone()
    time_bins, freq_bins = spec.shape

    # Frequency mask
    for _ in range(num_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, max(1, freq_bins - f), (1,)).item()
        spec[:, f0:f0 + f] = 0.0

    # Time mask
    for _ in range(num_time_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, max(1, time_bins - t), (1,)).item()
        spec[t0:t0 + t, :] = 0.0

    return spec


def mixup(
    spec1: torch.Tensor,
    label1: torch.Tensor,
    spec2: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup: blend two samples and their labels with a random weight.
    Returns (mixed_spectrogram, mixed_label).
    """
    # Draw λ from the Beta distribution
    lam = np.random.beta(alpha, alpha)
    # Ensure λ is at least 0.5 (the first sample is the "dominant" one)
    lam = max(lam, 1.0 - lam)

    spec_mix = lam * spec1 + (1.0 - lam) * spec2
    label_mix = lam * label1 + (1.0 - lam) * label2

    return spec_mix, label_mix
