# -*- coding: utf-8 -*-
"""
RST — Radio Spectrogram Preprocessing

Transforms raw SRT data into the format required by the RST model.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def extract_snippet(
    cadence: np.ndarray,
    center_channel: int,
    snippet_width: int = 1024,
) -> np.ndarray:
    """
    Extract a frequency "crop" from the full cadence at native resolution.
    Args:
        cadence: Array of shape (6, 16, n_freq)
        center_channel: Center coordinate
        snippet_width: Crop width
    Returns:
        Array of shape (6, 16, snippet_width).
    """
    n_freq = cadence.shape[2]
    half = snippet_width // 2

    # Compute window boundaries
    start = center_channel - half
    end = center_channel + half

    # Handle edges: if the window goes out of bounds, shift it
    if start < 0:
        start = 0
        end = snippet_width
    if end > n_freq:
        end = n_freq
        start = n_freq - snippet_width

    return cadence[:, :, start:end]


def stack_cadence(cadence: np.ndarray) -> np.ndarray:
    """
    Stack the 6 observations vertically into a single spectrogram.
    Transforms (6, 16, width) -> (96, width).
    """
    # (6, 16, 1024) → reshape → (96, 1024)
    return cadence.reshape(-1, cadence.shape[-1])


def normalize_zscore(
    spectrogram: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Z-score normalization: (x - μ) / (σ × 2)
    Factor of 2 follows the AST paper convention.
    """
    return (spectrogram - mean) / (std * 2)


def compute_dataset_stats(
    spectrograms: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of the dataset.

    These values should be computed ONCE on the training set
    and then used to normalize train, val, and test sets.

    Args:
        spectrograms: Array of shape (N, 96, 1024) with N spectrograms.

    Returns:
        Tuple (mean, std).
    """
    mean = float(np.mean(spectrograms))
    std = float(np.std(spectrograms))
    return mean, std


def preprocess_cadence(
    cadence: np.ndarray,
    center_channel: int,
    snippet_width: int = 1024,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """
    Full pipeline: snippet extraction → stack → normalization.

    Args:
        cadence: Array (6, 16, n_freq) raw.
        center_channel: Center channel for the snippet.
        snippet_width: Snippet width (default 1024).
        mean: Mean for normalization (if None, skip normalization).
        std: Std for normalization (if None, skip normalization).

    Returns:
        Array (96, 1024) ready for the model, as float32.
    """
    # 1. Extract snippet at native resolution
    snippet = extract_snippet(cadence, center_channel, snippet_width)

    # 2. Stack the 6 observations
    stacked = stack_cadence(snippet)

    # 3. Normalize (if stats are available)
    if mean is not None and std is not None:
        stacked = normalize_zscore(stacked, mean, std)

    return stacked.astype(np.float32)
