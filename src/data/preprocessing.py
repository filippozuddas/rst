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


def normalize_robust(spec: np.ndarray) -> np.ndarray:
    """Robust preprocessing pipeline for the RST model.

    1. Log-Scaling: compresses the huge dynamic range of raw radio power.
    2. Per-Observation Z-Score: zero-centers each of the 6 observations
       independently to remove broadband gain differences (Stripe Bias).
       NOTE: no division by 2 — preserves signal amplitude.
    3. Clip: removes extreme RFI peaks that survive normalization.
    """
    # 1. Log-Scaling (prevent log(0) and NaNs)
    spec = np.clip(spec, 1e-6, None)
    spec = np.log10(spec)

    # 2. Per-Observation Normalization (6 observations of 16 rows each)
    for i in range(6):
        obs = spec[i*16:(i+1)*16, :]
        mu = obs.mean()
        sigma = obs.std()
        if sigma < 1e-9:          # guard against constant snippets
            sigma = 1.0
        spec[i*16:(i+1)*16, :] = (obs - mu) / sigma

    # 3. Clip extreme outliers
    return np.clip(spec, -5, 5)


def preprocess_cadence(
    cadence: np.ndarray,
    center_channel: int,
    snippet_width: int = 1024,
) -> np.ndarray:
    """
    Full pipeline: snippet extraction → stack → normalization.

    Args:
        cadence: Array (6, 16, n_freq) raw.
        center_channel: Center channel for the snippet.
        snippet_width: Snippet width (default 1024).

    Returns:
        Array (96, 1024) ready for the model, as float32.
    """
    # 1. Extract snippet at native resolution
    snippet = extract_snippet(cadence, center_channel, snippet_width)

    # 2. Stack the 6 observations
    stacked = stack_cadence(snippet)

    # 3. Normalize (log10 + per-obs z-score + clip)
    return normalize_robust(stacked).astype(np.float32)
