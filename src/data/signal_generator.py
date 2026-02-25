# -*- coding: utf-8 -*-
"""
RST — Signal Injection

Uses setigen to inject synthetic narrowband drifting signals
into spectrogram data.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignalParams:
    """Parameters for signal injection."""
    df: float = 2.7939677238464355  # Hz per channel
    dt: float = 18.25361108         # Seconds per time bin
    fch1: float = 0                 # MHz (0 for injection on existing data)

    # SNR parameters
    snr_min: float = 10
    snr_max: float = 80
    snr_base: float = 20
    snr_range: float = 40

    # Signal width parameters
    width_base: float = 50          # Base width in Hz
    width_drift_factor: float = 18  # Additional width per Hz/s drift


class SignalGenerator:
    """Generator for synthetic narrowband drifting SETI signals."""

    def __init__(self, params: Optional[SignalParams] = None, seed: Optional[int] = None):
        self.params = params or SignalParams()
        self.rng = np.random.default_rng(seed)

    def _calculate_drift_rate(self,
                              start_channel: int,
                              width: int,
                              total_time_bins: int) -> float:
        """Calculate drift rate to traverse the observation."""
        direction = self.rng.choice([-1, 1])

        if direction > 0:
            true_slope = total_time_bins / start_channel
        else:
            true_slope = total_time_bins / (start_channel - width)

        # Convert slope to drift rate with small random perturbation
        slope = true_slope * (self.params.dt / self.params.df) + self.rng.uniform(0, 3) * direction
        drift_rate = -1 / slope

        return drift_rate, true_slope

    def _calculate_width(self, drift_rate: float) -> float:
        """Calculate signal width based on drift rate."""
        base_width = self.rng.uniform(0, self.params.width_base)
        drift_width = abs(drift_rate) * self.params.width_drift_factor
        return base_width + drift_width

    def inject_signal(self,
                      data: np.ndarray,
                      snr: Optional[float] = None,
                      start_channel: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a signal into existing spectrogram data.
        Returns (injected data, signal parameters dict).
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)

        if start_channel is None:
            start_channel = self.rng.integers(1, fchans - 1)

        drift_rate, true_slope = self._calculate_drift_rate(start_channel, fchans, tchans)
        width = self._calculate_width(drift_rate)

        # Intercept for tracking
        b = tchans - true_slope * start_channel

        # Create frame from existing data and inject
        frame = stg.Frame.from_data(
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz,
            ascending=False,
            data=data
        )

        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s
            ),
            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
            stg.gaussian_f_profile(width=width * u.Hz),
            stg.constant_bp_profile(level=1)
        )

        signal_info = {
            'snr': snr,
            'drift_rate': drift_rate,
            'start_channel': start_channel,
            'width': width,
            'slope': true_slope,
            'intercept': b
        }

        return frame.data, signal_info

    def inject_cadence_signal(self,
                              stacked_data: np.ndarray,
                              snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """Inject a signal that drifts across a full stacked cadence."""
        return self.inject_signal(stacked_data, snr)


def check_intersection(m1: float, m2: float, b1: float, b2: float,
                       num_observations: int = 6,
                       tchans: int = 16) -> bool:
    """
    Check if two signal paths intersect within observation windows.
    Returns True if signals DON'T intersect.
    """
    if m1 == m2:
        return True  # Parallel lines

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    # ON windows: 0-16, 32-48, 64-80
    # OFF windows: 16-32, 48-64, 80-96
    on_windows = [(0, 16), (32, 48), (64, 80)]
    off_windows = [(16, 32), (48, 64), (80, 96)]

    for start, end in on_windows + off_windows:
        if start <= y <= end:
            return False

    return True
