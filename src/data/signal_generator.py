# -*- coding: utf-8 -*-
"""
RST — Signal Injection

Uses setigen to inject synthetic narrowband drifting signals
into spectrogram data. Supports ETI signals (for True samples)
and diverse RFI types (for False samples).
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple, List
from dataclasses import dataclass


# Available RFI types for False sample generation
RFI_TYPES: List[str] = [
    'linear',        # Standard linear drift (same as ETI but in all obs)
    'stationary',    # Fixed frequency with jitter
    'random_walk',   # Frequency wanders randomly over time
    'broadband',     # Wide bandwidth, zero or low drift
    'scintillating', # Intensity oscillates sinusoidally over time
    'pulsed',        # Periodic Gaussian bursts
]


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
    """
    Generator for synthetic SETI signals.

    Methods:
        inject_signal: ETI-like narrowband drifting signal (for True samples)
        inject_rfi_signal: Diverse RFI patterns (for False samples)
    """

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

    def _make_frame(self, data: np.ndarray) -> stg.Frame:
        """Create a setigen Frame from existing data."""
        return stg.Frame.from_data(
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz,
            ascending=False,
            data=data
        )

    # ETI signal injection (used for True samples)
    def inject_signal(self,
                      data: np.ndarray,
                      snr: Optional[float] = None,
                      start_channel: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a narrowband drifting ETI signal.
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

        frame = self._make_frame(data)

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
        """Inject an ETI signal that drifts across a full stacked cadence."""
        return self.inject_signal(stacked_data, snr)

    # RFI signal injection (used for False samples)
    def inject_rfi_signal(self,
                          data: np.ndarray,
                          snr: Optional[float] = None,
                          rfi_type: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a realistic RFI signal into spectrogram data.

        Args:
            data: Input spectrogram (tchans, fchans).
            snr: Signal-to-noise ratio. If None, sampled randomly.
            rfi_type: One of RFI_TYPES. If None, picked randomly.

        Returns:
            (injected data, info dict with rfi_type and parameters).
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)

        if rfi_type is None:
            rfi_type = self.rng.choice(RFI_TYPES)

        start_channel = self.rng.integers(1, fchans - 1)
        frame = self._make_frame(data)
        f_start = frame.get_frequency(index=start_channel)
        intensity = frame.get_intensity(snr=snr)

        # Build path, t_profile, f_profile based on RFI type
        if rfi_type == 'linear':
            # Same as ETI but will be injected in ALL obs 
            drift_rate, _ = self._calculate_drift_rate(start_channel, fchans, tchans)
            width = self._calculate_width(drift_rate)
            path = stg.constant_path(f_start=f_start,
                                     drift_rate=drift_rate * u.Hz / u.s)
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'stationary':
            # RFI fixed in frequency with random jitter around center
            spread = self.rng.uniform(50, 500) * u.Hz
            drift_rate = self.rng.uniform(-0.1, 0.1)
            width = self.rng.uniform(10, 80) * u.Hz
            path = stg.simple_rfi_path(f_start=f_start,
                                       drift_rate=drift_rate * u.Hz / u.s,
                                       spread=spread,
                                       spread_type='normal',
                                       rfi_type='stationary')
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.box_f_profile(width=width)

        elif rfi_type == 'random_walk':
            # RFI that wanders in frequency over time
            spread = self.rng.uniform(30, 300) * u.Hz
            drift_rate = self.rng.uniform(-0.5, 0.5)
            width = self._calculate_width(drift_rate)
            path = stg.simple_rfi_path(f_start=f_start,
                                       drift_rate=drift_rate * u.Hz / u.s,
                                       spread=spread,
                                       spread_type='normal',
                                       rfi_type='random_walk')
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'broadband':
            # Wide bandwidth, zero or very low drift
            drift_rate = self.rng.uniform(-0.05, 0.05)
            width = self.rng.uniform(200, 1000) * u.Hz
            path = stg.constant_path(f_start=f_start,
                                     drift_rate=drift_rate * u.Hz / u.s)
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.box_f_profile(width=width)

        elif rfi_type == 'scintillating':
            # Intensity oscillates sinusoidally over time
            drift_rate, _ = self._calculate_drift_rate(start_channel, fchans, tchans)
            width = self._calculate_width(drift_rate)
            period = self.rng.uniform(50, 300) * u.s
            amplitude = intensity * self.rng.uniform(0.3, 0.8)
            path = stg.constant_path(f_start=f_start,
                                     drift_rate=drift_rate * u.Hz / u.s)
            t_prof = stg.sine_t_profile(period=period,
                                        amplitude=amplitude,
                                        level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'pulsed':
            # Periodic Gaussian bursts (pulsar-like RFI)
            drift_rate = self.rng.uniform(-0.3, 0.3)
            width = self.rng.uniform(10, 60) * u.Hz
            pulse_period = self.rng.uniform(30, 200) * u.s
            pulse_width = self.rng.uniform(10, 50) * u.s
            path = stg.constant_path(f_start=f_start,
                                     drift_rate=drift_rate * u.Hz / u.s)
            t_prof = stg.periodic_gaussian_t_profile(
                pulse_width=pulse_width,
                period=pulse_period,
                amplitude=intensity,
                level=intensity * 0.1,  # Low baseline between pulses
                pulse_direction='up'
            )
            f_prof = stg.gaussian_f_profile(width=width)

        else:
            raise ValueError(f"Unknown RFI type: {rfi_type}. Choose from {RFI_TYPES}")

        frame.add_signal(path, t_prof, f_prof, stg.constant_bp_profile(level=1))

        signal_info = {
            'rfi_type': rfi_type,
            'snr': snr,
            'start_channel': start_channel,
        }

        return frame.data, signal_info


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
