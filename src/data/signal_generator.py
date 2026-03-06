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
    'scintillating', # Intensity oscillates sinusoidally over time
    'pulsed',        # Periodic Gaussian bursts
]

# Default weights for weighted RFI type selection
RFI_TYPE_WEIGHTS: List[float] = [0.40, 0.25, 0.15, 0.12, 0.08]


@dataclass
class SignalParams:
    """Parameters for signal injection."""
    df: float = 2.7939677238464355  # Hz per channel
    dt: float = 18.25361108         # Seconds per time bin
    fch1: float = 0                 # MHz (0 for injection on existing data)

    # SNR parameters — log-uniform sampling in [snr_min, snr_max]
    snr_min: float = 5.0
    snr_max: float = 50.0

    # Signal width parameters (|DR|×dt + U(offset_min, offset_max) Hz)
    width_offset_min: float = 5.0    # Min offset above drift-induced width (Hz)
    width_offset_max: float = 55.0   # Max offset above drift-induced width (Hz)

    # Drift rate parameters — log-uniform with random sign
    max_drift_rate: float = 4.0      # Max drift rate in Hz/s
    min_nonzero_drift: float = 0.01  # Min non-zero drift rate for log sampling
    zero_drift_prob: float = 0.05    # Probability of exactly zero drift

    # Frequency profile selection
    freq_profiles: tuple = ('gaussian', 'sinc2')
    freq_profile_weights: tuple = (0.8, 0.2)

    # Temporal profile selection
    time_profiles: tuple = ('constant', 'scintillating')
    time_profile_weights: tuple = (0.6, 0.4)

    # RFI type weights
    rfi_types: tuple = ('linear', 'stationary', 'random_walk', 'scintillating', 'pulsed')
    rfi_type_weights: tuple = (0.40, 0.25, 0.15, 0.12, 0.08)


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

    # Sampling helpers
    def _sample_snr(self) -> float:
        """Sample SNR from a log-uniform distribution.

        Log-uniform produces more low-SNR samples, which better reflects
        the expected distribution of real signals (weak signals are far
        more common than strong ones).

        For [5, 50]: median ≈ √(5×50) ≈ 15.8 (vs 27.5 for uniform).
        """
        log_min = np.log10(self.params.snr_min)
        log_max = np.log10(self.params.snr_max)
        return float(10 ** self.rng.uniform(log_min, log_max))

    def _sample_drift_rate(self) -> Tuple[float, float]:
        """Sample drift rate from a log-uniform distribution with random sign.

        Includes a small probability of exactly zero drift (default 5%).
        Log-uniform concentrates most samples at low |DR| (≤ 0.3 Hz/s),
        matching the distribution of interesting candidates found so far.

        Returns (drift_rate, true_slope) tuple.
        """
        if self.rng.random() < self.params.zero_drift_prob:
            drift_rate = 0.0
        else:
            log_min = np.log10(self.params.min_nonzero_drift)
            log_max = np.log10(self.params.max_drift_rate)
            magnitude = 10 ** self.rng.uniform(log_min, log_max)
            drift_rate = float(magnitude * self.rng.choice([-1, 1]))

        # Compute true_slope for metadata / intersection checks
        if abs(drift_rate) < 1e-9:
            true_slope = 1e9  # effectively infinite (vertical signal)
        else:
            slope = -1.0 / drift_rate
            true_slope = slope / (self.params.dt / self.params.df)

        return drift_rate, true_slope

    def _calculate_width(self, drift_rate: float) -> float:
        """Width formula: |DR|×dt + U(offset_min, offset_max).

        The |DR|×dt term compensates for the drift within a single time bin,
        and the random offset prevents quantization artefacts.
        """
        drift_component = abs(drift_rate) * self.params.dt
        offset = self.rng.uniform(self.params.width_offset_min,
                                  self.params.width_offset_max)
        return drift_component + offset

    def _select_f_profile(self, width: float):
        """Select frequency profile based on configured weights.

        Returns a setigen frequency profile function.
        Available profiles: gaussian (default), sinc² (sinc2).
        """
        profiles = list(self.params.freq_profiles)
        weights = list(self.params.freq_profile_weights)
        choice = self.rng.choice(profiles, p=weights)

        if choice == 'gaussian':
            return stg.gaussian_f_profile(width=width * u.Hz), choice
        elif choice == 'sinc2':
            return stg.sinc2_f_profile(width=width * u.Hz), choice
        else:
            # Fallback to gaussian for unknown profiles
            return stg.gaussian_f_profile(width=width * u.Hz), 'gaussian'

    def _select_t_profile(self, intensity: float):
        """Select temporal profile based on configured weights.

        Returns a setigen temporal profile function.
        Available profiles: constant, scintillating (sine modulation).
        """
        profiles = list(self.params.time_profiles)
        weights = list(self.params.time_profile_weights)
        choice = self.rng.choice(profiles, p=weights)

        if choice == 'constant':
            return stg.constant_t_profile(level=intensity), choice
        elif choice == 'scintillating':
            period = self.rng.uniform(50, 300) * u.s
            amplitude = intensity * self.rng.uniform(0.2, 0.5)
            return stg.sine_t_profile(period=period,
                                       amplitude=amplitude,
                                       level=intensity), choice
        else:
            return stg.constant_t_profile(level=intensity), 'constant'

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

        Uses log-uniform SNR and drift rate sampling, with random
        frequency and temporal profile selection.

        Returns (injected data, signal parameters dict).
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        if start_channel is None:
            start_channel = self.rng.integers(1, fchans - 1)

        drift_rate, true_slope = self._sample_drift_rate()
        width = self._calculate_width(drift_rate)

        # Intercept for tracking
        b = tchans - true_slope * start_channel

        frame = self._make_frame(data)
        intensity = frame.get_intensity(snr=snr)

        # Select profiles
        f_profile, f_profile_name = self._select_f_profile(width)
        t_profile, t_profile_name = self._select_t_profile(intensity)

        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s
            ),
            t_profile,
            f_profile,
            stg.constant_bp_profile(level=1)
        )

        signal_info = {
            'snr': snr,
            'drift_rate': drift_rate,
            'start_channel': start_channel,
            'width': width,
            'slope': true_slope,
            'intercept': b,
            'f_profile': f_profile_name,
            't_profile': t_profile_name,
        }

        return frame.data, signal_info

    def inject_cadence_signal(self,
                              stacked_data: np.ndarray,
                              snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """Inject an ETI signal that drifts across a full stacked cadence."""
        return self.inject_signal(stacked_data, snr)

    # RFI signal injection (used for False samples)
    def _select_rfi_type(self) -> str:
        """Select RFI type based on configured weights."""
        types = list(self.params.rfi_types)
        weights = list(self.params.rfi_type_weights)
        return str(self.rng.choice(types, p=weights))

    def inject_rfi_signal(self,
                          data: np.ndarray,
                          snr: Optional[float] = None,
                          rfi_type: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a realistic RFI signal into spectrogram data.

        Args:
            data: Input spectrogram (tchans, fchans).
            snr: Signal-to-noise ratio. If None, sampled log-uniformly.
            rfi_type: One of RFI_TYPES. If None, picked by weighted random.

        Returns:
            (injected data, info dict with rfi_type and parameters).
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        if rfi_type is None:
            rfi_type = self._select_rfi_type()

        start_channel = self.rng.integers(1, fchans - 1)
        frame = self._make_frame(data)
        f_start = frame.get_frequency(index=start_channel)
        intensity = frame.get_intensity(snr=snr)

        # Build path, t_profile, f_profile based on RFI type
        if rfi_type == 'linear':
            # Same as ETI but will be injected in ALL obs (not just ON)
            drift_rate, _ = self._sample_drift_rate()
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

        elif rfi_type == 'scintillating':
            # Intensity oscillates sinusoidally over time
            drift_rate, _ = self._sample_drift_rate()
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
