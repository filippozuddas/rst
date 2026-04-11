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
from dataclasses import dataclass, field


# Available RFI types for False sample generation
RFI_TYPES: List[str] = [
    'linear',        # Standard linear drift (same as ETI but in all obs)
    'stationary',    # Fixed frequency with jitter
    'random_walk',   # Frequency wanders randomly over time
    'scintillating'  # Intensity oscillates sinusoidally over time
]

# Default weights for weighted RFI type selection
RFI_TYPE_WEIGHTS: List[float] = [0.40, 0.15, 0.25, 0.20]

def compute_max_drift_rate(
    snippet_width: int, df: float, dt: float, n_scans: int = 4, bins_per_scan: int = 16
) -> float:
    """
    Calculate the maximum theoretical drift rate a signal can have before it
    drifts completely out of the snippet window before n_scans have elapsed.

    Args:
        snippet_width: Width of the spectral window (in frequency bins).
        df: Frequency resolution (Hz per bin).
        dt: Time resolution (seconds per bin).
        n_scans: Number of observations the signal must cross without exiting (e.g. 4 for ON-OFF-ON-OFF).
        bins_per_scan: Number of time bins per observation.

    Returns:
        float: Maximum drift rate in Hz/s.
    """
    total_bandwidth_hz = snippet_width * df
    total_time_s = n_scans * bins_per_scan * dt
    return float(total_bandwidth_hz / total_time_s)


@dataclass
class SignalParams:
    """Parameters for signal injection."""
    df: float = 2.7939677238464355  # Hz per channel
    dt: float = 18.25361108         # Seconds per time bin
    fch1: float = 0                 # MHz (0 for injection on existing data)

    # SNR parameters — log-uniform sampling in [snr_min, snr_max]
    snr_min: float = 10.0
    snr_max: float = 50.0

    # ETI width parameters — narrowband (|DR|×dt + U(eti_offset_min, eti_offset_max))
    eti_width_offset_min: float = 1.0    # Hz
    eti_width_offset_max: float = 10.0   # Hz

    # RFI width parameters — broader (|DR|×dt + U(rfi_offset_min, rfi_offset_max))
    rfi_width_offset_min: float = 5.0    # Hz
    rfi_width_offset_max: float = 55.0   # Hz

    # Drift rate parameters — log-uniform with random sign
    max_drift_rate: float = field(
        default_factory=lambda: compute_max_drift_rate(
            snippet_width=1024, df=2.7939677238464355, dt=18.25361108, n_scans=4
        )
    )
    min_nonzero_drift: float = 0.01  # Min non-zero drift rate for log sampling
    zero_drift_prob: float = 0.05    # Probability of exactly zero drift

    # Frequency profile selection
    freq_profiles: tuple = ('gaussian', 'sinc2')
    freq_profile_weights: tuple = (0.8, 0.2)

    # Temporal profile selection
    time_profiles: tuple = ('constant', 'scintillating')
    time_profile_weights: tuple = (0.6, 0.4)

    # RFI type weights
    rfi_types: tuple = ('linear', 'stationary', 'random_walk', 'scintillating')
    rfi_type_weights: tuple = (0.40, 0.15, 0.25, 0.20)

    # Legacy ML-SRT-SETI behavior
    use_legacy_drift: bool = False   # If True, overrides log-uniform sampling with geometric corner-targeting

class SignalGenerator:
    """
    Generator for synthetic SETI signals.

    Methods:
        inject_signal: ETI-like narrowband drifting signal (for True samples)
        inject_rfi_signal: Diverse RFI patterns (for False samples)

    Sampling strategies:
        - SNR: log-uniform in [snr_min, snr_max] (more low-SNR samples)
        - Drift rate: log-uniform in [min_nonzero, max] with random sign
          (concentrates on low drift rates as seen in real candidates)
        - Freq profile: weighted random (gaussian 80%, sinc² 20%)
        - Time profile: weighted random (constant 60%, scintillating 40%)
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
        # --- ML-SRT-SETI Legacy Corner-Targeting Logic ---
        if self.params.use_legacy_drift:
            # We need to know where we start and total width to target opposite edges.
            # Notice this breaks the signature a bit if start_channel/fchans are not passed here,
            # so we handle it below in inject_signal where we have that context.
            pass

        # --- RST Log-Uniform Strategy ---
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

    def _calculate_legacy_drift_rate(self, start_channel: int, fchans: int, tchans: int) -> Tuple[float, float]:
        """ML-SRT-SETI logic: Calculate drift rate to traverse the entire observation from corner to opposite edge."""
        direction = self.rng.choice([-1, 1])

        if direction > 0:
            # Positive drift: signal drifts from lower to higher frequencies
            true_slope = tchans / start_channel if start_channel > 0 else 1e9
        else:
            # Negative drift: signal drifts from higher to lower frequencies
            # fchans is equivalent to width in the old code
            denominator = start_channel - fchans
            true_slope = tchans / denominator if denominator != 0 else 1e9

        # Add small random perturbation for variety
        slope = true_slope * (self.params.dt / self.params.df) + self.rng.uniform(0, 3) * direction

        if abs(slope) < 1e-9:
            drift_rate = self.params.max_drift_rate * direction # Cap it physically
        else:
            drift_rate = -1.0 / slope

        return drift_rate, true_slope

    def _calculate_eti_width(self, drift_rate: float) -> float:
        """ETI width formula: |DR|×dt + U(eti_offset_min, eti_offset_max).

        Narrowband: intrinsic width 1-10 Hz, plus smearing compensation.
        """
        drift_component = abs(drift_rate) * self.params.dt
        offset = self.rng.uniform(self.params.eti_width_offset_min,
                                  self.params.eti_width_offset_max)
        return drift_component + offset

    def _calculate_rfi_width(self, drift_rate: float) -> float:
        """RFI width formula: |DR|×dt + U(rfi_offset_min, rfi_offset_max).

        Broader: terrestrial RFI spans wider frequency ranges.
        """
        drift_component = abs(drift_rate) * self.params.dt
        offset = self.rng.uniform(self.params.rfi_width_offset_min,
                                  self.params.rfi_width_offset_max)
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
    # Minimum time bin the signal must still be in-bounds at (end of ON₂).
    # This guarantees the signal is visible in at least 2 of 3 ON windows.
    _MIN_VISIBLE_BIN = 47

    def inject_signal(self,
                      data: np.ndarray,
                      snr: Optional[float] = None,
                      start_channel: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a narrowband drifting ETI signal.

        Uses log-uniform SNR and drift rate sampling, with random
        frequency and temporal profile selection.

        The start_channel is automatically constrained so that the signal
        remains within the snippet through at least ON₂ (time bin 47),
        ensuring visibility in at least 2 of 3 ON windows.

        Returns (injected data, signal parameters dict).
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        # When start_channel is given explicitly, use legacy flow unchanged
        if start_channel is not None:
            if self.params.use_legacy_drift:
                drift_rate, true_slope = self._calculate_legacy_drift_rate(start_channel, fchans, tchans)
            else:
                drift_rate, true_slope = self._sample_drift_rate()
        else:
            # Sample drift rate FIRST, then constrain start_channel
            if self.params.use_legacy_drift:
                # Legacy mode needs start_channel first — pick freely, then compute drift
                start_channel = self.rng.integers(1, fchans - 1)
                drift_rate, true_slope = self._calculate_legacy_drift_rate(start_channel, fchans, tchans)
            else:
                drift_rate, true_slope = self._sample_drift_rate()

                # Compute how many channels the signal drifts through ON₂
                drift_channels = abs(drift_rate) / self.params.df * self._MIN_VISIBLE_BIN * self.params.dt

                if drift_rate > 0:
                    # Drifts right → start must leave room on the right
                    max_start = max(1, int(fchans - 1 - drift_channels))
                    start_channel = int(self.rng.integers(1, max_start + 1))
                elif drift_rate < 0:
                    # Drifts left → start must leave room on the left
                    min_start = min(fchans - 2, int(drift_channels + 1))
                    start_channel = int(self.rng.integers(min_start, fchans - 1))
                else:
                    start_channel = int(self.rng.integers(1, fchans - 1))

        width = self._calculate_eti_width(drift_rate)

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
            if self.params.use_legacy_drift:
                drift_rate, _ = self._calculate_legacy_drift_rate(start_channel, fchans, tchans)
            else:
                drift_rate, _ = self._sample_drift_rate()
            width = self._calculate_rfi_width(drift_rate)
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
            width = self._calculate_rfi_width(drift_rate)
            path = stg.simple_rfi_path(f_start=f_start,
                                       drift_rate=drift_rate * u.Hz / u.s,
                                       spread=spread,
                                       spread_type='normal',
                                       rfi_type='random_walk')
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'scintillating':
            # Intensity oscillates sinusoidally over time
            if self.params.use_legacy_drift:
                drift_rate, _ = self._calculate_legacy_drift_rate(start_channel, fchans, tchans)
            else:
                drift_rate, _ = self._sample_drift_rate()
            width = self._calculate_rfi_width(drift_rate)
            period = self.rng.uniform(50, 300) * u.s
            amplitude = intensity * self.rng.uniform(0.3, 0.8)
            path = stg.constant_path(f_start=f_start,
                                     drift_rate=drift_rate * u.Hz / u.s)
            t_prof = stg.sine_t_profile(period=period,
                                        amplitude=amplitude,
                                        level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        else:
            raise ValueError(f"Unknown RFI type: {rfi_type}. Choose from {RFI_TYPES}")

        frame.add_signal(path, t_prof, f_prof, stg.constant_bp_profile(level=1))

        signal_info = {
            'rfi_type': rfi_type,
            'snr': snr,
            'start_channel': start_channel,
        }

        return frame.data, signal_info