# -*- coding: utf-8 -*-
"""
RST — Cadence Generator

Creates True, False, and SingleShot training samples by combining
real observation backgrounds with signal injections following the
ON-OFF pattern.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from .signal_generator import SignalGenerator, SignalParams, check_intersection


@dataclass
class CadenceParams:
    """Parameters for cadence generation."""
    tchans: int = 16              # Time channels per observation
    fchans: int = 1024            # Frequency channels (RST snippet width)
    num_observations: int = 6     # Cadence length (ON-OFF-ON-OFF-ON-OFF)

    # Instrument parameters
    df: float = 2.7939677238464355
    dt: float = 18.25361108

    # SNR parameters — used for log-uniform sampling via SignalGenerator
    snr_min: float = 5.0
    snr_max: float = 50.0

    # True sample composition
    eti_only_fraction: float = 0.4   # 40% solo ETI, 60% ETI + RFI disturbance
    max_disturbance_rfi: int = 3     # Max RFI signals added to True samples

    # False sample composition
    rfi_fraction: float = 0.6        # 60% RFI injected, 40% pure background
    rfi_count_weights: tuple = (0.4, 0.3, 0.2, 0.1)  # P(1), P(2), P(3), P(4)


class CadenceGenerator:
    """
    Generator for SETI training cadences.

    Creates three types of samples:
    - True: ETI signal present in ON observations only
      - ETI-only (40%): single ETI signal in ON obs
      - ETI+RFI (60%): ETI in ON + 1-3 RFI across all obs
    - False: 60% RFI (1-4 signals across all obs), 40% pure real background
    - SingleShot: Single signal injection for sensitivity testing

    Requires a plate of real backgrounds
    """

    def __init__(self,
                 params: Optional[CadenceParams] = None,
                 plate: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Args:
            params: Cadence generation parameters
            plate: Real observation backgrounds, shape (N, 6, tchans, fchans).
            seed: Random seed (None for random)
        """
        self.params = params or CadenceParams()
        self.plate = plate
        self.rng = np.random.default_rng(seed)

        if self.plate is None:
            raise ValueError(
                "CadenceGenerator requires a plate of real backgrounds. "
                "Use background_extractor to extract from HDF5 files."
            )

        # Initialize signal generator with matching params
        signal_params = SignalParams(
            df=self.params.df,
            dt=self.params.dt,
            snr_min=self.params.snr_min,
            snr_max=self.params.snr_max,
        )
        self.signal_gen = SignalGenerator(signal_params, seed)

    def _get_background(self) -> np.ndarray:
        """Get a random background from the real plate. Returns (6, tchans, fchans)."""
        idx = self.rng.integers(0, self.plate.shape[0])
        return self.plate[idx].copy()

    def _stack_cadence(self, cadence: np.ndarray) -> np.ndarray:
        """Stack (6, tchans, fchans) → (6*tchans, fchans)."""
        stacked = np.zeros((self.params.num_observations * self.params.tchans,
                           self.params.fchans))
        for i in range(self.params.num_observations):
            start = i * self.params.tchans
            end = (i + 1) * self.params.tchans
            stacked[start:end, :] = cadence[i]
        return stacked

    def _unstack_cadence(self, stacked: np.ndarray) -> np.ndarray:
        """Unstack (6*tchans, fchans) → (6, tchans, fchans)."""
        cadence = np.zeros((self.params.num_observations,
                           self.params.tchans,
                           self.params.fchans))
        for i in range(self.params.num_observations):
            start = i * self.params.tchans
            end = (i + 1) * self.params.tchans
            cadence[i] = stacked[start:end, :]
        return cadence

    def _extract_on_off(self, stacked_on: np.ndarray,
                        stacked_off: np.ndarray) -> np.ndarray:
        """Combine ON and OFF stacked arrays into a cadence.

        ON observations (indices 0, 2, 4) come from stacked_on.
        OFF observations (indices 1, 3, 5) come from stacked_off.
        """
        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        result[0] = stacked_on[0:t, :]
        result[2] = stacked_on[2*t:3*t, :]
        result[4] = stacked_on[4*t:5*t, :]
        result[1] = stacked_off[t:2*t, :]
        result[3] = stacked_off[3*t:4*t, :]
        result[5] = stacked_off[5*t:6*t, :]
        return result

    def _sample_rfi_count(self) -> int:
        """Sample number of RFI signals from weighted distribution."""
        weights = np.array(self.params.rfi_count_weights, dtype=float)
        weights /= weights.sum()  # normalize
        counts = np.arange(1, len(weights) + 1)
        return int(self.rng.choice(counts, p=weights))

    # True samples
    def create_true_sample(self,
                           snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        Create a TRUE sample (ETI signal in ON observations only).

        Selects between two modes:
        - ETI-only (40%): single ETI signal in ON obs, background in OFF
        - ETI+RFI (60%): ETI in ON + 1-3 RFI across all obs

        Returns (cadence shape (6, tchans, fchans), metadata dict).
        """
        if self.rng.random() < self.params.eti_only_fraction:
            return self._create_eti_only_sample(snr)
        else:
            return self._create_eti_with_rfi_sample(snr)

    def _create_eti_only_sample(self,
                                snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        Single ETI signal present only in ON observations.
        OFF observations contain only the original background.
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        # Inject ETI signal across the full stacked cadence
        injected, eti_info = self.signal_gen.inject_cadence_signal(stacked, snr)

        # ON obs get the injected signal; OFF obs keep original background
        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        result[0] = injected[0:t, :]
        result[1] = stacked[t:2*t, :]       # Original background
        result[2] = injected[2*t:3*t, :]
        result[3] = stacked[3*t:4*t, :]     # Original background
        result[4] = injected[4*t:5*t, :]
        result[5] = stacked[5*t:6*t, :]     # Original background

        metadata = {
            'sample_type': 'true',
            'true_mode': 'eti_only',
            'eti_signal': eti_info,
        }

        return result, metadata

    def _create_eti_with_rfi_sample(self,
                                    snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        ETI signal in ON observations + 1-3 RFI signals across ALL observations.
        Simulates a realistic scenario where the ETI signal coexists with RFI.
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        # Step 1: inject 1-3 RFI signals across all observations
        n_rfi = self.rng.integers(1, self.params.max_disturbance_rfi + 1)
        rfi_infos = []
        current = stacked.copy()
        for _ in range(n_rfi):
            current, rfi_info = self.signal_gen.inject_rfi_signal(current)
            rfi_infos.append(rfi_info)

        # Step 2: inject ETI signal across the full cadence
        injected, eti_info = self.signal_gen.inject_cadence_signal(current, snr)

        # ON obs get ETI + RFI; OFF obs get only RFI (from 'current')
        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        result[0] = injected[0:t, :]
        result[1] = current[t:2*t, :]       # RFI only
        result[2] = injected[2*t:3*t, :]
        result[3] = current[3*t:4*t, :]     # RFI only
        result[4] = injected[4*t:5*t, :]
        result[5] = current[5*t:6*t, :]     # RFI only

        metadata = {
            'sample_type': 'true',
            'true_mode': 'eti_with_rfi',
            'eti_signal': eti_info,
            'n_rfi': n_rfi,
            'rfi_signals': rfi_infos,
        }

        return result, metadata

    def create_true_sample_fast(self,
                                snr: Optional[float] = None) -> np.ndarray:
        """Fast version — returns only the cadence array (no metadata)."""
        result, _ = self.create_true_sample(snr)
        return result

    # False samples
    def create_false_sample(self,
                            snr: Optional[float] = None) -> np.ndarray:
        """
        Create a FALSE sample: 60% RFI (1-4 signals), 40% pure background.

        Pure background samples contain only the natural noise and spectral
        features from the real observation plate.
        RFI count is sampled from a weighted distribution (default: P(1)=40%, 
        P(2)=30%, P(3)=20%, P(4)=10%).

        Returns (6, tchans, fchans).
        """
        background = self._get_background()

        if snr is None and self.rng.random() > self.params.rfi_fraction:
            return background
            
        stacked = self._stack_cadence(background)

        n_rfi = self._sample_rfi_count()

        for _ in range(n_rfi):
            stacked, _ = self.signal_gen.inject_rfi_signal(stacked, snr)

        return self._unstack_cadence(stacked)

    # Sensitivity testing
    def create_single_shot_sample(self,
                                  snr: Optional[float] = None) -> np.ndarray:
        """
        Create a SINGLE SHOT sample (signal in ON obs only, no second signal).
        Used for sensitivity testing. Returns (6, tchans, fchans).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)

        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        result[0] = injected[0:t, :]
        result[1] = stacked[t:2*t, :]       # Original background
        result[2] = injected[2*t:3*t, :]
        result[3] = stacked[3*t:4*t, :]
        result[4] = injected[4*t:5*t, :]
        result[5] = stacked[5*t:6*t, :]

        return result

    # Batch generation
    def generate_batch(self,
                       sample_type: str,
                       batch_size: int,
                       factor: float = 1.0) -> np.ndarray:
        """
        Generate a batch of samples.
        sample_type: 'true', 'true_fast', 'false', 'single_shot'.
        Returns (batch_size, 6, tchans, fchans).
        """
        batch = np.zeros((batch_size, self.params.num_observations,
                         self.params.tchans, self.params.fchans))

        for i in range(batch_size):
            if sample_type == 'true':
                batch[i], _ = self.create_true_sample()
            elif sample_type == 'true_fast':
                batch[i] = self.create_true_sample_fast()
            elif sample_type == 'false':
                batch[i] = self.create_false_sample()
            elif sample_type == 'single_shot':
                batch[i] = self.create_single_shot_sample()
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

        return batch
