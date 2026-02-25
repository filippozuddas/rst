# -*- coding: utf-8 -*-
"""
RST — Cadence Generator

Creates True, False, and SingleShot training samples by combining
noise backgrounds with signal injections following the ON-OFF pattern.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from .noise_generator import NoiseGenerator, NoiseParams
from .signal_generator import SignalGenerator, SignalParams, check_intersection


@dataclass
class CadenceParams:
    """Parameters for cadence generation."""
    tchans: int = 16              # Time channels per observation
    fchans: int = 1024            # Frequency channels (RST snippet width)
    num_observations: int = 6     # Cadence length (ON-OFF-ON-OFF-ON-OFF)

    # Signal parameters
    snr_base: float = 20
    snr_range: float = 40

    # Noise parameters
    df: float = 2.7939677238464355
    dt: float = 18.25361108
    noise_mean: float = 58348559


class CadenceGenerator:
    """
    Generator for SETI training cadences.

    Creates three types of samples:
    - True: ETI signal present in ON observations only
    - False: RFI pattern (signal in all obs) or pure noise
    - SingleShot: Single signal injection for testing
    """

    def __init__(self,
                 params: Optional[CadenceParams] = None,
                 plate: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Args:
            params: Cadence generation parameters
            plate: Real observation backgrounds, shape (N, 6, tchans, fchans).
                   If None, uses synthetic noise.
            seed: Random seed
        """
        self.params = params or CadenceParams()
        self.plate = plate
        self.rng = np.random.default_rng(seed)

        # Initialize sub-generators
        noise_params = NoiseParams(
            fchans=self.params.fchans,
            tchans=self.params.tchans,
            df=self.params.df,
            dt=self.params.dt,
            noise_mean=self.params.noise_mean
        )
        self.noise_gen = NoiseGenerator(noise_params)

        signal_params = SignalParams(
            df=self.params.df,
            dt=self.params.dt,
            snr_base=self.params.snr_base,
            snr_range=self.params.snr_range
        )
        self.signal_gen = SignalGenerator(signal_params, seed)

    def _get_background(self) -> np.ndarray:
        """Get background from plate or generate synthetic noise. Returns (6, tchans, fchans)."""
        if self.plate is not None:
            idx = self.rng.integers(0, self.plate.shape[0])
            return self.plate[idx].copy()
        else:
            return self.noise_gen.generate_cadence(
                self.params.num_observations,
                self.params.fchans,
                self.params.tchans
            )

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

    def create_true_sample(self,
                           snr: Optional[float] = None,
                           factor: float = 1.0,
                           ensure_non_crossing: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Create a TRUE sample (ETI signal in ON observations only).

        Two signals are injected: one across all obs, one extra in ON only.
        ON obs get both signals (distinguishable), OFF obs get only one.

        Returns (cadence shape (6, tchans, fchans), metadata dict).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        if snr is None:
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)

        # Inject two signals, optionally ensuring they don't cross
        if ensure_non_crossing:
            for _ in range(100):
                injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
                injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
                if check_intersection(info1['slope'], info2['slope'],
                                      info1['intercept'], info2['intercept']):
                    break
            else:
                injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
                injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
        else:
            injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
            injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)

        # ON obs (indices 0, 2, 4) get both signals; OFF obs (1, 3, 5) get only one
        result = np.zeros((6, self.params.tchans, self.params.fchans))
        result[0] = injected2[0:16, :]
        result[2] = injected2[32:48, :]
        result[4] = injected2[64:80, :]
        result[1] = injected1[16:32, :]
        result[3] = injected1[48:64, :]
        result[5] = injected1[80:96, :]

        metadata = {
            'sample_type': 'true',
            'snr': snr,
            'signal1': info1,
            'signal2': info2
        }

        return result, metadata

    def create_true_sample_fast(self,
                                snr: Optional[float] = None,
                                factor: float = 1.0) -> np.ndarray:
        """Fast version without intersection checking. Returns (6, tchans, fchans)."""
        background = self._get_background()
        stacked = self._stack_cadence(background)

        if snr is None:
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)

        injected1, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
        injected2, _ = self.signal_gen.inject_cadence_signal(injected1, snr * factor)

        result = np.zeros((6, self.params.tchans, self.params.fchans))
        result[0] = injected2[0:16, :]
        result[2] = injected2[32:48, :]
        result[4] = injected2[64:80, :]
        result[1] = injected1[16:32, :]
        result[3] = injected1[48:64, :]
        result[5] = injected1[80:96, :]

        return result

    def create_false_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Create a FALSE sample (RFI or pure noise).
        If snr is provided, always creates RFI pattern. Otherwise 50/50 RFI vs noise.
        Returns (6, tchans, fchans).
        """
        if snr is not None:
            background = self._get_background()
            stacked = self._stack_cadence(background)
            injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
            return self._unstack_cadence(injected)

        choice = self.rng.random()

        if choice > 0.5:
            # RFI: same signal in all observations
            background = self._get_background()
            stacked = self._stack_cadence(background)
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)
            injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
            return self._unstack_cadence(injected)
        else:
            # Pure noise/background
            return self._get_background()

    def create_single_shot_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Create a SINGLE SHOT sample (signal in ON obs only, no second signal).
        Used for sensitivity testing. Returns (6, tchans, fchans).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        if snr is None:
            snr = self.rng.uniform(self.params.snr_base,
                                   self.params.snr_base + self.params.snr_range)

        injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)

        result = np.zeros((6, self.params.tchans, self.params.fchans))
        result[0] = injected[0:16, :]
        result[1] = stacked[16:32, :]   # Original background
        result[2] = injected[32:48, :]
        result[3] = stacked[48:64, :]
        result[4] = injected[64:80, :]
        result[5] = stacked[80:96, :]

        return result

    def generate_batch(self,
                       sample_type: str,
                       batch_size: int,
                       snr_base: Optional[float] = None,
                       snr_range: Optional[float] = None,
                       factor: float = 1.0) -> np.ndarray:
        """
        Generate a batch of samples.
        sample_type: 'true', 'true_fast', 'false', 'single_shot'.
        Returns (batch_size, 6, tchans, fchans).
        """
        if snr_base is not None:
            self.params.snr_base = snr_base
        if snr_range is not None:
            self.params.snr_range = snr_range

        batch = np.zeros((batch_size, self.params.num_observations,
                         self.params.tchans, self.params.fchans))

        for i in range(batch_size):
            if sample_type == 'true':
                batch[i], _ = self.create_true_sample(factor=factor)
            elif sample_type == 'true_fast':
                batch[i] = self.create_true_sample_fast(factor=factor)
            elif sample_type == 'false':
                batch[i] = self.create_false_sample()
            elif sample_type == 'single_shot':
                batch[i] = self.create_single_shot_sample()
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

        return batch
