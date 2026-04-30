# -*- coding: utf-8 -*-
"""
rst-seti — Hardware Detection & Fallback

Automatically selects the optimal compute device for inference.

Priority chain (Linux / Windows):
  1. CUDA (NVIDIA GPU) — fastest, recommended
  2. CPU               — universal fallback

Apple Silicon (MPS) is intentionally excluded: RST targets Linux/Windows
servers used in SETI research environments.

Usage:
    from rst_seti.hardware import HardwareManager

    hw = HardwareManager.detect()
    print(hw)  # Hardware: NVIDIA RTX 4090 (48 GB VRAM) | dtype: bfloat16 | batch: 256
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRAM → recommended batch size table
# Tuned for RST base384 with (96, 1024) inputs and fp16/fp32 activations.
# ---------------------------------------------------------------------------
_VRAM_BATCH_TABLE = [
    (48_000, 256),   # 48 GB  — e.g. dual A6000 / RTX 4090
    (23_000, 128),   # 24 GB  — e.g. RTX 3090 / A5000
    ( 9_000,  64),   # 10 GB  — e.g. RTX 3080 / RTX 4080
    ( 5_000,  32),   # 6 GB   — e.g. RTX 3060 / GTX 1660
    (     0,  16),   # < 4 GB — minimal GPU
]

# CPU default is conservative: avoids thrashing with large spectrograms
_CPU_BATCH_SIZE = 16


@dataclass
class DeviceInfo:
    """Result of hardware auto-detection."""
    device: torch.device
    name: str
    dtype: torch.dtype
    recommended_batch_size: int
    vram_mb: int = 0          # 0 for CPU

    def __str__(self) -> str:
        dtype_str = str(self.dtype).replace("torch.", "")
        if self.vram_mb:
            return (
                f"Hardware: {self.name} ({self.vram_mb / 1024:.0f} GB VRAM) | "
                f"dtype: {dtype_str} | batch: {self.recommended_batch_size}"
            )
        return (
            f"Hardware: {self.name} | "
            f"dtype: {dtype_str} | batch: {self.recommended_batch_size}"
        )


class HardwareManager:
    """
    Static utility class for hardware detection and device selection.

    All methods are class-level (no instantiation needed).
    """

    @classmethod
    def detect(
        cls,
        preferred_device: Optional[str] = None,
        preferred_batch_size: Optional[int] = None,
    ) -> DeviceInfo:
        """
        Detect the best available compute device.

        Args:
            preferred_device:    Override string, e.g. 'cpu', 'cuda', 'cuda:1'.
                                 If None, auto-detects in priority order.
            preferred_batch_size: Override batch size.
                                  If None, infers from VRAM or uses CPU default.

        Returns:
            DeviceInfo with device, dtype, name, and recommended_batch_size.
        """
        if preferred_device is not None:
            return cls._build_from_string(preferred_device, preferred_batch_size)

        if torch.cuda.is_available():
            return cls._build_cuda(preferred_batch_size)

        return cls._build_cpu(preferred_batch_size)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @classmethod
    def _build_cuda(cls, batch_override: Optional[int]) -> DeviceInfo:
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        name = props.name
        vram_mb = props.total_memory // (1024 * 1024)

        # Choose dtype: bfloat16 on Ampere+ (compute capability >= 8.0),
        # float16 on older Volta/Turing/Turing cards.
        cc_major = props.major
        dtype = torch.bfloat16 if cc_major >= 8 else torch.float16

        batch = batch_override or cls._vram_to_batch(vram_mb)

        logger.debug("CUDA device: %s | VRAM: %d MB | dtype: %s | batch: %d",
                     name, vram_mb, dtype, batch)
        return DeviceInfo(
            device=device,
            name=name,
            dtype=dtype,
            recommended_batch_size=batch,
            vram_mb=vram_mb,
        )

    @classmethod
    def _build_cpu(cls, batch_override: Optional[int]) -> DeviceInfo:
        import platform
        cpu_name = platform.processor() or "CPU"
        dtype = torch.float32
        batch = batch_override or _CPU_BATCH_SIZE

        logger.debug("CPU fallback: %s | dtype: %s | batch: %d", cpu_name, dtype, batch)
        return DeviceInfo(
            device=torch.device("cpu"),
            name=cpu_name,
            dtype=dtype,
            recommended_batch_size=batch,
            vram_mb=0,
        )

    @classmethod
    def _build_from_string(
        cls,
        device_str: str,
        batch_override: Optional[int],
    ) -> DeviceInfo:
        """Build DeviceInfo from a user-supplied device string."""
        device = torch.device(device_str)
        if device.type == "cuda":
            # Validate the requested GPU index
            if not torch.cuda.is_available():
                logger.warning("CUDA not available; falling back to CPU.")
                return cls._build_cpu(batch_override)
            n_gpus = torch.cuda.device_count()
            idx = device.index or 0
            if idx >= n_gpus:
                logger.warning(
                    "GPU index %d not found (%d GPU(s) available); "
                    "using cuda:0.", idx, n_gpus
                )
                device = torch.device("cuda:0")
            with torch.cuda.device(device):
                return cls._build_cuda(batch_override)
        # CPU or unknown
        return cls._build_cpu(batch_override)

    @staticmethod
    def _vram_to_batch(vram_mb: int) -> int:
        """Map available VRAM (MB) to a safe default batch size."""
        for threshold_mb, batch in _VRAM_BATCH_TABLE:
            if vram_mb >= threshold_mb:
                return batch
        return _CPU_BATCH_SIZE
