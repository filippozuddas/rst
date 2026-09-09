# -*- coding: utf-8 -*-
"""
rst-seti — Model Hub

Handles automatic download and caching of RST model weights from
Hugging Face Hub. Supports multiple model versions and architecture
compatibility checking.

Cache directory: ~/.cache/rst-seti/models/

Usage:
    from rst_seti.hub import ModelHub

    # Download (or use cached) weights + config
    checkpoint_path, config_path = ModelHub.resolve(model_name="rst-base384-v1")

    # List available models
    ModelHub.list_models()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face repository
# ---------------------------------------------------------------------------
HF_REPO_ID = "filippozuddas/rst-seti"

# ---------------------------------------------------------------------------
# Model registry — update with each new release
# ---------------------------------------------------------------------------
# Each entry MUST contain:
#   filename     : weights file on HF Hub (must match exact upload name)
#   config       : YAML config file on HF Hub
#   description  : human-readable description
#   architecture : dict of RSTModel constructor kwargs that the weights require.
#                  Used for compatibility checking before loading.
MODEL_REGISTRY: dict[str, dict] = {
    "rst-base384-v1": {
        "filename": "rst_base384_v1.pth",
        "config": "default_v1.yaml",
        "description": (
            "RST Base384 — DeiT-base/384 backbone, trained on SRT C-band "
            "cadences with synthetic ETI injection. v1.0 release."
        ),
        "architecture": {
            "model_size": "base384",
            "input_fdim": 1024,
            "input_tdim": 96,
            "label_dim": 1,
            "stride": 16,
        },
    },
}

DEFAULT_MODEL = "rst-base384-v1"

# Local cache root
_CACHE_ROOT = Path.home() / ".cache" / "rst-seti" / "models"


class ModelHub:
    """
    Static utility class for resolving model weights.

    Supports:
    - Automatic download from HF Hub on first use
    - Local caching (re-uses cached files on subsequent runs)
    - Architecture compatibility checking
    - Offline mode (uses cached files if HF Hub is unreachable)
    """

    @classmethod
    def resolve(
        cls,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Resolve a model name to local (checkpoint_path, config_path).

        Downloads from HF Hub if not cached, otherwise uses the local copy.

        Args:
            model_name: Key from MODEL_REGISTRY (default: rst-base384-v1).
            cache_dir:  Override cache directory. Defaults to ~/.cache/rst-seti/models/.

        Returns:
            Tuple of (checkpoint_path, config_path) as Path objects.

        Raises:
            ValueError: If model_name is not in the registry.
            RuntimeError: If download fails and no cached version is available.
        """
        if model_name not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {available}. "
                f"Use ModelHub.list_models() to see details."
            )

        entry = MODEL_REGISTRY[model_name]
        root = Path(cache_dir) if cache_dir else _CACHE_ROOT / model_name
        root.mkdir(parents=True, exist_ok=True)

        ckpt_path = root / entry["filename"]
        cfg_path  = root / entry["config"]

        # ── Try HF Hub download ────────────────────────────────────────────
        try:
            from huggingface_hub import hf_hub_download

            if not ckpt_path.exists():
                logger.info("Downloading weights for '%s'...", model_name)
                print(f"ℹ️  Downloading model '{model_name}' from Hugging Face Hub...")
                downloaded = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=entry["filename"],
                    local_dir=str(root),
                )
                ckpt_path = Path(downloaded)
                print(f"✅ Weights cached: {ckpt_path}")
            else:
                print(f"✅ Using cached weights: {ckpt_path}")

            if not cfg_path.exists():
                downloaded_cfg = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=entry["config"],
                    local_dir=str(root),
                )
                cfg_path = Path(downloaded_cfg)

        except ImportError:
            raise RuntimeError(
                "huggingface-hub is required for automatic model download. "
                "Install it with: pip install huggingface-hub"
            )
        except Exception as exc:
            # Network error — fall back to cache if it exists
            if ckpt_path.exists() and cfg_path.exists():
                logger.warning(
                    "Could not reach HF Hub (%s). Using cached files.", exc
                )
                print(f"⚠️  HF Hub unreachable. Using cached model: {ckpt_path}")
            else:
                raise RuntimeError(
                    f"Failed to download model '{model_name}' and no cached "
                    f"version found at {root}. "
                    f"Check your internet connection or provide weights manually "
                    f"with --model /path/to/weights.pth.\n"
                    f"Original error: {exc}"
                ) from exc

        # ── Compatibility check ────────────────────────────────────────────
        cls._check_compatibility(ckpt_path, entry)

        return ckpt_path, cfg_path

    @classmethod
    def list_models(cls) -> None:
        """Print all available model versions and their descriptions."""
        print("\nAvailable RST models:")
        print("-" * 60)
        for name, entry in MODEL_REGISTRY.items():
            tag = " ← default" if name == DEFAULT_MODEL else ""
            print(f"  {name}{tag}")
            print(f"    {entry['description']}")
            arch = entry["architecture"]
            print(
                f"    Architecture: {arch['model_size']} | "
                f"input ({arch['input_tdim']}×{arch['input_fdim']})"
            )
        print()

    @staticmethod
    def _check_compatibility(ckpt_path: Path, registry_entry: dict) -> None:
        """
        Verify that the cached checkpoint matches the expected architecture.

        Writes a small metadata JSON alongside the checkpoint so we can
        detect if a re-download is needed (e.g. after a hub update).
        """
        meta_path = ckpt_path.with_suffix(".meta.json")
        expected_arch = registry_entry["architecture"]

        if meta_path.exists():
            with open(meta_path) as f:
                cached_meta = json.load(f)
            if cached_meta.get("architecture") != expected_arch:
                logger.warning(
                    "Cached checkpoint architecture mismatch. "
                    "Delete %s and re-run to download the new version.",
                    ckpt_path,
                )
                print(
                    f"⚠️  Cached model architecture differs from registry. "
                    f"If you experience errors, delete {ckpt_path} to force re-download."
                )
        else:
            # Write metadata for future checks
            meta = {"architecture": expected_arch, "filename": registry_entry["filename"]}
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
