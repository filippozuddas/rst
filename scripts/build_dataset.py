#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Dataset Builder (Orchestrator)

End-to-end pipeline for building RST training datasets:
1. Extract background snippets from HDF5 files (background_extractor)
2. Generate True/False samples with signal injection (cadence_generator)
3. Preprocess: stack 6 obs → (96, 1024), compute z-score stats
4. Save as .npz ready for training
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.data.preprocessing import stack_cadence, compute_dataset_stats


def build_dataset(
    backgrounds_path: str,
    output_dir: str,
    n_true: int = 5000,
    n_false: int = 5000,
    val_split: float = 0.15,
    seed: int = 42,
    fchans: int = 1024,
    snr_base: float = 20,
    snr_range: float = 40,
):
    """
    Build train/val .npz datasets from extracted backgrounds.

    Args:
        backgrounds_path: Path to .npz with extracted backgrounds (from background_extractor).
        output_dir: Directory to save train.npz and val.npz.
        n_true: Number of True (ETI) samples to generate.
        n_false: Number of False (RFI/noise) samples to generate.
        val_split: Fraction of data for validation.
        seed: Random seed.
        fchans: Frequency channels per snippet.
        snr_base: Base SNR for signal injection.
        snr_range: SNR range for signal injection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- 1. Load backgrounds ----
    print(f"\n{'='*60}")
    print("RST DATASET BUILDER")
    print(f"{'='*60}")

    data = np.load(backgrounds_path, allow_pickle=True)
    plate = data['backgrounds']
    print(f"  Loaded {len(plate)} backgrounds from {backgrounds_path}")
    print(f"  Shape: {plate.shape}")

    # ---- 2. Initialize cadence generator ----
    params = CadenceParams(
        fchans=fchans,
        snr_base=snr_base,
        snr_range=snr_range,
    )
    gen = CadenceGenerator(params=params, plate=plate, seed=seed)

    # ---- 3. Generate samples ----
    total = n_true + n_false
    print(f"\n  Generating {n_true} True + {n_false} False = {total} samples...")

    spectrograms = []
    labels = []

    # True samples (label = 1)
    print(f"\n  → True samples (ETI):")
    for _ in tqdm(range(n_true), desc="    True"):
        cadence = gen.create_true_sample_fast()
        stacked = stack_cadence(cadence)  # (96, 1024)
        spectrograms.append(stacked)
        labels.append(1)

    # False samples (label = 0)
    print(f"\n  → False samples (RFI/noise):")
    for _ in tqdm(range(n_false), desc="    False"):
        cadence = gen.create_false_sample()
        stacked = stack_cadence(cadence)  # (96, 1024)
        spectrograms.append(stacked)
        labels.append(0)

    spectrograms = np.array(spectrograms, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"\n  Dataset shape: {spectrograms.shape}")
    print(f"  Labels: {int(labels.sum())} True, {int(len(labels) - labels.sum())} False")

    # ---- 4. Shuffle and split ----
    indices = rng.permutation(total)
    spectrograms = spectrograms[indices]
    labels = labels[indices]

    n_val = int(total * val_split)
    n_train = total - n_val

    train_specs = spectrograms[:n_train]
    train_labels = labels[:n_train]
    val_specs = spectrograms[n_train:]
    val_labels = labels[n_train:]

    # ---- 5. Compute normalization stats on train set ----
    mean, std = compute_dataset_stats(train_specs)
    print(f"\n  Train stats: mean={mean:.4f}, std={std:.4f}")

    # ---- 6. Save ----
    train_path = output_dir / "train.npz"
    val_path = output_dir / "val.npz"

    np.savez_compressed(
        train_path,
        spectrograms=train_specs,
        labels=train_labels,
        mean=mean,
        std=std,
    )
    np.savez_compressed(
        val_path,
        spectrograms=val_specs,
        labels=val_labels,
        mean=mean,  # Use train stats for val
        std=std,
    )

    print(f"\n{'='*60}")
    print(f"✅ DATASET SAVED:")
    print(f"   Train: {train_path} ({n_train} samples)")
    print(f"   Val:   {val_path} ({n_val} samples)")
    print(f"   Shape: {train_specs.shape[1:]}")
    print(f"   Stats: mean={mean:.4f}, std={std:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="RST — Build training dataset from extracted backgrounds"
    )
    parser.add_argument('--backgrounds', '-b', required=True,
                        help='Path to backgrounds .npz (from background_extractor)')
    parser.add_argument('--output', '-o', default='data/processed',
                        help='Output directory for train/val .npz files')
    parser.add_argument('--n-true', type=int, default=5000,
                        help='Number of True (ETI) samples')
    parser.add_argument('--n-false', type=int, default=5000,
                        help='Number of False (RFI/noise) samples')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fchans', type=int, default=1024,
                        help='Frequency channels per snippet')
    parser.add_argument('--snr-base', type=float, default=20,
                        help='Base SNR for signal injection')
    parser.add_argument('--snr-range', type=float, default=40,
                        help='SNR range for signal injection')

    args = parser.parse_args()

    build_dataset(
        backgrounds_path=args.backgrounds,
        output_dir=args.output,
        n_true=args.n_true,
        n_false=args.n_false,
        val_split=args.val_split,
        seed=args.seed,
        fchans=args.fchans,
        snr_base=args.snr_base,
        snr_range=args.snr_range,
    )


if __name__ == '__main__':
    main()
