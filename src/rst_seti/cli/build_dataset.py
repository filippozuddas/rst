#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rst-build-dataset — Build RST training dataset with synthetic signal injection.

This is the second step in the dataset creation pipeline:
  1. rst-build-backgrounds  — extract backgrounds from real observations
  2. rst-build-dataset      ← inject synthetic signals + create .npz dataset

All generation logic (CadenceGenerator, SignalParams, stack_cadence) is
imported from rst_seti and preserved verbatim.

Usage:
    rst-build-dataset -b data/training/backgrounds_6GHz.npz -o data/processed/v1/
    rst-build-dataset -b backgrounds.npz -o data/processed/ --n-true 30000 --n-false 30000
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def build_dataset(
    backgrounds_path: str,
    output_dir: str,
    n_true: int = 30000,
    n_false: int = 30000,
    val_split: float = 0.15,
    seed: int = None,
    fchans: int = 1024,
    snr_min: float = 10.0,
    snr_max: float = 50.0,
    eti_only_fraction: float = 0.4,
    rfi_fraction: float = 0.6,
):
    """
    Build train/val .npz datasets from extracted backgrounds.

    This function is a verbatim port of the logic in scripts/build_dataset.py,
    with imports updated to use rst_seti.* namespace.

    Args:
        backgrounds_path: Path to .npz with extracted backgrounds.
        output_dir: Directory to save train.npz and val.npz.
        n_true: Number of True (ETI) samples.
        n_false: Number of False (RFI) samples.
        val_split: Fraction of data for validation.
        seed: Random seed. None → random with logging.
        fchans: Frequency channels per snippet.
        snr_min: Minimum SNR (log-uniform sampling).
        snr_max: Maximum SNR (log-uniform sampling).
        eti_only_fraction: Fraction of True samples that are ETI-only.
        rfi_fraction: Fraction of False samples with injected RFI.
    """
    from rst_seti.data.cadence_generator import CadenceGenerator, CadenceParams, SignalParams
    from rst_seti.data.preprocessing import stack_cadence

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if seed is None:
        seed = int(rng.integers(0, 2**31))
        rng = np.random.default_rng(seed)
        print(f"  Generated random seed: {seed}")

    print(f"\n{'=' * 60}")
    print("RST DATASET BUILDER (v2)")
    print(f"{'=' * 60}")

    data  = np.load(backgrounds_path, allow_pickle=True)
    plate = data['backgrounds']
    print(f"  Loaded {len(plate)} backgrounds from {backgrounds_path}")
    print(f"  Shape: {plate.shape}")

    params = CadenceParams(
        fchans=fchans,
        signal_params=SignalParams(snr_min=snr_min, snr_max=snr_max),
        eti_only_fraction=eti_only_fraction,
        rfi_fraction=rfi_fraction,
    )
    gen = CadenceGenerator(params=params, plate=plate, seed=seed)

    total = n_true + n_false
    max_drift = gen.signal_gen.params.max_drift_rate
    print(f"\n  Configuration:")
    print(f"    SNR: log-uniform [{snr_min}, {snr_max}]")
    print(f"    Drift rate: log-uniform ±{max_drift:.2f} Hz/s")
    print(f"    True samples: {int(eti_only_fraction*100)}% ETI-only, "
          f"{int((1-eti_only_fraction)*100)}% ETI+RFI")
    print(f"    False samples: {int(rfi_fraction*100)}% RFI, "
          f"{int((1-rfi_fraction)*100)}% pure background")
    print(f"    Seed: {seed}")
    print(f"\n  Generating {n_true} True + {n_false} False = {total} samples...")

    spectrograms = np.zeros((total, 96, 1024), dtype=np.float32)
    labels       = np.zeros(total, dtype=np.float32)

    all_indices  = np.arange(total)
    rng.shuffle(all_indices)
    true_indices  = all_indices[:n_true]
    false_indices = all_indices[n_true:]

    print(f"\n  → True samples (ETI):")
    for idx in tqdm(true_indices, desc="    True"):
        cadence = gen.create_true_sample_fast()
        stacked = stack_cadence(cadence)   # (96, 1024)
        spectrograms[idx] = stacked
        labels[idx] = 1

    print(f"\n  → False samples (RFI):")
    for idx in tqdm(false_indices, desc="    False"):
        cadence = gen.create_false_sample()
        stacked = stack_cadence(cadence)   # (96, 1024)
        spectrograms[idx] = stacked
        labels[idx] = 0

    n_val   = int(total * val_split)
    n_train = total - n_val

    train_specs  = spectrograms[:n_train]
    train_labels = labels[:n_train]
    val_specs    = spectrograms[n_train:]
    val_labels   = labels[n_train:]

    train_path = output_dir / "train.npz"
    val_path   = output_dir / "val.npz"

    np.savez_compressed(train_path, spectrograms=train_specs, labels=train_labels)
    np.savez_compressed(val_path,   spectrograms=val_specs,   labels=val_labels)

    meta = {
        'seed': seed,
        'n_true': n_true,
        'n_false': n_false,
        'n_train': n_train,
        'n_val': n_val,
        'snr_min': snr_min,
        'snr_max': snr_max,
        'snr_distribution': 'log_uniform',
        'drift_rate_distribution': 'log_uniform',
        'drift_rate_max': max_drift,
        'eti_only_fraction': eti_only_fraction,
        'rfi_fraction': rfi_fraction,
        'rfi_types': ['linear', 'stationary', 'random_walk', 'scintillating'],
        'freq_profiles': ['gaussian', 'sinc2'],
        'time_profiles': ['constant', 'scintillating'],
        'backgrounds_path': str(backgrounds_path),
        'n_backgrounds': len(plate),
    }
    meta_path = output_dir / "generation_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✅ DATASET SAVED:")
    print(f"   Train: {train_path} ({n_train} samples)")
    print(f"   Val:   {val_path} ({n_val} samples)")
    print(f"   Meta:  {meta_path}")
    print(f"   Shape: {train_specs.shape[1:]}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='rst-build-dataset — Build RST training dataset with synthetic signal injection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rst-build-dataset -b data/training/backgrounds_6GHz.npz -o data/processed/v1/
  rst-build-dataset -b backgrounds.npz -o data/processed/ \\
      --n-true 30000 --n-false 30000 --snr-min 5 --snr-max 50 --seed 42
        """,
    )
    parser.add_argument('--backgrounds', '-b', required=True,
                        help='Path to backgrounds .npz (from rst-build-backgrounds)')
    parser.add_argument('--output', '-o', default='data/processed',
                        help='Output directory for train/val .npz (default: data/processed)')
    parser.add_argument('--n-true', type=int, default=30000,
                        help='Number of True (ETI) samples (default: 30000)')
    parser.add_argument('--n-false', type=int, default=30000,
                        help='Number of False (RFI) samples (default: 30000)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split fraction (default: 0.15)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random with logging)')
    parser.add_argument('--fchans', type=int, default=1024,
                        help='Frequency channels per snippet (default: 1024)')
    parser.add_argument('--snr-min', type=float, default=10.0,
                        help='Minimum SNR for log-uniform sampling (default: 10)')
    parser.add_argument('--snr-max', type=float, default=50.0,
                        help='Maximum SNR for log-uniform sampling (default: 50)')
    parser.add_argument('--eti-only-fraction', type=float, default=0.4,
                        help='Fraction of True samples that are ETI-only (default: 0.4)')
    parser.add_argument('--rfi-fraction', type=float, default=0.6,
                        help='Fraction of False samples with injected RFI (default: 0.6)')
    args = parser.parse_args()

    build_dataset(
        backgrounds_path=args.backgrounds,
        output_dir=args.output,
        n_true=args.n_true,
        n_false=args.n_false,
        val_split=args.val_split,
        seed=args.seed,
        fchans=args.fchans,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        eti_only_fraction=args.eti_only_fraction,
        rfi_fraction=args.rfi_fraction,
    )


if __name__ == '__main__':
    main()
