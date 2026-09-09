#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Dataset Builder (Orchestrator)

End-to-end pipeline for building RST training datasets:
1. Load real background plates from HDF5 (via background_extractor)
2. Generate True/False samples with realistic signal injection
3. Preprocess: stack 6 obs → (96, 1024), compute z-score stats
4. Save as .npz ready for training

Design choices (PDR v2):
- True samples: 40% ETI-only + 60% ETI with RFI disturbance
- False samples: 60% injected RFI (1-4 signals), 40% pure background
- SNR: log-uniform [5, 50]
- Drift rate: log-uniform ±4 Hz/s
- Target dataset size: 60k (30k True + 30k False)
"""

import argparse
import io
import os
import zipfile
import numpy as np
import numpy.lib.format as _npformat
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from rst_seti.data.cadence_generator import CadenceGenerator, CadenceParams, SignalParams
from rst_seti.data.preprocessing import stack_cadence


def _save_npz_chunked(path, spectrograms, labels, chunk_size=2000):
    """Write a .npz without loading the full spectrograms array into RAM at once.

    Standard np.savez_compressed buffers the entire array in a BytesIO before
    writing, which doubles peak RAM. This writer reads the array chunk_size rows
    at a time and streams the compressed bytes directly into the zip file.
    """
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        with zf.open('spectrograms.npy', 'w', force_zip64=True) as f:
            header = {
                'descr': _npformat.dtype_to_descr(spectrograms.dtype),
                'fortran_order': False,
                'shape': spectrograms.shape,
            }
            _npformat.write_array_header_2_0(f, header)
            n = len(spectrograms)
            for start in range(0, n, chunk_size):
                chunk = np.array(spectrograms[start:start + chunk_size])
                f.write(chunk.tobytes())
        buf = io.BytesIO()
        _npformat.write_array(buf, labels)
        zf.writestr('labels.npy', buf.getvalue())


def build_dataset(
    backgrounds_path: str,
    output_dir: str,
    n_true: int = 30000,
    n_false: int = 30000,
    val_split: float = 0.15,
    seed: int = None,
    fchans: int = 1024,
    snr_min: float = 5.0,
    snr_max: float = 50.0,
    eti_only_fraction: float = 0.4,
    rfi_fraction: float = 0.6,
    hard_false_fraction: float = 0.3,
    drift_distribution: str = 'lognormal',
    drift_median: float = 0.3,
    drift_log_sigma: float = 0.5,
):
    """
    Build train/val .npz datasets from extracted backgrounds.

    Args:
        backgrounds_path: Path to .npz with extracted backgrounds (from background_extractor).
        output_dir: Directory to save train.npz and val.npz.
        n_true: Number of True (ETI) samples to generate.
        n_false: Number of False (RFI) samples to generate.
        val_split: Fraction of data for validation.
        seed: Random seed. None for random generation with parameter logging.
        fchans: Frequency channels per snippet.
        snr_min: Minimum SNR for log-uniform sampling.
        snr_max: Maximum SNR for log-uniform sampling.
        eti_only_fraction: Fraction of True samples that are ETI-only (vs ETI+RFI).
        rfi_fraction: Fraction of False samples that contain injected RFI (vs pure background).
        hard_false_fraction: Fraction of False samples that are "hard-false" traps
            (strong signal in ALL 6 scans, no ON/OFF mask) — forces the model to
            use ON/OFF contrast instead of mere signal presence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # If no seed provided, generate one and log it for reproducibility
    if seed is None:
        seed = int(rng.integers(0, 2**31))
        rng = np.random.default_rng(seed)
        print(f"  Generated random seed: {seed}")

    # ---- 1. Load backgrounds ----
    print(f"\n{'='*60}")
    print("RST DATASET BUILDER (v2)")
    print(f"{'='*60}")

    data = np.load(backgrounds_path, allow_pickle=True)
    plate = data['backgrounds']
    print(f"  Loaded {len(plate)} backgrounds from {backgrounds_path}")
    print(f"  Shape: {plate.shape}")

    # Split the background plate into disjoint train/val pools BEFORE generation
    # so that no real-observation snippet feeds both splits (prevents the model
    # from memorizing background texture and inflating val metrics).
    n_bg = len(plate)
    bg_perm = rng.permutation(n_bg)
    n_bg_val = max(1, int(round(n_bg * val_split))) if val_split > 0 else 0
    plate_val   = plate[bg_perm[:n_bg_val]] if n_bg_val > 0 else None
    plate_train = plate[bg_perm[n_bg_val:]]
    if len(plate_train) == 0:
        raise ValueError(
            f"plate_train is empty (n_bg={n_bg}, val_split={val_split}). "
            "Provide more backgrounds or lower --val-split."
        )
    print(f"  Background split (snippet-level): "
          f"{len(plate_train)} train / {n_bg_val} val (disjoint)")

    # ---- 2. Initialize cadence generators (one per split, disjoint plates) ----
    params = CadenceParams(
        fchans=fchans,
        signal_params=SignalParams(
            snr_min=snr_min, snr_max=snr_max,
            drift_distribution=drift_distribution,
            drift_median=drift_median,
            drift_log_sigma=drift_log_sigma,
        ),
        eti_only_fraction=eti_only_fraction,
        rfi_fraction=rfi_fraction,
        hard_false_fraction=hard_false_fraction,
    )
    gen_train = CadenceGenerator(params=params, plate=plate_train, seed=seed)
    gen_val   = (CadenceGenerator(params=params, plate=plate_val, seed=seed + 1)
                 if plate_val is not None else None)

    # ---- 3. Generate samples ----
    total = n_true + n_false
    sp = gen_train.signal_gen.params
    max_drift = sp.max_drift_rate
    print(f"\n  Configuration:")
    print(f"    SNR: log-uniform [{snr_min}, {snr_max}]")
    if sp.drift_distribution == 'lognormal':
        print(f"    Drift rate: log-normal (median {sp.drift_median} Hz/s, "
              f"σ {sp.drift_log_sigma} dex), |DR| ≤ {max_drift:.2f} Hz/s")
    else:
        print(f"    Drift rate: log-uniform [{sp.min_nonzero_drift}, {max_drift:.2f}] Hz/s")
    print(f"    True samples: {int(eti_only_fraction*100)}% ETI-only, "
          f"{int((1-eti_only_fraction)*100)}% ETI+RFI")
    print(f"    False samples: {int(hard_false_fraction*100)}% hard-false (trap), "
          f"then {int(rfi_fraction*100)}% RFI / "
          f"{int((1-rfi_fraction)*100)}% pure background")
    print(f"    Seed: {seed}")
    print(f"\n  Generating {n_true} True + {n_false} False = {total} samples...")

    # Use a disk-backed memmap so the full array never lives in RAM.
    # For 200k samples (96×1024 float32) this avoids ~73 GB of RAM allocation.
    tmp_path = str(output_dir / "_spectrograms_tmp.dat")
    spectrograms = np.memmap(tmp_path, dtype=np.float32, mode='w+',
                             shape=(total, 96, 1024))
    labels = np.zeros(total, dtype=np.float32)

    try:
        # Generate shuffled indices to populate directly
        all_indices = np.arange(total)
        rng.shuffle(all_indices)
        true_indices = all_indices[:n_true]
        false_indices = all_indices[n_true:]

        # Train occupies positions [0, n_train); val occupies [n_train, total).
        # Route each sample to the generator whose plate matches its destination
        # split, keeping train/val backgrounds strictly disjoint.
        n_val = int(total * val_split)
        n_train = total - n_val

        # True samples (label = 1)
        print(f"\n  → True samples (ETI):")
        for idx in tqdm(true_indices, desc="    True"):
            g = gen_train if idx < n_train else gen_val
            cadence = g.create_true_sample_fast()
            stacked = stack_cadence(cadence)  # (96, 1024)
            spectrograms[idx] = stacked
            labels[idx] = 1

        # False samples (label = 0)
        print(f"\n  → False samples (RFI):")
        for idx in tqdm(false_indices, desc="    False"):
            g = gen_train if idx < n_train else gen_val
            cadence = g.create_false_sample()
            stacked = stack_cadence(cadence)  # (96, 1024)
            spectrograms[idx] = stacked
            labels[idx] = 0

        spectrograms.flush()

        train_specs = spectrograms[:n_train]
        train_labels = labels[:n_train]
        val_specs = spectrograms[n_train:]
        val_labels = labels[n_train:]

        # ---- 6. Save ----
        # _save_npz_chunked streams data in chunks to avoid the ~62 GB BytesIO
        # copy that np.savez_compressed creates internally.
        train_path = output_dir / "train.npz"
        val_path = output_dir / "val.npz"

        print(f"\n  → Saving train split ({n_train} samples)...")
        _save_npz_chunked(train_path, train_specs, train_labels)
        print(f"  → Saving val split ({n_val} samples)...")
        _save_npz_chunked(val_path, val_specs, val_labels)

        sample_shape = train_specs.shape[1:]

        # ---- 7. Save generation metadata ----
        import json
        meta = {
            'seed': seed,
            'n_true': n_true,
            'n_false': n_false,
            'n_train': n_train,
            'n_val': n_val,
            'snr_min': snr_min,
            'snr_max': snr_max,
            'snr_distribution': 'log_uniform',
            'snr_convention': 'per_on_scan_visible',
            'drift_rate_distribution': sp.drift_distribution,
            'drift_rate_max': max_drift,
            'drift_median': sp.drift_median,
            'drift_log_sigma': sp.drift_log_sigma,
            'min_nonzero_drift': sp.min_nonzero_drift,
            'zero_drift_prob': sp.zero_drift_prob,
            'eti_only_fraction': eti_only_fraction,
            'rfi_fraction': rfi_fraction,
            'hard_false_fraction': hard_false_fraction,
            'rfi_types': ['linear', 'stationary', 'random_walk',
                          'scintillating', 'broadband', 'pulsed'],
            'freq_profiles': ['gaussian', 'sinc2', 'lorentzian', 'voigt'],
            'time_profiles': ['constant', 'scintillating_stochastic'],
            'backgrounds_path': str(backgrounds_path),
            'n_backgrounds': len(plate),
            'background_split': 'snippet-level',
            'n_backgrounds_train': int(len(plate_train)),
            'n_backgrounds_val': int(n_bg_val),
        }
        meta_path = output_dir / "generation_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    finally:
        del spectrograms
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\n{'='*60}")
    print(f"✅ DATASET SAVED:")
    print(f"   Train: {train_path} ({n_train} samples)")
    print(f"   Val:   {val_path} ({n_val} samples)")
    print(f"   Meta:  {meta_path}")
    print(f"   Shape: {sample_shape}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="RST — Build training dataset from extracted backgrounds (v2)"
    )
    parser.add_argument('--backgrounds', '-b', required=True,
                        help='Path to backgrounds .npz (from background_extractor)')
    parser.add_argument('--output', '-o', default='data/processed',
                        help='Output directory for train/val .npz files')
    parser.add_argument('--n-true', type=int, default=30000,
                        help='Number of True (ETI) samples (default: 30000)')
    parser.add_argument('--n-false', type=int, default=30000,
                        help='Number of False (RFI) samples (default: 30000)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split fraction')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None = random with logging)')
    parser.add_argument('--fchans', type=int, default=1024,
                        help='Frequency channels per snippet')
    parser.add_argument('--snr-min', type=float, default=5.0,
                        help='Minimum SNR for log-uniform sampling, per-ON-scan visible (default: 5)')
    parser.add_argument('--snr-max', type=float, default=50.0,
                        help='Maximum SNR for log-uniform sampling (default: 50)')
    parser.add_argument('--eti-only-fraction', type=float, default=0.4,
                        help='Fraction of True samples that are ETI-only (default: 0.4)')
    parser.add_argument('--rfi-fraction', type=float, default=0.6,
                        help='Fraction of False samples that contain injected RFI (default: 0.6)')
    parser.add_argument('--hard-false-fraction', type=float, default=0.3,
                        help='Fraction of False samples that are hard-false traps: '
                             'strong signal in ALL 6 scans, forces ON/OFF discrimination (default: 0.3)')
    parser.add_argument('--drift-distribution', choices=['lognormal', 'loguniform'],
                        default='lognormal',
                        help="Drift magnitude distribution (default: lognormal)")
    parser.add_argument('--drift-median', type=float, default=0.3,
                        help='Log-normal drift median in Hz/s (default: 0.3)')
    parser.add_argument('--drift-log-sigma', type=float, default=0.5,
                        help='Log-normal drift spread in dex (default: 0.5)')

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
        hard_false_fraction=args.hard_false_fraction,
        drift_distribution=args.drift_distribution,
        drift_median=args.drift_median,
        drift_log_sigma=args.drift_log_sigma,
    )


if __name__ == '__main__':
    main()
