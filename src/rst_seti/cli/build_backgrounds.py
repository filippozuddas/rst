#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rst-build-backgrounds — Extract raw background snippets from real HDF5 observations.

Thin CLI wrapper around rst_seti.data.background_extractor.DatasetBuilder.
This is the first step in the dataset creation pipeline:
  1. rst-build-backgrounds  ← extract backgrounds from real observations
  2. rst-build-dataset      ← inject synthetic signals + create .npz dataset

Usage:
    rst-build-backgrounds -s /data/observations/ -o data/training/
    rst-build-backgrounds -s /data/obs/ -o data/training/ --band 6GHz -n 500
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=(
            'rst-build-backgrounds — Extract raw background snippets '
            'from real HDF5 telescope observations'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all bands:
  rst-build-backgrounds -s /data/observations/ -o data/training/

  # Extract only C-band (6 GHz), 500 snippets per cadence:
  rst-build-backgrounds -s /data/obs/ -o data/training/ --band 6GHz -n 500

  # Mixed multi-band (balanced by frequency bin):
  rst-build-backgrounds -s /data/obs/ -o data/training/ --band mixed

  # Just list cadences without extracting:
  rst-build-backgrounds -s /data/obs/ --list-only
        """,
    )
    parser.add_argument('--scan', '-s', nargs='+', required=True,
                        help='Directories to scan for HDF5 files')
    parser.add_argument('--output', '-o', default='data/training',
                        help='Output directory (default: data/training)')
    parser.add_argument('--snippet-width', type=int, default=1024,
                        help='Frequency channels per snippet (default: 1024)')
    parser.add_argument('--snippets-per-cadence', '-n', type=int, default=500,
                        help='Max snippets per cadence (default: 500)')
    parser.add_argument('--max-snippets', '-m', type=int, default=15000,
                        help='Max total snippets per band (default: 15000)')
    parser.add_argument('--name', default='backgrounds',
                        help='Output filename prefix (default: backgrounds)')
    parser.add_argument('--band', '-b',
                        choices=['6GHz', '18GHz', '1.4GHz', 'all', 'mixed'],
                        default='all',
                        help='Frequency band to process (default: all)')
    parser.add_argument('--mix-bins', type=float, default=1000.0,
                        help='Bin size in MHz for mixed-band balancing (default: 1000)')
    parser.add_argument('--cadences-per-bin', type=int, default=10,
                        help='(legacy) Fixed cadences per bin; mixed mode now uses --train-fraction')
    parser.add_argument('--train-fraction', type=float, default=0.5,
                        help='Fraction of cadences per bin used for training in mixed mode; '
                             'the rest are held out for inference (default: 0.5)')
    parser.add_argument('--exclude-targets', nargs='+', default=None,
                        help='Target names (e.g. TIC368536386) to keep OUT of training backgrounds; '
                             'routed to the inference pool instead')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible cadence selection')
    parser.add_argument('--training-cadences', '-t', type=int, default=None,
                        help='Number of cadences for training (rest held out for inference)')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list found cadences, do not extract')
    args = parser.parse_args()

    from rst_seti.data.background_extractor import DatasetBuilder
    from collections import defaultdict
    import random

    builder = DatasetBuilder(
        output_dir=args.output,
        snippet_width=args.snippet_width,
    )

    all_files = []
    for directory in args.scan:
        all_files.extend(builder.scan_directory(directory))

    print(f"\nTotal files: {len(all_files)}")
    builder.group_into_cadences(all_files)
    builder.print_cadence_summary()

    if args.list_only:
        return

    if args.band == 'mixed':
        print(f"\n{'=' * 60}")
        print("PROCESSING: MIXED MULTI-BAND DATASET")
        print(f"{'=' * 60}")

        complete_cadences = [c for c in builder.cadences.values() if c.is_complete]
        if not complete_cadences:
            print("No complete cadences found.")
            return

        if args.seed is not None:
            random.seed(args.seed)

        # Hold out explicitly-excluded targets (e.g. the benchmark) from training:
        # they go straight to the inference pool and never feed the background set.
        exclude = set(args.exclude_targets or [])
        excluded_cadences = [c for c in complete_cadences if c.target_name in exclude]
        complete_cadences = [c for c in complete_cadences if c.target_name not in exclude]
        if excluded_cadences:
            print(f"  Excluded {len(excluded_cadences)} cadence(s) from training "
                  f"(targets: {sorted(exclude)}) → inference pool")

        by_freq = defaultdict(list)
        for c in complete_cadences:
            bin_mhz = round(c.freq_start / args.mix_bins) * args.mix_bins
            by_freq[bin_mhz].append(c)

        selected_cadences  = []
        inference_cadences = list(excluded_cadences)
        print(f"Splitting each of {len(by_freq)} frequency bins "
              f"{args.train_fraction:.0%} train / {1 - args.train_fraction:.0%} inference:")
        for bin_mhz in sorted(by_freq.keys()):
            cads   = by_freq[bin_mhz]
            # floor() → at least half of every bin is held out for inference
            n_take = int(len(cads) * args.train_fraction)
            selected = random.sample(cads, n_take)
            selected_ids = {id(c) for c in selected}
            for c in cads:
                if id(c) not in selected_ids:
                    inference_cadences.append(c)
            selected_cadences.extend(selected)
            print(f"  - ~{bin_mhz/1000:.1f} GHz: {n_take}/{len(cads)} train, "
                  f"{len(cads)-n_take} inference")

        # Shuffle so the snippet-budget cap (if ever hit) doesn't systematically
        # starve whichever frequency bin is processed last.
        random.shuffle(selected_cadences)

        if inference_cadences:
            inf_path = Path(args.output) / "inference_cadences_mixed.txt"
            with open(inf_path, 'w') as f:
                for c in inference_cadences:
                    files_str = ','.join(str(fp) for fp in c.files)
                    f.write(f"{c.target_name}|{c.freq_start}|{files_str}\n")
            print(f"\n  Saved {len(inference_cadences)} held-out cadences: {inf_path}")

        builder.build_training_dataset(
            cadences=selected_cadences,
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=f"{args.name}_mixed",
        )
        return

    bands_to_process = (
        [args.band] if args.band != 'all' else list(builder.band_config.keys())
    )

    for band_name in bands_to_process:
        by_band       = builder.get_cadences_by_band(band_name)
        band_cadences = by_band.get(band_name, [])

        if not band_cadences:
            print(f"\n⚠️  No cadences found for {band_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {builder.band_config[band_name]['name']}")
        print(f"{'=' * 60}")
        print(f"  Cadences: {len(band_cadences)}")

        if args.training_cadences and args.training_cadences < len(band_cadences):
            training_cadences  = band_cadences[:args.training_cadences]
            inference_cadences = band_cadences[args.training_cadences:]

            print(f"  Training: {len(training_cadences)} cadences")
            print(f"  Inference: {len(inference_cadences)} cadences")

            inf_path = Path(args.output) / f"inference_cadences_{band_name}.txt"
            with open(inf_path, 'w') as f:
                for c in inference_cadences:
                    files_str = ','.join(str(fp) for fp in c.files)
                    f.write(f"{c.target_name}|{files_str}\n")
            print(f"  Saved: {inf_path}")
        else:
            training_cadences = band_cadences

        builder.build_training_dataset(
            cadences=training_cadences,
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=f"{args.name}_{band_name}",
        )

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
