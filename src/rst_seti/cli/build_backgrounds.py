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
                        help='Cadences per frequency bin in mixed mode (default: 10)')
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

        by_freq = defaultdict(list)
        for c in complete_cadences:
            bin_mhz = round(c.freq_start / args.mix_bins) * args.mix_bins
            by_freq[bin_mhz].append(c)

        selected_cadences  = []
        inference_cadences = []
        print(f"Balancing dataset across {len(by_freq)} frequency bins:")
        for bin_mhz in sorted(by_freq.keys()):
            cads   = by_freq[bin_mhz]
            n_take = min(args.cadences_per_bin, len(cads))
            selected = random.sample(cads, n_take)
            for c in cads:
                if c not in selected:
                    inference_cadences.append(c)
            selected_cadences.extend(selected)
            print(f"  - ~{bin_mhz/1000:.1f} GHz: "
                  f"Selected {n_take}/{len(cads)} (left {len(cads)-n_take} for inference)")

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
