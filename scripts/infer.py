#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Inference Pipeline

End-to-end inference on raw HDF5 cadences using the trained RST model.
Supports single-cadence mode (--files) and directory-scan mode (--scan).

Usage:
    # Single cadence
    python scripts/infer.py -m checkpoints/best.pth \\
        -f obs1_ON.h5 obs2_OFF.h5 obs3_ON.h5 obs4_OFF.h5 obs5_ON.h5 obs6_OFF.h5

    # Directory scan
    python scripts/infer.py -m checkpoints/best.pth -s /path/to/observations/
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.engine import InferenceEngine
from src.utils.visualization import (
    AttentionExtractor,
    plot_candidate,
    plot_attention_map,
)
from src.data.background_extractor import DatasetBuilder


def load_cadence_from_files(file_paths: list) -> tuple:
    """
    Load a cadence from 6 HDF5 files using blimpy.

    Args:
        file_paths: List of 6 HDF5 file paths (ON/OFF/ON/OFF/ON/OFF order).

    Returns:
        Tuple of (cadence_array, freq_start_mhz, freq_resolution_mhz, target_name).
        cadence_array has shape (6, 16, n_freq).
    """
    from blimpy import Waterfall

    cadence_data = []
    freq_start_mhz = 0.0
    freq_resolution_mhz = 0.0
    target_name = "unknown"

    # Try to extract target name from filename
    import re
    for pattern in [
        re.compile(r'(TIC\d+)_(ON|OFF)'),
        re.compile(r'(HIP\d+|GJ\d+[A-Za-z]?)_(ON|OFF)'),
        re.compile(r'([A-Za-z0-9_]+?)_(ON|OFF)(?:_|\.)'),
    ]:
        match = pattern.search(Path(file_paths[0]).stem)
        if match:
            target_name = match.group(1)
            break

    for i, fpath in enumerate(file_paths):
        print(f"  Loading file {i+1}/6: {Path(fpath).name}...", end=" ",
              flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = Waterfall(str(fpath))

            data = wf.data.squeeze()
            cadence_data.append(data)

            # Get frequency info from the first file
            if i == 0:
                freq_start_mhz = wf.header.get('fch1', 0.0)
                freq_resolution_mhz = abs(wf.header.get('foff', 0.0))

            print(f"✓ ({data.shape})")
        except OSError as e:
            print(f"❌ FAILED")
            if "truncated file" in str(e).lower():
                raise OSError(f"File is truncated or corrupt: {fpath}") from e
            raise OSError(f"Could not open HDF5 file {fpath}: {e}") from e
        except Exception as e:
            print(f"❌ ERROR")
            raise RuntimeError(f"Unexpected error loading {fpath}: {e}") from e

    cadence_array = np.stack(cadence_data, axis=0)
    return cadence_array, freq_start_mhz, freq_resolution_mhz, target_name


def process_cadence(
    engine: InferenceEngine,
    cadence: np.ndarray,
    freq_start_mhz: float,
    freq_resolution_mhz: float,
    target_name: str,
    output_dir: Path,
    threshold: float,
    attn_threshold: float,
    generate_plots: bool,
    top_n: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single cadence: inference + clustering + plots + attention maps.

    Returns:
        Tuple of (raw_results DataFrame, clusters DataFrame).
    """
    # 1. Run inference + clustering
    results, clusters = engine.run_cadence(
        cadence, freq_start_mhz, freq_resolution_mhz,
    )

    if results.empty:
        return results, clusters

    # 2. Save CSVs (raw + clustered)
    csv_path = output_dir / "cadence_results.csv"
    results.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  📄 Saved: {csv_path}")

    if not clusters.empty:
        cluster_csv = output_dir / "cadence_clusters.csv"
        clusters.to_csv(cluster_csv, index=False, float_format='%.6f')
        print(f"  📄 Saved: {cluster_csv}")

    if not generate_plots:
        return results, clusters

    # 3. Generate plots for top-N clusters (peak snippet of each cluster)
    if not clusters.empty and top_n > 0:
        to_plot = clusters.head(top_n)
        skipped = len(clusters) - len(to_plot)
        skip_msg = f" (skipped {skipped})" if skipped > 0 else ""
        print(f"  🎨 Generating {len(to_plot)} cluster plot(s)"
              f"{skip_msg}...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(to_plot.iterrows(),
                           total=len(to_plot),
                           desc="    Plots", leave=False):
            center = int(row['center_channel'])
            prob = row['peak_probability']
            freq = row['freq_mhz']
            n_snip = int(row['n_snippets'])

            spec = engine.get_snippet_spectrogram(cadence, center)

            freq_str = f"{freq:.2f}MHz" if freq > 0 else "nofreq"
            fname = f"cluster_ch{center:05d}_{freq_str}_p{prob:.2f}_n{n_snip}.png"

            plot_candidate(
                spec=spec, prob=prob, center_channel=center,
                freq_mhz=freq, output_path=str(plots_dir / fname),
            )

    # 4. Generate attention maps for high-confidence clusters (peak_prob >= attn_threshold)
    if not clusters.empty and top_n > 0:
        high_conf = clusters[clusters['peak_probability'] >= attn_threshold]
        to_attn = high_conf.head(top_n)

        if len(to_attn) > 0:
            print(f"  🧠 Generating {len(to_attn)} attention map(s)...")
            attn_dir = output_dir / "attention_maps"
            attn_dir.mkdir(parents=True, exist_ok=True)

            extractor = AttentionExtractor(engine.model)

            for _, row in tqdm(to_attn.iterrows(),
                               total=len(to_attn),
                               desc="    Attention", leave=False):
                center = int(row['center_channel'])
                prob = row['peak_probability']
                freq = row['freq_mhz']

                spec = engine.get_snippet_spectrogram(cadence, center)
                spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(engine.device)

                attn_weights = extractor.get_attention(spec_tensor)

                freq_str = f"{freq:.2f}MHz" if freq > 0 else "nofreq"
                fname = (f"cluster_ch{center:05d}_{freq_str}"
                         f"_p{prob:.2f}_attn.png")

                plot_attention_map(
                    spec=spec, attn_weights=attn_weights,
                    prob=prob, center_channel=center,
                    freq_mhz=freq, output_path=str(attn_dir / fname),
                )

    return results, clusters


def main():
    parser = argparse.ArgumentParser(
        description='RST — End-to-End Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single cadence:
  python scripts/infer.py -m checkpoints/best.pth \\
      -f ON1.h5 OFF1.h5 ON2.h5 OFF2.h5 ON3.h5 OFF3.h5

  # Directory scan:
  python scripts/infer.py -m checkpoints/best.pth -s /data/observations/
        """,
    )

    # Required
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model checkpoint (.pth)')

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--files', '-f', nargs=6, type=str,
                             help='6 HDF5 files (ON/OFF order)')
    input_group.add_argument('--scan', '-s', type=str,
                             help='Directory to scan for cadences')

    # Options
    parser.add_argument('--config', '-c', type=str,
                        default='configs/default.yaml',
                        help='Config file (default: configs/default.yaml)')
    parser.add_argument('--output', '-o', type=str, default='results/',
                        help='Output directory (default: results/)')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                        help='Batch size (default: from config, 128)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='ETI threshold (default: from config, 0.5)')
    parser.add_argument('--attn_threshold', type=float, default=None,
                        help='Attention map threshold (default: from config, 0.9)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID (default: "0")')
    parser.add_argument('--band', choices=['6GHz', '18GHz', '1.4GHz', 'all'],
                        default='all',
                        help='Frequency band filter for --scan mode')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--top_n', '-n', type=int, default=200,
                        help='Max plots per cadence (default: 200, 0=all)')

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RST — Inference Pipeline")
    print("=" * 60)

    # Initialize engine
    engine = InferenceEngine(
        config_path=args.config,
        checkpoint_path=args.model,
    )

    # Override config values if CLI args are provided
    if args.batch_size is not None:
        engine.batch_size = args.batch_size
    if args.threshold is not None:
        engine.threshold = args.threshold
    if args.attn_threshold is not None:
        engine.attn_threshold = args.attn_threshold

    threshold = engine.threshold
    attn_threshold = engine.attn_threshold

    all_results = []
    all_clusters = []

    # ------------------------------------------------------------------ #
    #  Mode 1: Single cadence from 6 files
    # ------------------------------------------------------------------ #
    if args.files:
        print(f"\n📁 Mode: Single Cadence (6 files)")
        try:
            cadence, freq_start, freq_res, target = load_cadence_from_files(
                args.files
            )

            cadence_dir = output_dir / target
            cadence_dir.mkdir(parents=True, exist_ok=True)

            results, clusters = process_cadence(
                engine=engine, cadence=cadence,
                freq_start_mhz=freq_start,
                freq_resolution_mhz=freq_res,
                target_name=target,
                output_dir=cadence_dir,
                threshold=threshold,
                attn_threshold=attn_threshold,
                generate_plots=not args.no_plots,
                top_n=args.top_n,
            )
            if not results.empty:
                results['target'] = target
                all_results.append(results)
            if not clusters.empty:
                clusters['target'] = target
                all_clusters.append(clusters)
        except Exception as e:
            print(f"\n❌ Error processing cadence: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Mode 2: Directory scan
    # ------------------------------------------------------------------ #
    elif args.scan:
        print(f"\n📁 Mode: Directory Scan ({args.scan})")
        builder = DatasetBuilder(output_dir=str(output_dir))
        files = builder.scan_directory(args.scan)
        cadence_infos = builder.group_into_cadences(files)
        builder.print_cadence_summary()

        # Filter by band if specified
        complete_cadences = [c for c in cadence_infos.values()
                             if c.is_complete]
        if args.band != 'all':
            complete_cadences = [c for c in complete_cadences
                                 if c.freq_band == args.band]

        if not complete_cadences:
            print("⚠️  No complete cadences found!")
            return

        print(f"\n🔬 Processing {len(complete_cadences)} cadence(s)...")

        for idx, cadence_info in enumerate(complete_cadences, 1):
            print(f"\n{'─' * 60}")
            print(f"  Cadence {idx}/{len(complete_cadences)}: "
                  f"{cadence_info.target_name} ({cadence_info.freq_band})")
            print(f"{'─' * 60}")

            try:
                cadence, freq_start, freq_res, target = \
                    load_cadence_from_files(
                        [str(f) for f in cadence_info.files]
                    )
                target = cadence_info.target_name

                cadence_dir = output_dir / f"{target}_{cadence_info.date}"
                cadence_dir.mkdir(parents=True, exist_ok=True)

                results, clusters = process_cadence(
                    engine=engine, cadence=cadence,
                    freq_start_mhz=freq_start,
                    freq_resolution_mhz=freq_res,
                    target_name=target,
                    output_dir=cadence_dir,
                    threshold=threshold,
                    attn_threshold=attn_threshold,
                    generate_plots=not args.no_plots,
                    top_n=args.top_n,
                )

                if not results.empty:
                    results['target'] = target
                    results['date'] = cadence_info.date
                    results['freq_band'] = cadence_info.freq_band
                    all_results.append(results)
                if not clusters.empty:
                    clusters['target'] = target
                    clusters['date'] = cadence_info.date
                    clusters['freq_band'] = cadence_info.freq_band
                    all_clusters.append(clusters)

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

    # ------------------------------------------------------------------ #
    #  Save global summary
    # ------------------------------------------------------------------ #
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = output_dir / "results_summary.csv"
        summary.to_csv(summary_path, index=False, float_format='%.6f')

        n_total = len(summary)
        n_eti = (summary['classification'] == 'ETI').sum()

        # Save cluster summary
        n_clusters = 0
        n_high_clusters = 0
        if all_clusters:
            cluster_summary = pd.concat(all_clusters, ignore_index=True)
            cluster_path = output_dir / "clusters_summary.csv"
            cluster_summary.to_csv(cluster_path, index=False, float_format='%.6f')
            n_clusters = len(cluster_summary)
            n_high_clusters = (cluster_summary['peak_probability'] >= attn_threshold).sum()

        print(f"\n{'=' * 60}")
        print(f"  INFERENCE COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total snippets analyzed: {n_total:,}")
        print(f"  ETI snippets (p≥{threshold}): {n_eti:,}")
        print(f"  Distinct signals (clusters): {n_clusters:,}")
        print(f"  High-confidence clusters (p≥{attn_threshold}): {n_high_clusters:,}")
        print(f"  📄 Summary: {summary_path}")
        if all_clusters:
            print(f"  📄 Clusters: {cluster_path}")
        print(f"  📁 Output:  {output_dir}/")
    else:
        print("\n⚠️  No results generated.")


if __name__ == '__main__':
    main()
