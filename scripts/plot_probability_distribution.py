#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Probability Distribution Plot

Visualizes the distribution of ETI classification probabilities from
the inference pipeline output CSV (results_summary.csv or cadence_results.csv).

Usage:
    python scripts/plot_probability_distribution.py results/results_summary.csv
    python scripts/plot_probability_distribution.py results/results_summary.csv \
        --threshold 0.5 --output results/prob_distribution.png
    python scripts/plot_probability_distribution.py results/results_summary.csv \
        --split-by target
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path

from rst_seti.utils.visualization import (
    plot_prob_distribution,
    plot_prob_split,
    plot_prob_ccdf,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RST — Plot probability distribution from inference CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic distribution plot
  python scripts/plot_probability_distribution.py results/results_summary.csv

  # Log scale + higher bin count, custom output
  python scripts/plot_probability_distribution.py results/results_summary.csv \
      --log --bins 200 --output results/prob_dist.png

  # Split histogram by target star
  python scripts/plot_probability_distribution.py results/results_summary.csv \
      --split-by target

  # Full suite (histogram + CCDF)
  python scripts/plot_probability_distribution.py results/results_summary.csv \
      --all
        """,
    )

    parser.add_argument("csv", type=str,
                        help="Path to the inference CSV (results_summary.csv or "
                             "cadence_results.csv)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Classification threshold line (default: 0.5)")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of histogram bins (default: 100)")
    parser.add_argument("--log", action="store_true",
                        help="Use log scale on the Y axis")
    parser.add_argument("--split-by", type=str, default=None,
                        metavar="COLUMN",
                        help="Column to split histograms by (e.g. target, freq_band)")
    parser.add_argument("--ccdf", action="store_true",
                        help="Also generate a CCDF plot")
    parser.add_argument("--all", dest="all_plots", action="store_true",
                        help="Generate all available plots")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for the main plot "
                             "(default: same dir as CSV)")

    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"probability", "classification"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ CSV is missing required columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"   Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # Infer threshold from data if 'classification' column already reflects it
    n_eti = (df["classification"] == "ETI").sum()
    n_tot = len(df)
    print(f"   ETI: {n_eti:,} ({n_eti/n_tot*100:.3f}%)  |  "
          f"RFI: {n_tot - n_eti:,}  |  Threshold line: {args.threshold}")

    out_dir = csv_path.parent
    stem = csv_path.stem

    # ── Main histogram ───────────────────────────────────────────────────────
    main_out = Path(args.output) if args.output else out_dir / f"{stem}_prob_dist.png"
    plot_prob_distribution(df, args.threshold, args.bins, args.log, main_out)
    print(f"  📊 Saved: {main_out}")

    # ── Split histograms ────────────────────────────────────────────────────
    split_col = args.split_by
    if split_col or args.all_plots:
        # Auto-detect a reasonable column to split by
        if not split_col:
            for candidate in ("target", "freq_band", "date"):
                if candidate in df.columns and df[candidate].nunique() > 1:
                    split_col = candidate
                    break

        if split_col and split_col in df.columns:
            split_out = out_dir / f"{stem}_prob_dist_by_{split_col}.png"
            plot_prob_split(df, split_col, args.threshold,
                            args.bins, args.log, split_out)
            print(f"  📊 Saved: {split_out}")
        else:
            if args.split_by:
                print(f"  ⚠️  Column '{args.split_by}' not found; skipping split plot.")

    # ── CCDF ────────────────────────────────────────────────────────────────
    if args.ccdf or args.all_plots:
        ccdf_out = out_dir / f"{stem}_ccdf.png"
        plot_prob_ccdf(df, args.threshold, ccdf_out)
        print(f"  📊 Saved: {ccdf_out}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
