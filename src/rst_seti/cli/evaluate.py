#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rst-eval — Evaluate a trained RST model on a processed .npz dataset.

Thin CLI wrapper around scripts/evaluate.py logic.

Usage:
    rst-eval --model checkpoints/best.pth --data data/processed/val.npz
    rst-eval --model checkpoints/best.pth --data data/processed/val.npz --find-optimal --plot
"""

import sys
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='rst-eval — Evaluate RST model on a processed .npz dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rst-eval -m checkpoints/best.pth -d data/processed/val.npz
  rst-eval -m checkpoints/best.pth -d data/processed/val.npz --find-optimal --plot
        """,
    )
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to .npz dataset (spectrograms + labels)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Config YAML (default: bundled default.yaml)')
    parser.add_argument('--output', '-o', type=str, default='results/eval',
                        help='Output directory (default: results/eval)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Classification threshold (default: from config)')
    parser.add_argument('--find-optimal', action='store_true',
                        help='Sweep thresholds to find the optimal F1 threshold')
    parser.add_argument('--plot', action='store_true',
                        help='Generate evaluation plots')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Batch size override')
    parser.add_argument('--device', type=str, default=None,
                        help="Device override: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID (default: "0", used only if --device not set)')
    args = parser.parse_args()

    if args.device is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Resolve config
    if args.config:
        config_path = args.config
    else:
        import importlib.resources as pkg_resources
        try:
            ref = pkg_resources.files("rst_seti.configs").joinpath("default.yaml")
            config_path = str(ref)
        except Exception:
            config_path = str(Path(__file__).parent.parent / "configs" / "default.yaml")

    # Delegate to evaluate.py logic (import from scripts or inline)
    try:
        # Try importing the evaluate script if repo layout is available
        _scripts = Path(__file__).resolve().parents[4] / "scripts"
        if _scripts.exists():
            sys.path.insert(0, str(_scripts.parent))
            from scripts.evaluate import main as _eval_main
            # Re-inject sys.argv
            _argv = ['rst-eval',
                     '--config', config_path,
                     '--model', args.model,
                     '--data', args.data,
                     '--output', args.output]
            if args.threshold is not None:
                _argv += ['--threshold', str(args.threshold)]
            if args.find_optimal:
                _argv.append('--find-optimal')
            if args.plot:
                _argv.append('--plot')
            if args.batch_size is not None:
                _argv += ['--batch_size', str(args.batch_size)]
            sys.argv = _argv
            _eval_main()
        else:
            raise ImportError("scripts/ not found")
    except ImportError:
        print("⚠️  Evaluation script not found in expected location.")
        print("   Make sure rst-seti is installed from the full repository,")
        print("   or run: python scripts/evaluate.py directly.")
        sys.exit(1)


if __name__ == '__main__':
    main()
