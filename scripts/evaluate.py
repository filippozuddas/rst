#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Evaluation Script

Evaluates a trained model on a validation/test set.
Loads the model checkpoint and computes metrics (AUC-ROC, AUC-PR, F1, etc.)
using the metrics module.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --model checkpoints/best_model.pth --data data/processed/val.npz
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

# Add project root to path

from rst_seti.models.rst_model import RSTModel
from rst_seti.data.dataset import SETIDataset
from rst_seti.evaluation.metrics import print_report, compute_metrics
from rst_seti.utils.visualization import plot_threshold_sweep


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: SETIDataset,
    batch_size: int = 32,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the dataset and return (labels, probabilities)."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    all_probs = []
    all_labels = []

    print(f"  Evaluating on {len(dataset)} samples...")
    for specs, labels in tqdm(loader, desc="    Inference"):
        specs = specs.to(device)
        
        # Forward pass
        with torch.amp.autocast('cuda'):
            logits = model(specs)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    return all_labels, all_probs


def find_optimal_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = 'f1',
    n_steps: int = 199,
) -> dict:
    """
    Sweep thresholds and return the one that maximises the chosen metric.

    Args:
        labels:  Ground-truth binary labels.
        probs:   Model output probabilities in [0, 1].
        metric:  'f1' (default), 'precision', or 'recall'.
        n_steps: Number of threshold values to evaluate.

    Returns:
        dict with keys: optimal_threshold, best_f1, best_precision,
        best_recall, thresholds, f1_scores, precisions, recalls.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(labels, preds, zero_division=0))
        precs.append(precision_score(labels, preds, zero_division=0))
        recs.append(recall_score(labels, preds, zero_division=0))

    f1s = np.array(f1s)
    precs = np.array(precs)
    recs = np.array(recs)

    best_idx = int(np.argmax(f1s))
    opt_t = float(thresholds[best_idx])

    print("\n" + "=" * 52)
    print("  Threshold Sweep (metric: F1-score maximisation)")
    print("=" * 52)
    print(f"  {'Threshold':>10}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}")
    print("  " + "-" * 48)

    # Print a compact table at a few key thresholds
    display_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t_disp in display_thresholds:
        idx = int(np.argmin(np.abs(thresholds - t_disp)))
        marker = " ◄ OPTIMAL" if idx == best_idx else ""
        print(f"  {thresholds[idx]:>10.2f}  {f1s[idx]:>8.4f}  "
              f"{precs[idx]:>10.4f}  {recs[idx]:>8.4f}{marker}")

    # Always show the true optimum if it's not in the display list
    if not any(abs(opt_t - t) < 0.01 for t in display_thresholds):
        print(f"  {opt_t:>10.2f}  {f1s[best_idx]:>8.4f}  "
              f"{precs[best_idx]:>10.4f}  {recs[best_idx]:>8.4f}  ◄ OPTIMAL")

    print("=" * 52)
    print(f"\n  ✅ Optimal threshold: {opt_t:.4f}")
    print(f"     F1={f1s[best_idx]:.4f}  "
          f"Precision={precs[best_idx]:.4f}  "
          f"Recall={recs[best_idx]:.4f}")

    return {
        'optimal_threshold': opt_t,
        'best_f1': float(f1s[best_idx]),
        'best_precision': float(precs[best_idx]),
        'best_recall': float(recs[best_idx]),
        'thresholds': thresholds.tolist(),
        'f1_scores': f1s.tolist(),
        'precisions': precs.tolist(),
        'recalls': recs.tolist(),
    }





def main():
    parser = argparse.ArgumentParser(description='RST — Model Evaluation')
    parser.add_argument('--config', '-c', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to evaluation data (.npz)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID (default: "0")')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save results (JSON)')
    parser.add_argument('--find-optimal', action='store_true',
                        help='Sweep thresholds and report the one maximising F1')
    parser.add_argument('--plot', action='store_true',
                        help='Plot F1/P/R vs threshold and PR curve (requires --find-optimal)')

    args = parser.parse_args()

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    config = load_config(args.config)
    m_cfg = config['model']

    # Initialize model
    print("\n--- Initializing Model ---")
    model = RSTModel(
        label_dim=m_cfg['label_dim'],
        fstride=m_cfg['stride'],
        tstride=m_cfg['stride'],
        input_fdim=m_cfg['input_fdim'],
        input_tdim=m_cfg['input_tdim'],
        imagenet_pretrain=False,  # Weights will be loaded from checkpoint
        model_size=m_cfg['model_size'],
    )

    # Load checkpoint
    print(f"--- Loading weights from: {args.model} ---")
    state_dict = torch.load(args.model, map_location=device)
    
    # Handle DataParallel prefix if necessary
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)

    # Prepare dataset
    print(f"\n--- Loading Data: {args.data} ---")
    dataset = SETIDataset(
        data_path=args.data,
        is_training=False,
    )

    # Run evaluation
    labels, probs = evaluate(model, dataset, args.batch_size, device)

    # Threshold sweep (optional)
    sweep_results = None
    if args.find_optimal:
        sweep_results = find_optimal_threshold(labels, probs)
        # Override threshold with the optimal one for the report below
        args.threshold = sweep_results['optimal_threshold']

        # Plot if requested
        if args.plot:
            plot_path = (
                str(Path(args.output).with_suffix('.png'))
                if args.output else 'threshold_analysis.png'
            )
            plot_threshold_sweep(sweep_results, save_path=plot_path)
            print(f"\n  📈 Plot saved to: {plot_path}")

    # Print full classification report at chosen threshold
    print_report(labels, probs, threshold=args.threshold)

    # Save to JSON if requested
    if args.output:
        import json
        metrics = compute_metrics(labels, probs, threshold=args.threshold)
        if sweep_results is not None:
            metrics['threshold_sweep'] = sweep_results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
