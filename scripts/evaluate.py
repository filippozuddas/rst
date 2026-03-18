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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.rst_model import RSTModel
from src.data.dataset import SETIDataset
from src.evaluation.metrics import print_report, compute_metrics


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
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
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

    # Print results
    print_report(labels, probs, threshold=args.threshold)

    # Save to JSON if requested
    if args.output:
        import json
        metrics = compute_metrics(labels, probs, threshold=args.threshold)
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
