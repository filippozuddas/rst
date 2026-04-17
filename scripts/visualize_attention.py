#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Attention Visualization

Extracts and visualizes self-attention maps from the trained Transformer model.
Shows which parts of the spectrogram the model "focuses" on when classifying.

Usage:
    python scripts/visualize_attention.py --model checkpoints/best_model.pth --data data/processed/val.npz --num_plots 5
"""

import sys
import os
import random
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.rst_model import RSTModel
from src.data.dataset import SETIDataset
from src.utils.visualization import AttentionExtractor, plot_attention_map


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='RST — Attention Map Visualization')
    parser.add_argument('--config', '-c', type=str, default='configs/default.yaml')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--data', '-d', type=str, required=True)
    parser.add_argument('--num_plots', '-n', type=int, default=1, help="Number of random samples to visualize")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output', '-o', type=str, default='attention_map.png')

    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    config = load_config(args.config)
    m_cfg = config['model']

    # Load Model
    model = RSTModel(
        imagenet_pretrain=False,
        model_size=m_cfg['model_size'],
        input_fdim=m_cfg['input_fdim'],
        input_tdim=m_cfg['input_tdim'],
        label_dim=m_cfg['label_dim'],
        verbose=False
    )
    checkpoint = torch.load(args.model, map_location='cpu')
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load Data
    dataset = SETIDataset(args.data, is_training=False)

    # Randomly sample indices
    num_samples = len(dataset)
    num_plots = min(args.num_plots, num_samples)
    indices = random.sample(range(num_samples), num_plots)
    print(f"📊 Generating {num_plots} attention maps from {num_samples} samples...")
    print(f"   Selected indices: {indices}")

    # Prepare output path
    out_base, out_ext = os.path.splitext(args.output)

    extractor = AttentionExtractor(model)

    for i, idx in enumerate(indices):
        spec_tensor, label_tensor = dataset[idx]
        label_str = "TRUE (ETI)" if label_tensor.item() > 0.5 else "FALSE (RFI)"

        # Extract Attention
        cls_attn = extractor.get_attention(spec_tensor.unsqueeze(0).to(device))

        # Output path: attention_map_0.png, attention_map_1.png, ...
        if num_plots == 1:
            out_path = args.output
        else:
            out_path = f"{out_base}_{i}{out_ext}"

        # Visualize
        spec_np = spec_tensor.numpy()
        plot_attention_map(
            spec=spec_np,
            attn_weights=cls_attn,
            output_path=out_path,
            custom_title=f"{label_str} [idx={idx}]"
        )
        print(f"✅ Saved visualization to: {out_path}")

    print(f"\n✅ Done! Generated {num_plots} attention map(s).")


if __name__ == '__main__':
    main()
