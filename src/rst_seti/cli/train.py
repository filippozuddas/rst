#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rst-train — Train the RST model.

Thin CLI wrapper around rst_seti.training.trainer.train().
Mirrors the logic of scripts/train.py exactly.

Usage:
    rst-train --config configs/default.yaml --gpu 0
    rst-train --config configs/default.yaml --gpu 0,1 --mode progressive
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='rst-train — Train the Radio Spectrogram Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rst-train --config configs/default.yaml --gpu 0
  rst-train --config configs/default.yaml --gpu 0,1 --mode progressive
  rst-train --config configs/default.yaml --gpu 0 --resume checkpoints/epoch_010.pth
        """,
    )
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID(s), comma-separated (default: "0")')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['full', 'progressive'],
                        help='Training mode override (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import yaml
    import torch
    from rst_seti.models.rst_model import RSTModel
    from rst_seti.data.dataset import create_dataloaders
    from rst_seti.training.trainer import train

    # ── Load config ────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode is not None:
        config['training']['mode'] = args.mode

    m_cfg   = config['model']
    d_cfg   = config['data']
    t_cfg   = config['training']

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────────
    model = RSTModel(
        label_dim=m_cfg['label_dim'],
        fstride=m_cfg['stride'],
        tstride=m_cfg['stride'],
        input_fdim=m_cfg['input_fdim'],
        input_tdim=m_cfg['input_tdim'],
        imagenet_pretrain=m_cfg.get('imagenet_pretrain', True),
        model_size=m_cfg['model_size'],
        verbose=True,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

    # ── Multi-GPU ──────────────────────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # ── Dataloaders ────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        train_path=d_cfg['train_path'],
        val_path=d_cfg['val_path'],
        batch_size=t_cfg['batch_size'],
        num_workers=d_cfg.get('num_workers', 4),
        mixup_alpha=t_cfg.get('mixup_alpha', 0.0),
        freq_mask_param=t_cfg.get('freq_mask_param', 0),
        time_mask_param=t_cfg.get('time_mask_param', 0),
    )

    # ── Train ──────────────────────────────────────────────────────────────
    save_dir = t_cfg.get('save_dir', 'checkpoints/')
    t_cfg['_full_config'] = config   # passed through to the training log

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=t_cfg,
        save_dir=save_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
