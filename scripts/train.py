#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Training Script

Main entry point for training the RST model. Loads config, creates the model,
and launches training. Supports "full" and "progressive" modes.

Usage: python scripts/train.py --config configs/default.yaml
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

# Add the parent directory to the path for imports

import torch
from rst_seti.models.rst_model import RSTModel
from rst_seti.data.dataset import create_dataloaders
from rst_seti.training.trainer import train


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # ------------------------------------------------------------------ #
    #  Parse command-line arguments
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description='RST — Radio Spectrogram Transformer Training'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to the YAML configuration file',
    )
    parser.add_argument(
        '--save_dir', type=str, default='checkpoints',
        help='Directory for saving checkpoints',
    )
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='GPU ID to use (e.g. "0" or "0,1" for multi-GPU)',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (set to 0 if out of shared memory)',
    )
    parser.add_argument(
        '--pin_memory', action='store_true', default=True,
        help='Whether to use pin_memory in DataLoader',
    )
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory',
        help='Disable pin_memory in DataLoader',
    )
    args = parser.parse_args()

    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

    # ------------------------------------------------------------------ #
    #  Load configuration
    # ------------------------------------------------------------------ #
    config = load_config(args.config)
    print(f'\nConfiguration loaded from: {args.config}')

    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['training']
    aug_cfg = config['augmentation']

    # ------------------------------------------------------------------ #
    #  Create the model
    # ------------------------------------------------------------------ #
    print('\n--- Creating model ---')
    model = RSTModel(
        label_dim=model_cfg['label_dim'],
        fstride=model_cfg['stride'],
        tstride=model_cfg['stride'],
        input_fdim=model_cfg['input_fdim'],
        input_tdim=model_cfg['input_tdim'],
        imagenet_pretrain=model_cfg['imagenet_pretrain'],
        model_size=model_cfg['model_size'],
    )

    # Multi-GPU (if available)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = torch.nn.DataParallel(model)

    # ------------------------------------------------------------------ #
    #  Create DataLoaders
    # ------------------------------------------------------------------ #
    print('\n--- Loading data ---')
    print(f"Train data path: {data_cfg['train_data']}")
    print(f"Val data path:   {data_cfg['val_data']}")
    train_loader, val_loader = create_dataloaders(
        train_path=data_cfg['train_data'],
        val_path=data_cfg['val_data'],
        batch_size=train_cfg['batch_size'],
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        freq_mask=aug_cfg['freq_mask'],
        time_mask=aug_cfg['time_mask'],
        mixup_alpha=aug_cfg['mixup_alpha'],
    )

    # ------------------------------------------------------------------ #
    #  Launch training
    # ------------------------------------------------------------------ #
    print('\n--- Starting training ---')

    # If model is wrapped in DataParallel, access the inner model
    # for progressive unfreezing functions
    actual_model = model.module if hasattr(model, 'module') else model

    # Inject full config to save in the JSON training log
    train_cfg['_full_config'] = config

    history = train(
        model=model,  # <-- BUG FIX: We must pass the DataParallel-wrapped model, not actual_model!
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        save_dir=args.save_dir,
        device=device,
    )

    print('\n--- Training complete! ---')
    print(f'Best val loss: {min(history["val_loss"]):.4f}')
    print(f'Checkpoints saved to: {args.save_dir}/')


if __name__ == '__main__':
    main()
