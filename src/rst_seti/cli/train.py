#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rst-train — Train the RST model.

Thin CLI wrapper around scripts/train.py logic.
All training logic lives in rst_seti.training.trainer.

Usage:
    rst-train --config configs/default.yaml --gpu 0
    rst-train --config configs/default.yaml --gpu 0,1
"""

import sys
import os
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
        """,
    )
    parser.add_argument('--config', '-c', type=str, default='configs/default.yaml',
                        help='YAML config file (default: configs/default.yaml)')
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
    from rst_seti.training.trainer import Trainer

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode is not None:
        config['training']['mode'] = args.mode

    trainer = Trainer(config=config, resume_path=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
