#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Attention Visualization

Extracts and visualizes self-attention maps from the trained Transformer model.
Shows which parts of the spectrogram the model "focuses" on when classifying.

Usage:
    python scripts/visualize_attention.py --model checkpoints/best_model.pth --data data/processed/val.npz --index 0
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.rst_model import RSTModel
from src.data.dataset import SETIDataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class AttentionExtractor:
    """Helper class to hook into the Transformer and extract attention maps."""
    def __init__(self, model):
        self.model = model
        self.attn_weights = None
        # Hook into the last attention block
        # In timm ViT: model.blocks[i].attn
        # We want the last block as it contains the most integrated information
        self.hook = self.model.v.blocks[-1].attn.attn_drop.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # The attention weights are usually calculated inside the attn module
        # In timm, we have to be careful where we hook. 
        # A more robust way is to hook the softmax output if possible,
        # but timm's forward pass is integrated.
        # Alternatively, we can use a library or a manual forward pass.
        pass

    def get_attention(self, x):
        """
        Manually run the forward pass up to the attention weights.
        This is safer than hooks for some timm versions.
        """
        model = self.model
        B = x.shape[0]
        
        # 1. Prepare input
        x = x.unsqueeze(1).transpose(2, 3) # (B, 1, 1024, 96)
        x = model.v.patch_embed(x)
        
        cls_tokens = model.v.cls_token.expand(B, -1, -1)
        dist_token = model.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + model.v.pos_embed
        x = model.v.pos_drop(x)

        # 2. Pass through blocks until the last one
        for i, blk in enumerate(model.v.blocks):
            if i < len(model.v.blocks) - 1:
                x = blk(x)
            else:
                # Last block - we need the attention weights from here
                # Manual decomposition of the last block's attention
                norm_x = blk.norm1(x)
                
                # Inside blk.attn(norm_x):
                B, N, C = norm_x.shape
                qkv = blk.attn.qkv(norm_x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                # attn shape: (B, num_heads, N, N) where N = num_patches + 2
                
                self.attn_weights = attn
                
                # Complete the block (optional, just for correctness)
                x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x_attn = blk.attn.proj(x_attn)
                x_attn = blk.attn.proj_drop(x_attn)
                x = x + blk.drop_path(x_attn)
                x = x + blk.drop_path(blk.mlp(blk.norm2(x)))

        return x, self.attn_weights


def visualize_attention(
    spec: np.ndarray,
    attn_map: np.ndarray,
    label: str,
    output_path: str = None,
    f_grid: int = 64,
    t_grid: int = 6,
):
    """
    spec: (96, 1024)
    attn_map: (num_patches,) - attention from [CLS] token to patches
    """
    # 1. Reshape attention to the patch grid (freq x time)
    # Note: in our model, patches are (freq, time) after transpose
    grid = attn_map.reshape(f_grid, t_grid)
    
    # 2. Upsample grid to (1024, 96) using bilinear interpolation
    grid_img = Image.fromarray(grid)
    grid_upsampled = np.array(grid_img.resize((96, 1024), resample=Image.BILINEAR))
    
    # Transpose back to match spec orientation
    grid_upsampled = grid_upsampled.T # (96, 1024)

    # 3. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top: Original Spectrogram
    im0 = axes[0].imshow(spec, aspect='auto', cmap='inferno', origin='upper')
    axes[0].set_title(f"Original Spectrogram ({label})", fontweight='bold')
    axes[0].set_ylabel("Time (bins)")
    fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04, label="Intensity")

    # Bottom: Spectrogram + Overlayed Attention
    axes[1].imshow(spec, aspect='auto', cmap='gray', origin='upper', alpha=0.8)
    im1 = axes[1].imshow(grid_upsampled, aspect='auto', cmap='jet', origin='upper', alpha=0.5)
    axes[1].set_title("Attention Map Overlay (Last Block)", fontweight='bold')
    axes[1].set_ylabel("Time (bins)")
    axes[1].set_xlabel("Frequency (channels)")
    fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04, label="Attention score")

    # Add lines for ON/OFF boundaries
    for i in range(1, 6):
        for ax in axes:
            ax.axhline(i*16-0.5, color='white', lw=0.5, ls='--', alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()



def main():
    parser = argparse.ArgumentParser(description='RST — Attention Map Visualization')
    parser.add_argument('--config', '-c', type=str, default='configs/default.yaml')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--data', '-d', type=str, required=True)
    parser.add_argument('--index', '-i', type=int, default=0, help="Index of the sample in the .npz")
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
    dataset = SETIDataset(args.data, is_training=False, norm_mean=config['data']['norm_mean'], norm_std=config['data']['norm_std'])
    spec_tensor, label_tensor = dataset[args.index]
    label_str = "TRUE (ETI)" if label_tensor.item() > 0.5 else "FALSE (RFI)"
    
    # Extract Attention
    extractor = AttentionExtractor(model)
    with torch.no_grad():
        _, attn_weights = extractor.get_attention(spec_tensor.unsqueeze(0).to(device))
    
    # attn_weights shape: (1, num_heads, 386, 386)
    # 386 = [CLS] + [DIST] + 384 patches
    # We want attention of [CLS] (index 0) over the patches (indices 2 to 386)
    # Average across all heads
    avg_attn = attn_weights[0].mean(dim=0) # (386, 386)
    cls_attn = avg_attn[0, 2:].cpu().numpy() # (384,)
    
    # Normalize for visualization
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    # Visualize
    spec_np = spec_tensor.numpy()
    visualize_attention(spec_np, cls_attn, label_str, output_path=args.output)


if __name__ == '__main__':
    main()
