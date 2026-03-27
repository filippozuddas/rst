# -*- coding: utf-8 -*-
"""
RST — Inference Visualization

Produces candidate spectrogram plots and attention map overlays
for ETI candidates identified during inference.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Optional

from src.models.rst_model import RSTModel


# ------------------------------------------------------------------ #
#  Attention Extraction (adapted from visualize_attention.py)
# ------------------------------------------------------------------ #
class AttentionExtractor:
    """
    Extracts self-attention weights from the last Transformer block.

    Manually decomposes the forward pass of the last block to capture
    the attention matrix after softmax.
    """

    def __init__(self, model: RSTModel):
        self.model = model

    @torch.no_grad()
    def get_attention(self, x: torch.Tensor) -> np.ndarray:
        """
        Run forward pass and return the CLS→patch attention from the last block.

        Args:
            x: Input tensor of shape (1, 96, 1024).

        Returns:
            Attention array of shape (num_patches,), normalized to [0, 1].
        """
        model = self.model
        B = x.shape[0]

        # 1. Prepare input: (B, 96, 1024) → (B, 1, 1024, 96)
        x = x.unsqueeze(1).transpose(2, 3)
        x = model.v.patch_embed(x)

        cls_tokens = model.v.cls_token.expand(B, -1, -1)
        dist_token = model.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + model.v.pos_embed
        x = model.v.pos_drop(x)

        # 2. Pass through all blocks except the last
        for i, blk in enumerate(model.v.blocks):
            if i < len(model.v.blocks) - 1:
                x = blk(x)
            else:
                # Last block: manual decomposition to get attention weights
                norm_x = blk.norm1(x)
                B, N, C = norm_x.shape
                qkv = (blk.attn.qkv(norm_x)
                       .reshape(B, N, 3, blk.attn.num_heads,
                                C // blk.attn.num_heads)
                       .permute(2, 0, 3, 1, 4))
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                # attn shape: (B, num_heads, N, N)

        # 3. Extract CLS → patch attention, averaged across heads
        # N = [CLS] + [DIST] + num_patches → patches start at index 2
        avg_attn = attn[0].mean(dim=0)       # (N, N)
        cls_attn = avg_attn[0, 2:].cpu().numpy()  # (num_patches,)

        # Normalize to [0, 1]
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

        return cls_attn


# ------------------------------------------------------------------ #
#  Plot Functions
# ------------------------------------------------------------------ #
def plot_candidate(
    spec: np.ndarray,
    prob: float,
    center_channel: int,
    freq_mhz: float,
    output_path: str,
) -> None:
    """
    Save a spectrogram plot for an ETI candidate.

    Args:
        spec: Spectrogram of shape (96, 1024), normalized.
        prob: P(ETI) from the model.
        center_channel: Center channel index.
        freq_mhz: Center frequency in MHz (0 if unknown).
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    im = ax.imshow(spec, aspect='auto', cmap='viridis', origin='upper')

    # Title with probability and frequency
    freq_str = f" — {freq_mhz:.2f} MHz" if freq_mhz > 0 else ""
    ax.set_title(
        f"ETI Candidate  |  P(ETI) = {prob:.4f}  |  "
        f"Channel {center_channel}{freq_str}",
        fontweight='bold', fontsize=13,
    )
    ax.set_ylabel("Time (bins)")
    ax.set_xlabel("Frequency (channels)")
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046,
                 pad=0.04, label="Intensity")

    # ON/OFF boundary lines
    for i in range(1, 6):
        ax.axhline(i * 16 - 0.5, color='white', lw=0.5, ls='--', alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_attention_map(
    spec: np.ndarray,
    attn_weights: np.ndarray,
    prob: float,
    center_channel: int,
    freq_mhz: float,
    output_path: str,
    f_grid: int = 64,
    t_grid: int = 6,
) -> None:
    """
    Save a spectrogram with attention map overlay.

    Args:
        spec: Spectrogram of shape (96, 1024), normalized.
        attn_weights: Attention array of shape (num_patches,), [0, 1].
        prob: P(ETI) from the model.
        center_channel: Center channel index.
        freq_mhz: Center frequency in MHz (0 if unknown).
        output_path: Path to save the plot.
        f_grid: Number of patches in the frequency dimension.
        t_grid: Number of patches in the time dimension.
    """
    # 1. Reshape attention to patch grid and upsample
    grid = attn_weights.reshape(f_grid, t_grid)
    grid_img = Image.fromarray(grid.astype(np.float32))
    grid_upsampled = np.array(
        grid_img.resize((96, 1024), resample=Image.BILINEAR)
    )
    grid_upsampled = grid_upsampled.T  # (96, 1024)

    # 2. Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    freq_str = f" — {freq_mhz:.2f} MHz" if freq_mhz > 0 else ""
    title = (f"P(ETI) = {prob:.4f}  |  "
             f"Channel {center_channel}{freq_str}")

    # Top: Original spectrogram
    im0 = axes[0].imshow(spec, aspect='auto', cmap='inferno', origin='upper')
    axes[0].set_title(f"Original Spectrogram  |  {title}", fontweight='bold')
    axes[0].set_ylabel("Time (bins)")
    fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046,
                 pad=0.04, label="Intensity")

    # Bottom: Spectrogram + attention overlay
    axes[1].imshow(spec, aspect='auto', cmap='gray', origin='upper', alpha=0.8)
    im1 = axes[1].imshow(grid_upsampled, aspect='auto', cmap='jet',
                          origin='upper', alpha=0.5)
    axes[1].set_title("Attention Map Overlay (Last Block)", fontweight='bold')
    axes[1].set_ylabel("Time (bins)")
    axes[1].set_xlabel("Frequency (channels)")
    fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046,
                 pad=0.04, label="Attention score")

    # ON/OFF boundary lines
    for i in range(1, 6):
        for ax in axes:
            ax.axhline(i * 16 - 0.5, color='white', lw=0.5, ls='--',
                       alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
