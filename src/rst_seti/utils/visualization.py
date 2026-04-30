# -*- coding: utf-8 -*-
"""
RST — Visualization Utilities

Centralized module for all plotting functions across the project.
Includes functions for spectrograms, attention maps, evaluation metrics,
and probability distributions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any

from rst_seti.models.rst_model import RSTModel


# ------------------------------------------------------------------ #
#  Design System & Styling
# ------------------------------------------------------------------ #

# Light theme colors
BG_COLOR     = "#ffffff"
PANEL_COLOR  = "#f8f9fa"
GRID_COLOR   = "#e5e7eb"
TEXT_COLOR   = "#1f2937"
MUTED_COLOR  = "#6b7280"

RFI_COLOR    = "#3b82f6"   # blue
ETI_COLOR    = "#f97316"   # orange
THRESH_COLOR = "#f43f5e"   # rose

FONT_FAMILY  = "DejaVu Sans"
DPI          = 300


def apply_light_style() -> None:
    """Apply a consistent light aesthetic to all subsequent matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    PANEL_COLOR,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.grid":         False,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.7,
        "xtick.color":       MUTED_COLOR,
        "ytick.color":       MUTED_COLOR,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.facecolor":  PANEL_COLOR,
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "font.family":       FONT_FAMILY,
        "figure.dpi":        DPI,
    })


def _threshold_annotation(ax: plt.Axes, threshold: float, y_max: float) -> None:
    """Draw a vertical dashed line + label for the classification threshold."""
    ax.axvline(
        threshold, color=THRESH_COLOR, linewidth=1.5,
        linestyle="--", alpha=0.9, zorder=5,
    )
    ax.text(
        threshold + 0.01, y_max * 0.95,
        f"threshold\n{threshold:.2f}",
        color=THRESH_COLOR, fontsize=9, va="top", ha="left",
        path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_COLOR)],
    )


# ------------------------------------------------------------------ #
#  Attention Extraction
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
#  Spectrogram Plots
# ------------------------------------------------------------------ #

def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Spectrogram",
    output_path: Optional[str] = None,
    cmap: str = "inferno",
    show_onoff_lines: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a generic spectrogram, useful for debugging or notebooks.

    Args:
        spec:             2-D array (time, freq).
        title:            Figure title.
        output_path:      If given, saves to disk at DPI=300.
        cmap:             Matplotlib colormap.
        show_onoff_lines: Draw dashed white lines at ON/OFF boundaries.
        show:             If True, call plt.show() — useful in Jupyter.

    Returns:
        The matplotlib Figure object.
    """
    apply_light_style()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    im = ax.imshow(spec, aspect='auto', cmap=cmap, origin='upper',
                   interpolation='nearest')

    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_ylabel("Time (bins)")
    ax.set_xlabel("Frequency (channels)")
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046,
                 pad=0.04, label="Intensity")

    if show_onoff_lines:
        for i in range(1, 6):
            ax.axhline(i * 16 - 0.5, color='white', lw=0.5, ls='--', alpha=0.5)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


_OBS_LABELS = ['ON-1', 'OFF-1', 'ON-2', 'OFF-2', 'ON-3', 'OFF-3']
_ON_COLOR    = '#4ade80'   # green
_OFF_COLOR   = '#f87171'   # red


def plot_cadence_panels(
    cadence: np.ndarray,
    title: str = "Cadence",
    output_path: Optional[str] = None,
    cmap: str = "inferno",
    tchans_obs: int = 16,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a 6-panel cadence waterfall (turboSETI-style).

    Accepts either:
      - stacked spectrogram of shape (96, n_freq), split internally into 6×16 panels
      - pre-split array of shape (6, tchans_obs, n_freq)

    Args:
        cadence:      Input array, shape (96, W) or (6, tchans_obs, W).
        title:        Figure suptitle.
        output_path:  If given, saves the figure to disk at DPI=300.
        cmap:         Matplotlib colormap.
        tchans_obs:   Time bins per observation (used when cadence is stacked).
        show:         If True, call plt.show() — useful in Jupyter notebooks.

    Returns:
        The matplotlib Figure object.
    """
    apply_light_style()

    # ── Normalize input to (6, tchans_obs, n_freq) ──────────────────────────
    if cadence.ndim == 2:
        # stacked (96, W) → split into 6 panels
        n_obs = cadence.shape[0] // tchans_obs
        panels = [cadence[i * tchans_obs:(i + 1) * tchans_obs, :]
                  for i in range(n_obs)]
    elif cadence.ndim == 3:
        panels = [cadence[i] for i in range(cadence.shape[0])]
    else:
        raise ValueError(f"cadence must be 2-D or 3-D, got shape {cadence.shape}")

    n_obs = len(panels)

    # Common intensity scale (2nd–98th percentile) across all panels
    all_data = np.concatenate([p.flatten() for p in panels])
    vmin = np.percentile(all_data, 2)
    vmax = np.percentile(all_data, 98)

    fig, axes = plt.subplots(
        n_obs, 1,
        figsize=(12, 1.3 * n_obs),
        sharex=True, sharey=True,
    )
    if n_obs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(
            panels[i], aspect='auto', cmap=cmap,
            vmin=vmin, vmax=vmax, origin='upper',
            interpolation='nearest', rasterized=True,
        )

        # ON/OFF label badge inside each panel
        is_on = (i % 2 == 0)
        label_color = _ON_COLOR if is_on else _OFF_COLOR
        ax.text(
            0.015, 0.5, _OBS_LABELS[i],
            transform=ax.transAxes, va='center', ha='left',
            fontsize=11, fontweight='bold', color=label_color,
            bbox=dict(facecolor='black', alpha=0.55,
                      edgecolor='none', boxstyle='round,pad=0.3'),
        )

        ax.set_ylabel('Time', fontsize=10)
        ax.set_yticks([])
        if i < n_obs - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel('Frequency Channel')
    if title:
        axes[0].set_title(title, fontweight='bold', fontsize=14, pad=8)

    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02,
                 label='Normalized Intensity')
    plt.subplots_adjust(hspace=0, wspace=0)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                    facecolor=BG_COLOR)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_candidate(
    spec: np.ndarray,
    prob: float,
    center_channel: int,
    freq_mhz: float,
    output_path: str,
) -> None:
    """
    Save a spectrogram plot for an ETI candidate.
    """
    apply_light_style()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    im = ax.imshow(spec, aspect='auto', cmap='inferno', origin='upper',
                   interpolation='nearest')

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

    for i in range(1, 6):
        ax.axhline(i * 16 - 0.5, color='white', lw=0.5, ls='--', alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


def plot_attention_map(
    spec: np.ndarray,
    attn_weights: np.ndarray,
    prob: float = 0.0,
    center_channel: int = 0,
    freq_mhz: float = 0.0,
    output_path: Optional[str] = None,
    f_grid: int = 64,
    t_grid: int = 6,
    custom_title: Optional[str] = None,
) -> None:
    """
    Save a spectrogram with attention map overlay.
    """
    apply_light_style()
    # 1. Reshape attention to patch grid and upsample
    grid = attn_weights.reshape(f_grid, t_grid)
    grid_img = Image.fromarray(grid.astype(np.float32))
    grid_upsampled = np.array(
        grid_img.resize((96, 1024), resample=Image.BILINEAR)
    )
    grid_upsampled = grid_upsampled.T  # (96, 1024)

    # 2. Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    if custom_title:
        title = custom_title
    else:
        freq_str = f" — {freq_mhz:.2f} MHz" if freq_mhz > 0 else ""
        title = (f"P(ETI) = {prob:.4f}  |  "
                 f"Channel {center_channel}{freq_str}")

    # Top: Original spectrogram
    im0 = axes[0].imshow(spec, aspect='auto', cmap='inferno', origin='upper',
                         interpolation='nearest')
    axes[0].set_title(f"Original Spectrogram  |  {title}", fontweight='bold')
    axes[0].set_ylabel("Time (bins)")
    fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046,
                 pad=0.04, label="Intensity")

    # Bottom: Spectrogram + attention overlay
    axes[1].imshow(spec, aspect='auto', cmap='gray', origin='upper', alpha=0.8,
                   interpolation='nearest')
    im1 = axes[1].imshow(grid_upsampled, aspect='auto', cmap='jet',
                         origin='upper', alpha=0.5, interpolation='bilinear')
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
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Evaluation / Metrics Plots
# ------------------------------------------------------------------ #

def plot_threshold_sweep(sweep: Dict[str, Any], save_path: str) -> None:
    """
    Plot F1 / Precision / Recall vs threshold, plus the PR curve.
    """
    apply_light_style()
    thresholds = np.array(sweep['thresholds'])
    f1s        = np.array(sweep['f1_scores'])
    precs      = np.array(sweep['precisions'])
    recs       = np.array(sweep['recalls'])
    opt_t      = sweep['optimal_threshold']
    opt_f1     = sweep['best_f1']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RST — Threshold Analysis', fontsize=14, fontweight='bold')

    # ── Panel 1: metrics vs threshold ──────────────────────────────────────
    ax = axes[0]
    ax.plot(thresholds, f1s,   color='#2196F3', lw=2,   label='F1-score')
    ax.plot(thresholds, precs, color='#4CAF50', lw=1.5, label='Precision', ls='--')
    ax.plot(thresholds, recs,  color='#FF5722', lw=1.5, label='Recall',    ls=':')
    ax.axvline(opt_t, color='#FFC107', lw=1.5, ls='--',
               label=f'Optimal t={opt_t:.3f}  (F1={opt_f1:.4f})')
    ax.scatter([opt_t], [opt_f1], color='#FFC107', zorder=5, s=80)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs Threshold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.01)
    ax.grid(alpha=0.3)

    # ── Panel 2: Precision-Recall curve ────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(recs, precs, color='#9C27B0', lw=2)
    # Mark the optimal operating point
    opt_idx = int(np.argmin(np.abs(thresholds - opt_t)))
    ax2.scatter([recs[opt_idx]], [precs[opt_idx]], color='#FFC107',
                zorder=5, s=80, label=f't={opt_t:.3f}')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.01)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Distribution Plots
# ------------------------------------------------------------------ #

def plot_prob_distribution(
    df: pd.DataFrame,
    threshold: float,
    bins: int,
    log_scale: bool,
    output_path: Path,
) -> None:
    """Single-panel histogram of the full probability distribution."""
    apply_light_style()

    probs = df["probability"].values
    n_rfi = (df["classification"] == "RFI").sum()
    n_eti = (df["classification"] == "ETI").sum()
    n_tot = len(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate distributions
    rfi_probs = probs[probs < threshold]
    eti_probs = probs[probs >= threshold]

    bin_edges = np.linspace(0, 1, bins + 1)

    ax.hist(
        rfi_probs, bins=bin_edges,
        color=RFI_COLOR, alpha=0.75, label=f"RFI  ({n_rfi:,})",
        edgecolor=BG_COLOR, linewidth=0.3,
    )
    ax.hist(
        eti_probs, bins=bin_edges,
        color=ETI_COLOR, alpha=0.85, label=f"ETI  ({n_eti:,})",
        edgecolor=BG_COLOR, linewidth=0.3,
    )

    if log_scale:
        ax.set_yscale("log")

    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.5)

    y_max = ax.get_ylim()[1]
    _threshold_annotation(ax, threshold, y_max)

    # Stats box
    stats_text = (
        f"Total snippets: {n_tot:,}\n"
        f"ETI rate: {n_eti/n_tot*100:.3f}%\n"
        f"Mean P(ETI): {probs.mean():.4f}\n"
        f"Median P(ETI): {np.median(probs):.4f}"
    )
    ax.text(
        0.99, 0.97, stats_text,
        transform=ax.transAxes, fontsize=9,
        va="top", ha="right", color=MUTED_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                  edgecolor=GRID_COLOR, alpha=0.9),
    )

    ax.set_xlabel("P(ETI signal)", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of snippets" + (" (log)" if log_scale else ""),
                  fontsize=12, labelpad=8)
    ax.set_title("RST — Classification Probability Distribution",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="upper center", fontsize=10, framealpha=0.8)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def plot_prob_split(
    df: pd.DataFrame,
    split_col: str,
    threshold: float,
    bins: int,
    log_scale: bool,
    output_path: Path,
) -> None:
    """One subplot per unique value of *split_col* (e.g. target or freq_band)."""
    apply_light_style()

    groups = sorted(df[split_col].unique())
    n_groups = len(groups)

    ncols = min(3, n_groups)
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"RST — Probability Distribution by {split_col.replace('_', ' ').title()}",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    bin_edges = np.linspace(0, 1, bins + 1)

    for idx, (group, ax) in enumerate(zip(groups, axes.flat)):
        sub = df[df[split_col] == group]
        probs = sub["probability"].values
        n_rfi = (sub["classification"] == "RFI").sum()
        n_eti = (sub["classification"] == "ETI").sum()

        rfi_p = probs[probs < threshold]
        eti_p = probs[probs >= threshold]

        ax.hist(rfi_p, bins=bin_edges, color=RFI_COLOR, alpha=0.75,
                label=f"RFI ({n_rfi:,})", edgecolor=BG_COLOR, linewidth=0.3)
        ax.hist(eti_p, bins=bin_edges, color=ETI_COLOR, alpha=0.85,
                label=f"ETI ({n_eti:,})", edgecolor=BG_COLOR, linewidth=0.3)

        if log_scale:
            ax.set_yscale("log")

        ax.axvline(threshold, color=THRESH_COLOR, linewidth=1.2,
                   linestyle="--", alpha=0.9)

        ax.set_title(str(group), fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("P(ETI)", fontsize=9)
        ax.set_ylabel("Count" + (" (log)" if log_scale else ""), fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.8)

        eti_rate = n_eti / max(len(sub), 1) * 100
        ax.text(
            0.98, 0.97,
            f"n={len(sub):,}\nETI: {eti_rate:.3f}%",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color=MUTED_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR,
                      edgecolor=GRID_COLOR, alpha=0.85),
        )

    # Hide unused subplots
    for ax in axes.flat[n_groups:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def plot_prob_ccdf(
    df: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    """
    CCDF (1 - CDF) of probabilities — shows how many snippets exceed a
    given threshold, useful for choosing the operating point.
    """
    apply_light_style()

    probs = np.sort(df["probability"].values)
    ccdf  = 1.0 - np.arange(1, len(probs) + 1) / len(probs)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(probs, ccdf, color=ETI_COLOR, linewidth=2, label="CCDF  P(p ≥ x)")
    ax.fill_between(probs, ccdf, alpha=0.15, color=ETI_COLOR)

    # Mark the current threshold
    idx = np.searchsorted(probs, threshold)
    frac_above = ccdf[min(idx, len(ccdf) - 1)]
    ax.axvline(threshold, color=THRESH_COLOR, linewidth=1.5,
               linestyle="--", alpha=0.9)
    ax.axhline(frac_above, color=THRESH_COLOR, linewidth=1.0,
               linestyle=":", alpha=0.7)
    ax.scatter([threshold], [frac_above], color=THRESH_COLOR, s=60, zorder=6)
    ax.text(
        threshold + 0.015, frac_above + 0.015,
        f"({threshold:.2f}, {frac_above:.4f})\n"
        f"{frac_above*100:.3f}% of snippets above",
        color=THRESH_COLOR, fontsize=9,
        path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_COLOR)],
    )

    ax.set_yscale("log")
    ax.set_xlabel("P(ETI signal)", fontsize=12, labelpad=8)
    ax.set_ylabel("Fraction of snippets ≥ x  (log)", fontsize=12, labelpad=8)
    ax.set_title("RST — Complementary CDF of Classification Probabilities",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=10, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
