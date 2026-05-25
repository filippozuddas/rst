# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RST (Radio Spectrogram Transformer) is a binary classifier for technosignature detection in radio observations (SETI). It classifies 6-observation "cadences" (ON-OFF-ON-OFF-ON-OFF) as ETI signals vs. RFI. The model is DeiT (`base384`) fine-tuned on 1-channel radio spectrograms of shape `(96, 1024)`.

## Environment Setup

```bash
pip install -e .[train]
# or
conda env create -f environment.yml && conda activate rst
```

The package is installed in editable mode, so modules are importable as `from rst_seti.models.rst_model import RSTModel`.

## Common Commands

```bash
# Background extraction
rst-build-backgrounds --scan /path/to/h5/files --output data/training --name backgrounds

# Dataset generation
rst-build-dataset --backgrounds data/training/backgrounds_6GHz.npz --output data/processed --n-true 30000 --n-false 30000

# Training
rst-train --config configs/default.yaml --gpu 0

# Evaluation
rst-eval --model checkpoints/best_model.pth --data data/processed/val.npz --find-optimal --plot

# Inference (with auto-downloaded weights)
rst-infer -s /path/to/observations/

# Inference (with local weights)
rst-infer -m checkpoints/best.pth -f obs1_ON.h5 obs2_OFF.h5 ...
```

## Architecture

### Data flow

```
Raw HDF5 (blimpy) → (6, 16, n_freq) cadence array
  → extract_snippet() → (6, 16, 1024) crop
  → stack_cadence()   → (96, 1024) spectrogram
  → normalize_robust() → log10 + per-obs Z-score + clip[-5,5]
  → RSTModel forward  → unsqueeze + transpose → (1, 1024, 96)
  → PatchEmbed (16×16 conv) → 384 patches of 768-dim embeddings
  → 12 DeiT Transformer blocks
  → avg(CLS, DIST) tokens → MLP head → logit → sigmoid → P(ETI)
```

### Key design decisions

- **Input shape:** `(96, 1024)` — 6 observations × 16 time bins stacked vertically, 1024 raw frequency channels.
- **Patch stride 16×16 non-overlapping** → exactly 384 patches (6×64 grid after transpose).
- **RGB→1 channel adaptation:** ImageNet patch embedding weights are summed across the 3 channels (preserves pretrained features).
- **Positional embedding adaptation:** center-crop for dims ≤ 24, bilinear interpolation for larger dims.
- **Forward transpose:** input `(B, 96, 1024)` is transposed to `(B, 1, 1024, 96)` before the Conv2d so freq is the "height" axis.
- **Output:** raw logit; apply `torch.sigmoid()` at inference time. Threshold default: `0.9`.

### Inference sliding window

`InferenceEngine` slides a 1024-channel window over the full cadence frequency range with configurable step (default 512, 50% overlap), runs batch inference, then clusters adjacent detections into signals via connected-component grouping on the 1D grid.

### Training modes

- **`full`:** single LR, all parameters unfrozen from epoch 1 (AST-style). Works well when dataset is diverse and regularization is strong (FocalLoss + mixup + weight decay). Converges fast; use early stopping + WA around convergence.
- **`progressive`:** 3-phase unfreezing (head only → last 4 blocks → all). Preserves pretrained ImageNet features longer; tends to produce higher-confidence outputs and better generalization on out-of-distribution RFI. Preferred when background diversity is limited or domain gap is large.

### Loss functions

- `FocalLoss` (default, `focal_loss: true`): down-weights easy examples. Controlled by `focal_gamma` and `focal_alpha`.
- `LabelSmoothBCELoss`: alternative BCE with label smoothing.

## Configuration

All hyperparameters live in `configs/default.yaml`. Scripts accept `--config` to override. Key sections: `model`, `data`, `training`, `augmentation`, `inference`.

## Data Formats

- **Raw:** `.h5` HDF5 waterfall files processed by `blimpy`. Cadence = 6 files in ON/OFF alternating order.
- **Processed:** `.npz` with keys `spectrograms` (N, 96, 1024) float32 and `labels` (N,) binary.
- **Checkpoints:** `.pth` state dicts. `DataParallel` checkpoints have `module.` prefix — `InferenceEngine` strips it automatically.

## Module Map

| Path | Responsibility |
|---|---|
| `src/rst_seti/models/` | `RSTModel`, `PatchEmbed`, progressive unfreezing |
| `src/rst_seti/data/` | Preprocessing, augmentation, datasets, generators |
| `src/rst_seti/training/` | Training loop, loss functions, weight averaging |
| `src/rst_seti/inference/` | `InferenceEngine` — sliding window, clustering |
| `src/rst_seti/cli/` | CLI entry points (`rst-*` commands) |
| `src/rst_seti/hub.py` | Model weight management (Hugging Face Hub) |
| `src/rst_seti/utils/` | Plotting and visualization |
| `scripts/` | Legacy scripts (kept for backward compatibility) |
