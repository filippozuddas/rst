# RST — Radio Spectrogram Transformer

RST is a deep learning project for technosignature detection in radio observations (SETI). It uses a transformer-based architecture adapted from DeiT (Data-efficient Image Transformer) and AST (Audio Spectrogram Transformer) to classify radio spectrograms and identify potential extraterrestrial (ETI) signals.

**RST is built around real cadences from the Sardinia Radio Telescope (SRT, C-band) with synthetic ETI signal injection, and is telescope-agnostic enough to ingest any GUPPI/filterbank-format ON/OFF cadence (SRT, GBT, ...).**

## Project Overview

- **Purpose:** Detect ETI signals and distinguish them from Radio Frequency Interference (RFI) in high-resolution radio spectrograms.
- **Architecture:** `RSTModel`, a `timm` DeiT-base/384 backbone adapted for 1-channel, non-square spectrogram input.
- **Input:** "Cadences" of 6 observations (ON-OFF-ON-OFF-ON-OFF), preprocessed into `(96, 1024)` spectrograms.
- **Output:** A single logit → `sigmoid` → P(ETI), thresholded (default `0.9`) to classify ETI vs RFI.
- **Main technologies:** PyTorch, `timm`, `setigen` (signal injection), `blimpy`/`h5py` (HDF5/Waterfall handling), `huggingface_hub` (model distribution).

## Documentation

This README covers setup and the high-level pipeline. For details on individual stages, see:

- **[docs/data_pipeline.md](docs/data_pipeline.md)** — background extraction, synthetic signal realism (drift, profiles, scintillation, RFI taxonomy), and dataset construction.
- **[docs/training.md](docs/training.md)** — training modes, loss functions, schedulers, weight averaging, early stopping.
- **[docs/inference.md](docs/inference.md)** — sliding-window inference, clustering, CLI usage, and output layout.

## Directory Structure

```text
rst/
├── src/rst_seti/        # Installable package (pip install -e .[train])
│   ├── cli/             # rst-* command entry points
│   ├── models/          # RSTModel, PatchEmbed
│   ├── data/            # background extraction, signal injection, datasets, augmentation, preprocessing
│   ├── training/        # training loop, loss functions, weight averaging
│   ├── inference/        # InferenceEngine (sliding window + clustering)
│   ├── evaluation/        # evaluation metrics
│   ├── utils/              # plotting / attention visualization
│   ├── configs/             # bundled default.yaml (package data)
│   ├── hub.py                # Hugging Face Hub model download/cache
│   └── hardware.py            # device & batch-size auto-detection
├── scripts/              # legacy standalone scripts, kept in sync with src/rst_seti
├── configs/               # development configuration(s)
├── docs/                   # pipeline documentation
├── data/                    # raw/processed datasets (ignored by git)
├── checkpoints/              # model weights saved during training
├── results/                    # inference outputs (CSVs, plots, attention maps)
└── notebooks/                   # exploration notebooks
```

## Setup and Installation

The package is installed in editable mode, so modules are importable as `from rst_seti.models.rst_model import RSTModel` and the `rst-*` CLI commands become available on `PATH`.

```bash
# Editable install with the full training/dataset-generation pipeline
pip install -e .[train]
```

```bash
# Or via conda/mamba
conda env create -f environment.yml
conda activate rst
```

`.[train]` adds `setigen` (signal injection) and `scikit-learn` (evaluation metrics). Omit it for an inference-only install.

## Quick Start

The first `rst-infer` run auto-downloads pretrained weights (`rst-base384-v1`, ~350 MB) from the [Hugging Face Hub](https://huggingface.co/filippozuddas/rst-seti) and caches them in `~/.cache/rst-seti/models/`:

```bash
rst-infer -s /path/to/observations/
```

## Usage

### 1. Data Preparation

Building a dataset is a two-step pipeline: extract real backgrounds from telescope data, then inject synthetic ETI/RFI signals.

```bash
# A. Extract 1024-channel background snippets from raw HDF5 cadences
rst-build-backgrounds --scan /path/to/h5/files --output data/training --name backgrounds --band 6GHz

# B. Inject synthetic signals and build train/val .npz datasets
rst-build-dataset --backgrounds data/training/backgrounds_6GHz.npz --output data/processed --n-true 30000 --n-false 30000
```

`rst-build-backgrounds` also supports `--band mixed` (frequency-balanced multi-band extraction with a `--train-fraction` train/inference split), `--exclude-targets` (hold out specific targets for inference), and `--seed`. Held-out cadences are written to an `inference_cadences_*.txt` manifest consumable directly by `rst-infer -i`. See **[docs/data_pipeline.md](docs/data_pipeline.md)** for the full signal-realism model (drift distributions, frequency/time profiles, RFI types) and dataset schema.

### 2. Training

```bash
rst-train --config configs/default.yaml --gpu 0
rst-train --config configs/default.yaml --gpu 0,1 --mode progressive
```

RST supports two training modes, configured in `configs/default.yaml`:

- **`full`** (default): single LR, all parameters unfrozen from epoch 1 (AST-style).
- **`progressive`**: 3-phase unfreezing (head only → last 4 blocks → all).

Both use FocalLoss, mixup, SpecAugment, weight averaging around convergence, and early stopping on validation F1. See **[docs/training.md](docs/training.md)** for details and current hyperparameters.

### 3. Evaluation

```bash
rst-eval --model checkpoints/best_model.pth --data data/processed/val.npz --find-optimal --plot
```

`rst-eval` delegates to `scripts/evaluate.py`, so it requires a full editable checkout of this repository (not a standalone `pip install rst-seti`).

### 4. Inference

```bash
# Auto-download weights, scan a directory of HDF5 cadences
rst-infer -s /path/to/observations/

# Local weights, single cadence (6 files in ON/OFF order)
rst-infer -m checkpoints/best.pth -f ON1.h5 OFF1.h5 ON2.h5 OFF2.h5 ON3.h5 OFF3.h5

# Process an inference-cadences manifest produced by rst-build-backgrounds
rst-infer -i data/training/inference_cadences_mixed.txt
```

`rst-infer` slides a 1024-channel window across the full cadence, runs batch inference, clusters adjacent detections into signals, and writes per-cadence CSVs, plots, and attention maps under `results/`. See **[docs/inference.md](docs/inference.md)** for the sliding-window/clustering algorithm, output directory naming, and file formats.

## Architecture

```text
Raw HDF5 (blimpy/h5py)  →  (6, 16, n_freq) cadence array
  → extract_snippet()   →  (6, 16, 1024) crop
  → stack_cadence()     →  (96, 1024) spectrogram
  → normalize_robust()  →  log10 + per-observation Z-score + clip[-5, 5]
  → RSTModel.forward()  →  unsqueeze + transpose → (1, 1, 1024, 96)
  → PatchEmbed (16×16 conv, stride 16) → 384 patches × 768-dim embeddings
  → CLS + DIST tokens + positional embedding → 12 DeiT Transformer blocks
  → avg(CLS, DIST) → LayerNorm → Linear(768, 1) → raw logit → sigmoid → P(ETI)
```

**Key design decisions:**

- **Input shape `(96, 1024)`:** 6 observations × 16 time bins stacked vertically, 1024 raw frequency channels (no binning).
- **Non-overlapping 16×16 patches** → exactly 384 patches (64×6 grid after the freq/time transpose).
- **RGB → 1-channel adaptation:** ImageNet patch-embedding weights are summed across the 3 input channels to preserve pretrained features.
- **Positional embedding adaptation:** center-crop for grid dims ≤ 24 (the original 24×24 DeiT-384 grid), bilinear interpolation for larger dims.
- **Progressive unfreezing:** `freeze_backbone()` / `unfreeze_last_n_blocks(n)` / `unfreeze_all()` support the 3-phase training mode.

## Module Map

| Path | Responsibility |
| --- | --- |
| `src/rst_seti/models/` | `RSTModel`, `PatchEmbed`, progressive unfreezing |
| `src/rst_seti/data/` | Background extraction, signal injection, preprocessing, augmentation, datasets |
| `src/rst_seti/training/` | Training loop, loss functions, weight averaging |
| `src/rst_seti/inference/` | `InferenceEngine` — sliding window, clustering |
| `src/rst_seti/evaluation/` | Evaluation metrics |
| `src/rst_seti/cli/` | CLI entry points (`rst-*` commands) |
| `src/rst_seti/hub.py` | Model weight management (Hugging Face Hub) |
| `src/rst_seti/hardware.py` | Device/dtype/batch-size auto-detection |
| `src/rst_seti/utils/` | Plotting and attention-map visualization |
| `scripts/` | Legacy standalone scripts (kept in sync with `src/rst_seti`) |

## Data Formats

- **Raw:** `.h5`/`.hdf5` HDF5 waterfall files (GUPPI/filterbank format), read via `h5py`/`blimpy`. A cadence is 6 files in ON/OFF/ON/OFF/ON/OFF order.
- **Processed:** `.npz` with keys `spectrograms` (N, 96, 1024) float32 and `labels` (N,) binary, plus a `generation_metadata.json` describing the signal-injection parameters used.
- **Backgrounds:** `.npz` with key `backgrounds` (N, 6, 16, 1024) float32 — raw, un-normalized real observation snippets, output of `rst-build-backgrounds`.
- **Checkpoints:** `.pth` state dicts. `DataParallel` checkpoints carry a `module.` prefix, stripped automatically by `InferenceEngine`/`rst-train`.

## References

- **AST Repository:** [YuanGongND/ast](https://github.com/YuanGongND/ast)
- **AST Paper:** [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Gong et al., 2021)
- **setigen:** [UCBerkeleySETI/setigen](https://github.com/UCBerkeleySETI/setigen) — synthetic signal injection
- **Pretrained weights:** [filippozuddas/rst-seti](https://huggingface.co/filippozuddas/rst-seti) on Hugging Face Hub
