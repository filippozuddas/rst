# RST — Radio Spectrogram Transformer

RST is a deep learning project for technosignature detection in radio observations (SETI). It uses a transformer-based architecture adapted from DeiT (Data-efficient Image Transformer) and AST (Audio Spectrogram Transformer) to classify radio spectrograms and identify potential extraterrestrial signals.

**RST is designed to be a scalable, production-ready tool fully integrated into the SETI Breakthrough Listen (BL) research pipeline.**

## Project Overview

- **Purpose:** Detect ET signals (ETI) and distinguish them from Radio Frequency Interference (RFI) in high-resolution radio spectrograms.
- **Architecture:** `RSTModel` based on `timm`'s DeiT backbone, modified for 1-channel input and non-square spectrograms.
- **Input:** "Cadences" of 6 observations (ON-OFF-ON-OFF-ON-OFF) represented as (96, 1024) spectrograms.
- **Main Technologies:** PyTorch, `timm`, `setigen` (signal injection), `blimpy` (HDF5/Waterfall handling), `numpy`, `scipy`.

## Directory Structure

```text
/
├── src/                # Core logic and modules
│   ├── models/         # RST architecture implementation
│   ├── data/           # Dataset loading, augmentation, and preprocessing
│   ├── training/       # Training loops and logic
│   ├── inference/      # End-to-end inference engine
│   └── utils/          # Visualization and helper functions
├── scripts/            # Entry points for various tasks (train, evaluate, infer)
├── configs/            # YAML configuration files
├── data/               # Raw and processed data (ignored by git)
├── checkpoints/        # Model weights saved during training
├── results/            # Inference results, plots, and attention maps
└── notebooks/          # Jupyter notebooks for data exploration
```

## Setup and Installation

The project uses Conda/Mamba for environment management.

```bash
# Create environment
conda env create -f environment.yml
conda activate rst

# Or use pip
pip install -r requirements.txt
```

## Usage

### 0. Data Preparation
Building a dataset involves two steps: extracting real backgrounds from telescope data and then generating synthetic training samples with signal injection.

#### A. Background Extraction
Scan directories for raw HDF5 files and extract 1024-channel background snippets.
```bash
python src/data/background_extractor.py --scan /path/to/h5/files --output data/training --name backgrounds
```

#### B. Dataset Building
Generate the final train/val datasets (`.npz`) by injecting synthetic ETI signals and RFI into the extracted backgrounds.
```bash
python scripts/build_dataset.py --backgrounds data/training/backgrounds_6GHz.npz --output data/processed --n-true 30000 --n-false 30000
```

### 1. Training
The project supports "full" and "progressive" training modes. Progressive mode unfreezes the model in 3 phases (head only, last 4 blocks, all).

```bash
python scripts/train.py --config configs/default.yaml --gpu 0
```

### 2. Evaluation
Evaluate a trained model on a processed `.npz` dataset.

```bash
python scripts/evaluate.py --config configs/default.yaml --model checkpoints/best_model.pth --data data/processed/val.npz --find-optimal --plot
```

### 3. Inference
Run end-to-end inference on raw HDF5 files.

```bash
# Single cadence (requires 6 files in ON/OFF order)
python scripts/infer.py -m checkpoints/best.pth -f obs1_ON.h5 obs2_OFF.h5 ...

# Directory scan
python scripts/infer.py -m checkpoints/best.pth -s /path/to/observations/
```

### 4. Visualization
Generate attention maps to see which parts of the spectrogram the model is focusing on.

```bash
python scripts/visualize_attention.py --model checkpoints/best.pth --input data/processed/sample.npz
```

## Development Conventions

- **Model Adaptation:** When loading ImageNet weights for 1-channel input, the weights are summed across the RGB channels to preserve the pretrained features.
- **Normalization:** Spectrograms are typically normalized using log-scaling and robust Z-score normalization per observation.
- **Augmentation:** SpecAugment (frequency and time masking) and Mixup are used during training.
- **Progressive Unfreezing:** Recommended for small datasets to prevent catastrophic forgetting of ImageNet features.
- **Configuration:** All hyperparameters should be managed via YAML files in `configs/`.

## Data Format

- **Raw Data:** HDF5 files (`.h5`) produced by telescope backends (processed via `blimpy`).
- **Processed Data:** `.npz` files containing `spectrograms` (N, 96, 1024) and `labels` (N,).

## References

- **AST Repository:** [YuanGongND/ast](https://github.com/YuanGongND/ast)
- **AST Paper:** [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Gong et al., 2021)
