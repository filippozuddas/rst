# RST — Radio Spectrogram Transformer

**RST (Radio Spectrogram Transformer)** is an end-to-end machine learning pipeline designed to assist Search for Extraterrestrial Intelligence (SETI) researchers in detecting anomalous signals (ETI) within radio spectrograms.

This project is an adaptation of the original [Audio Spectrogram Transformer (AST)](https://github.com/YuanGongND/ast) model to the radio astronomy domain. By leveraging attention mechanisms, RST is capable of identifying complex, narrow-band signals amidst various forms of Radio Frequency Interference (RFI) and background noise.

* **Original AST Paper:** [AST: Audio Spectrogram Transformer (Gong et al., 2021)](https://arxiv.org/abs/2104.01778)
* **Original AST Repository:** [YuanGongND/ast](https://github.com/YuanGongND/ast)

---

## 🎯 Target Audience & Vision

This repository is built for **SETI researchers** and **Machine Learning Engineers**. While the underlying model relies on deep learning architectures (Vision Transformers), the pipeline is designed to be accessible even for researchers without an extensive deep learning background.

**Future Vision:** The current scripts form the core pipeline. In the future, this project is intended to evolve into a fully scalable, easy-to-use tool with detailed documentation for every module.

---

## ⚙️ Installation & Requirements

RST relies on standard machine learning libraries (`torch`, `torchvision`, `timm`), radio astronomy data processing tools (`blimpy`), and synthetic data generation tools (`setigen`).

You can install the dependencies using either the provided Conda environment or standard `pip`.

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate rst
```

**Using Pip:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Overview & Quickstart

The RST pipeline is composed of four main stages: Dataset Generation, Model Training, Evaluation, and Inference. All main scripts are located in the `scripts/` directory.

### 1. Dataset Generation
Before training, you need to generate a training and validation dataset. The dataset builder takes extracted background plates (real observations) and injects synthetic ETI signals and RFI to create a robust dataset.

```bash
python scripts/build_dataset.py \
    --backgrounds data/extracted_backgrounds.npz \
    --output data/processed \
    --n-true 30000 \
    --n-false 30000
```
*Outputs: `train.npz`, `val.npz`, and generation metadata.*

### 2. Model Training
Train the RST model using the generated dataset. The training behaviour is controlled by a YAML configuration file (`configs/default.yaml`), which defines the model architecture, learning rate scheduling, batch size, and augmentations.

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --save_dir checkpoints \
    --gpu 0
```
*The model supports both full fine-tuning and progressive unfreezing modes (configurable in the YAML file).*

### 3. Evaluation
Evaluate the trained model on your validation or test set to compute metrics like AUC-ROC, F1-Score, Precision, and Recall. You can also run an automatic threshold sweep to find the optimal classification threshold.

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --model checkpoints/best_model.pth \
    --data data/processed/val.npz \
    --find-optimal \
    --plot \
    --output results/metrics.json
```

### 4. Inference
Run the trained model on real, unlabelled data (HDF5 cadences). The inference engine can process a single 6-file cadence (ON-OFF sequences) or scan entire directories. It automatically generates predictions, clusters them, and outputs visualizations (including attention maps showing what the model focused on).

**Process a single cadence:**
```bash
python scripts/infer.py \
    -m checkpoints/best_model.pth \
    -f obs1_ON.h5 obs2_OFF.h5 obs3_ON.h5 obs4_OFF.h5 obs5_ON.h5 obs6_OFF.h5 \
    -o results/
```

**Scan a directory:**
```bash
python scripts/infer.py \
    -m checkpoints/best_model.pth \
    -s /path/to/observations/ \
    -o results/ \
    --band all
```

---

## 📁 Repository Structure

* `configs/` - YAML configuration files for training and model parameters.
* `scripts/` - Executable scripts for the pipeline (build dataset, train, evaluate, infer).
* `src/` - Core Python packages:
  * `data/` - Dataset handling, synthetic signal generation, preprocessing, background extraction.
  * `models/` - The RST model architecture (based on `timm`).
  * `training/` - Training loops and optimizers.
  * `evaluation/` - Metrics and validation functions.
  * `inference/` - Inference engine and clustering logic.
  * `utils/` - Visualization and attention extraction tools.
* `notebooks/` - Jupyter notebooks for exploratory data analysis and visualizing processed data.

---

## 🤝 Contributing & Future Work

As mentioned, this project is the foundation for a larger, highly scalable SETI tool. Contributions regarding performance optimizations, new signal injection profiles (via `setigen`), or improved visualization capabilities are highly welcomed.
