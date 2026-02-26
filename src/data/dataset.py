# -*- coding: utf-8 -*-
"""
RST — PyTorch Dataset

Provides SETIDataset to load preprocessed spectrograms from .npz files,
apply SpecAugment/Mixup during training, and return (spectrogram, label) pairs.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

from .augmentation import spec_augment, mixup


class SETIDataset(Dataset):
    """
    PyTorch Dataset for SETI spectrograms.

    Args:
        data_path: Path to the .npz file with preprocessed data.
        is_training: If True, applies augmentation.
        norm_mean: Mean for normalization. If None, uses the value from the file.
        norm_std: Std for normalization. If None, uses the value from the file.
        freq_mask: SpecAugment param — max frequency channels to mask.
        time_mask: SpecAugment param — max time bins to mask.
        mixup_alpha: Mixup param — α of the Beta distribution. 0 = disabled.
    """

    def __init__(
        self,
        data_path: str,
        is_training: bool = True,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
        freq_mask: int = 32,
        time_mask: int = 8,
        mixup_alpha: float = 0.5,
    ):
        super().__init__()

        # Load the .npz file
        data = np.load(data_path, allow_pickle=True)
        self.spectrograms = data['spectrograms']  # (N, 96, 1024)
        self.labels = data['labels']               # (N,)

        # Normalization statistics
        # Priority: explicit params > values in file > safe defaults
        self.mean = norm_mean if norm_mean is not None else float(data.get('mean', 0.0))
        self.std = norm_std if norm_std is not None else float(data.get('std', 1.0))

        # Augmentation settings (active only during training)
        self.is_training = is_training
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.mixup_alpha = mixup_alpha

        print(f'Dataset loaded: {len(self)} samples '
              f'({"TRAIN" if is_training else "EVAL"}), '
              f'shape={self.spectrograms.shape}, '
              f'mean={self.mean:.4f}, std={self.std:.4f}')

    def __len__(self) -> int:
        """How many samples the dataset contains."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single sample (spectrogram, label).
        Loads, normalizes, and applies augmentation (Mixup, SpecAugment) if training.

        Args:
            index: Sample index (0 ≤ index < len(dataset)).

        Returns:
            Tuple of:
            - spectrogram: float32 Tensor of shape (96, 1024)
            - label: float32 Tensor of shape (1,)
        """
        # 1. Load and normalize
        spec = self.spectrograms[index].astype(np.float32)
        spec = (spec - self.mean) / (self.std * 2)
        label = float(self.labels[index])

        # Convert to PyTorch tensors
        spec = torch.from_numpy(spec)      # (96, 1024)
        label = torch.tensor([label])       # (1,)

        # 2. Mixup (training only, if alpha > 0)
        if self.is_training and self.mixup_alpha > 0:
            # Pick a random second sample
            mix_idx = np.random.randint(0, len(self))
            spec2 = self.spectrograms[mix_idx].astype(np.float32)
            spec2 = (spec2 - self.mean) / (self.std * 2)
            label2 = float(self.labels[mix_idx])

            spec2 = torch.from_numpy(spec2)
            label2 = torch.tensor([label2])

            spec, label = mixup(spec, label, spec2, label2, alpha=self.mixup_alpha)

        # 3. SpecAugment (training only)
        if self.is_training:
            if self.freq_mask > 0 or self.time_mask > 0:
                spec = spec_augment(
                    spec,
                    freq_mask_param=self.freq_mask,
                    time_mask_param=self.time_mask,
                )

        return spec, label


def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    norm_mean: Optional[float] = None,
    norm_std: Optional[float] = None,
    freq_mask: int = 32,
    time_mask: int = 8,
    mixup_alpha: float = 0.5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.

    Args:
        train_path: Path to the training set .npz file.
        val_path: Path to the validation set .npz file.
        batch_size: Samples per batch (default: 32).
        num_workers: Parallel processes for data loading.
        norm_mean/std: Normalization statistics.
        freq_mask/time_mask: SpecAugment parameters.
        mixup_alpha: Mixup parameter.

    Returns:
        Tuple (train_loader, val_loader).
    """
    train_dataset = SETIDataset(
        data_path=train_path,
        is_training=True,
        norm_mean=norm_mean,
        norm_std=norm_std,
        freq_mask=freq_mask,
        time_mask=time_mask,
        mixup_alpha=mixup_alpha,
    )

    val_dataset = SETIDataset(
        data_path=val_path,
        is_training=False,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle every epoch
        num_workers=num_workers,
        pin_memory=True,     # Speed up CPU → GPU transfer
        drop_last=True,      # Drop the last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle during validation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,     # Evaluate all samples
    )

    return train_loader, val_loader
