# -*- coding: utf-8 -*-
"""
RST — Trainer

PyTorch training loop supporting two modes:
- Full fine-tune (AST-style): single LR, all parameters unfrozen
- Progressive Unfreezing: 3-phase training with gradual unfreezing

Both modes include:
- Mixed Precision (FP16) to save VRAM
- Gradient Clipping for stability
- Checkpoint saving and Weight Averaging
- Metrics logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

# Mixed precision:
# autocast() runs computations in FP16 (half memory)
# GradScaler() prevents gradients from "vanishing" with FP16
from torch.amp import autocast, GradScaler


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Focuses training on hard examples by down-weighting well-classified ones.
    
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    save_dir: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Train the RST model.
    Supports "full" (single-phase) or "progressive" (multi-phase) unfreezing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode = config.get('mode', 'full')
    gradient_clip = config.get('gradient_clip', 1.0)

    # Loss function configuration
    use_focal = config.get('focal_loss', False)
    if use_focal:
        gamma = config.get('focal_gamma', 2.0)
        alpha = config.get('focal_alpha', 0.75)
        loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
        print(f"Loss Function: FocalLoss (gamma={gamma}, alpha={alpha})")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Loss Function: BCEWithLogitsLoss")

    # Mixed precision scaler
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_f1': [],
        'lr': [],
        'phase': [],
    }

    # Initialize best Val F1 to 0
    best_val_f1 = -1.0
    global_epoch = 0

    # Build the list of phases depending on mode
    if mode == 'full':
        phases = [{
            'name': 'full_finetune',
            'layers': 'all',
            'lr': config['lr'],
            'epochs': config['epochs'],
        }]
    elif mode == 'progressive':
        phases = config['phases']
    else:
        raise ValueError(f"Unknown training mode: '{mode}'. Use 'full' or 'progressive'.")

    print(f'\nTraining mode: {mode.upper()}')
    print(f'Total phases: {len(phases)}')

    # ====================================================================== #
    #  MAIN LOOP: iterate over training phases
    # ====================================================================== #
    for phase_idx, phase in enumerate(phases):
        phase_name = phase['name']
        lr = phase['lr']
        epochs = phase['epochs']
        layers = phase['layers']

        print(f'\n{"="*60}')
        print(f'  PHASE {phase_idx + 1}/{len(phases)}: {phase_name}')
        print(f'  LR: {lr}, Epochs: {epochs}, Layers: {layers}')
        print(f'{"="*60}')

        # Freeze/unfreeze the appropriate layers
        if layers == 'head':
            model.freeze_backbone()
        elif layers == 'last_4_blocks':
            model.unfreeze_last_n_blocks(4)
        elif layers == 'all':
            model.unfreeze_all()
        else:
            raise ValueError(f'Unknown layers: {layers}')

        trainable = model.get_trainable_params_count()
        total = sum(p.numel() for p in model.parameters())
        print(f'  Trainable params: {trainable:,} / {total:,} '
              f'({100 * trainable / total:.1f}%)')

        # Create the optimizer for this phase
        # Recreated per phase because trainable parameters may change.
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4,       # Increased regularization (was 5e-7)
            betas=(0.95, 0.999),     # Same as the AST paper
        )

        # CosineAnnealingLR: smoothly decreases LR following a cosine curve
        min_lr = config.get('eta_min', 1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
        print(f'  Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={min_lr})')

        # ============= EPOCH LOOP ============= #
        for epoch in range(epochs):
            global_epoch += 1
            current_lr = optimizer.param_groups[0]['lr']

            # --- Training ---
            train_loss = _train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler,
                gradient_clip, device,
            )

            # --- Validation ---
            val_loss, val_acc, val_auc, val_f1 = _validate(
                model, val_loader, loss_fn, device,
            )

            # Step the scheduler (Cosine doesn't need val_loss)
            scheduler.step()

            # Save to history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_auc'].append(val_auc)
            history['val_f1'].append(val_f1)
            history['lr'].append(current_lr)
            history['phase'].append(phase_name)

            print(f'  Epoch {global_epoch:3d} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'Val AUC: {val_auc:.4f} | '
                  f'Val F1: {val_f1:.4f} | '
                  f'LR: {current_lr:.2e}')

            # Save best model based on F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                print(f'  → New best model saved! (val_f1={val_f1:.4f})')

            # Save checkpoint every epoch (for weight averaging)
            torch.save(
                model.state_dict(),
                save_dir / f'epoch_{global_epoch:03d}.pth',
            )

    # ====================================================================== #
    #  Weight Averaging: average the weights of the last N checkpoints
    # ====================================================================== #
    if config.get('weight_averaging', True):
        wa_model = weight_average(model, save_dir, n_last=5)
        if wa_model is not None:
            torch.save(wa_model, save_dir / 'model_wa.pth')
            print(f'\nWeight averaging saved (last 5 checkpoints)')

    # Save training history
    np.savez(save_dir / 'history.npz', **history)

    return history


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    gradient_clip: float,
    device: torch.device,
) -> float:
    """
    Run ONE training epoch.
    Returns the average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for specs, labels in train_loader:
        specs = specs.to(device)    # (batch, 96, 1024)
        labels = labels.to(device)  # (batch, 1)

        # Forward pass in mixed precision
        with autocast('cuda'):
            output = model(specs)
            loss = loss_fn(output, labels)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            gradient_clip,
        )

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the validation set.

    @torch.no_grad() disables gradient computation →
    uses less memory and runs faster (not needed during evaluation).

    Returns:
        Tuple (val_loss, accuracy, auc_roc, f1_score).
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for specs, labels in val_loader:
        specs = specs.to(device)
        labels = labels.to(device)

        with autocast('cuda'):
            output = model(specs)
            loss = loss_fn(output, labels)

        # Convert logits → probabilities with sigmoid
        probs = torch.sigmoid(output)

        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # Compute metrics
    avg_loss = total_loss / max(n_batches, 1)

    # Accuracy: threshold at 0.5
    predictions = (all_preds >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == all_labels.astype(int)))

    # AUC-ROC and F1
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        auc = float(roc_auc_score(all_labels, all_preds))
        f1 = float(f1_score(all_labels.astype(int), predictions, zero_division=0))
    except (ImportError, ValueError):
        auc = 0.0  # If only one class is present, AUC is undefined
        f1 = 0.0

    return avg_loss, accuracy, auc, f1


def weight_average(
    model: nn.Module,
    checkpoint_dir: Path,
    n_last: int = 5,
) -> Optional[Dict]:
    """
    Average the weights of the last N checkpoints to improve generalization.
    Returns averaged state dict, or None if not enough checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('epoch_*.pth'))

    if len(checkpoints) < n_last:
        print(f'Only {len(checkpoints)} checkpoints, need at least {n_last} for WA')
        return None

    # Take the last n_last
    checkpoints = checkpoints[-n_last:]

    # Load the first as base
    avg_state = torch.load(checkpoints[0], map_location='cpu')

    # Sum all the others
    for ckpt_path in checkpoints[1:]:
        state = torch.load(ckpt_path, map_location='cpu')
        for key in avg_state:
            avg_state[key] = avg_state[key] + state[key]

    # Divide by the number of checkpoints
    for key in avg_state:
        avg_state[key] = avg_state[key] / float(n_last)

    return avg_state
