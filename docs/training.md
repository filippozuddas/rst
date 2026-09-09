# Training

```bash
rst-train --config configs/default.yaml --gpu 0
rst-train --config configs/default.yaml --gpu 0,1 --mode progressive
rst-train --config configs/default.yaml --gpu 0 --resume checkpoints/epoch_010.pth
```

`rst-train` is a thin wrapper around `rst_seti.training.trainer.train()`. It builds an `RSTModel` from `config['model']`, optionally wraps it in `DataParallel` for multi-GPU, builds dataloaders from `config['data']` (`train_data`/`val_data`) and `config['augmentation']`, then calls `train()`.

## Loss functions

Selected by `config['training']`:

- **`FocalLoss(gamma, alpha)`** (used when `focal_loss: true`, the default) — `FL(p) = -α_t (1-p_t)^γ · BCE`, down-weighting easy examples so the model focuses on hard true/hard-false cases. `focal_gamma` controls how aggressively easy examples are down-weighted (`configs/default.yaml`: `2.0`, the standard value from the focal-loss paper); `focal_alpha` balances false positives vs. false negatives (the `FocalLoss` class itself defaults to `0.75`, but `configs/default.yaml` sets `0.5` — equal FP/FN weighting).
- **`LabelSmoothBCELoss(smoothing)`** (used when `focal_loss: false` and `label_smoothing > 0`) — remaps hard targets `{0, 1} → {ε/2, 1-ε/2}` before BCE, preventing logits from saturating to ±∞.
- **Plain `BCEWithLogitsLoss`** — fallback if neither of the above is enabled.

`focal_loss: true` always takes priority over `label_smoothing`.

## Training modes

`config['training']['mode']` (`full` | `progressive`, overridable via `rst-train --mode`) determines the **phase list**:

- **`full`** — a single synthetic phase: `{name: 'full_finetune', layers: 'all', lr: config['lr'], epochs: config['epochs']}`. All parameters are trainable from epoch 1 (AST-style). Relies on FocalLoss + mixup + weight decay + weight averaging for regularization; converges fast.
- **`progressive`** — uses `config['training']['phases']` directly (see [Current configuration](#current-configuration) below): `head_only` → `last_4_blocks` → `full_finetune`. Preserves pretrained ImageNet features longer; preferred when background diversity is limited or the domain gap (real SRT vs. synthetic injection) is large.

For each phase, `train()`:

1. Calls one of `actual_model.freeze_backbone()` (`layers: "head"`), `unfreeze_last_n_blocks(4)` (`layers: "last_4_blocks"`), or `unfreeze_all()` (`layers: "all"`) on the underlying model (unwrapped from `DataParallel` via `.module` if present).
2. **Recreates the optimizer** — `AdamW(filter(lambda p: p.requires_grad, ...), lr=phase['lr'], weight_decay=1e-4, betas=(0.95, 0.999))` — since the trainable parameter set changes between phases. `betas=(0.95, 0.999)` matches the AST paper.
3. Builds a fresh LR scheduler for the phase (see below).
4. Runs `phase['epochs']` epochs of train/validate, subject to early stopping.

`global_epoch` is a running counter across all phases — checkpoint filenames (`epoch_NNN.pth`) and `history['phase']` use it, so phase boundaries are visible in `history.npz`/`training_log.json` but checkpoint numbering is continuous.

## LR schedulers

`scheduler_name = phase.get('scheduler', config.get('scheduler', 'warmup_cosine'))` — i.e. each phase can override the global `training.scheduler`. Supported values:

| Scheduler | Behavior |
| --- | --- |
| `plateau` | `ReduceLROnPlateau(mode='min', ...)` on `val_loss`. Uses `plateau_patience`/`plateau_factor`/`eta_min`. |
| `plateau_f1` | Same, but `mode='max'` on `val_f1`. |
| `warmup_cosine` (default) | `LinearLR` warmup (`warmup_start_factor` → 1.0 over `warmup_epochs`) → `CosineAnnealingLR(T_max=epochs - warmup_epochs, eta_min)`, chained via `SequentialLR`. `warmup_epochs` is read per-phase (falls back to global `config['warmup_epochs']`) and clamped to `min(warmup_epochs, epochs - 1)` so short phases never produce a negative `T_max`. |
| `onecycle` | `OneCycleLR(max_lr=config['max_lr'], steps_per_epoch=len(train_loader), epochs, pct_start=0.3, anneal_strategy='cos')`, stepped **per batch** (passed into `_train_one_epoch` as `batch_scheduler`). |
| anything else (`cosine`, fallback) | `CosineAnnealingLR(T_max=epochs, eta_min)`. |

All schedulers except `onecycle` are stepped once per epoch (`scheduler.step()`, or `scheduler.step(val_loss)`/`scheduler.step(val_f1)` for the plateau variants).

## Mixed precision & gradient clipping

Each training step runs the forward pass under `torch.amp.autocast(device.type)` and uses `GradScaler` for the backward pass. Gradients are unscaled (`scaler.unscale_`) and clipped to `config['gradient_clip']` (default `1.0`) via `torch.nn.utils.clip_grad_norm_` over only the trainable parameters, before `scaler.step(optimizer)`.

## Validation metrics

After each epoch, `_validate()` computes, over the full validation set:

- `val_loss` — same loss function as training.
- `val_accuracy` — `sigmoid(logit) >= 0.5` vs. label.
- `val_auc` — `sklearn.metrics.roc_auc_score` (falls back to `0.0` if only one class is present or `sklearn` is unavailable).
- `val_f1` — `sklearn.metrics.f1_score` at the `0.5` threshold (note: this is a fixed `0.5` threshold for model-selection purposes, independent of the `inference.threshold = 0.9` used at deployment time — see [docs/inference.md](inference.md)).

## Checkpointing & early stopping

- **`best_model.pth`** — overwritten whenever `val_f1` improves (strictly greater than the previous best).
- **`epoch_NNN.pth`** — saved every epoch (`global_epoch`, zero-padded to 3 digits), to support weight averaging.
- **Early stopping** — `patience = config['early_stopping_patience']` (default `6`; `0` disables it). `epochs_no_improve` increments whenever `val_f1` doesn't improve and resets to 0 on improvement; **it is reset at the start of every phase**, so each phase in progressive mode gets its own early-stopping budget. When `epochs_no_improve >= patience`, training stops immediately (breaking out of all remaining phases too).

## Weight averaging

If `config['weight_averaging']` is true (default), `weight_average()` runs once at the end of training:

```python
start_epoch = max(1, best_val_epoch - wa_n_before)   # wa_n_before defaults to 4
epochs_to_avg = range(start_epoch, best_val_epoch + 1)
```

It loads `epoch_{ep:03d}.pth` for each epoch in that (inclusive) range, averages every parameter tensor element-wise, casts back to the original dtype, and writes the result to `model_wa.pth`. Averaging only epochs **up to and including** the best epoch (never after) avoids blending in post-convergence overfit weights — this was a historical bug (averaging the *last* N epochs regardless of where the best epoch fell) that produced significantly more false positives than averaging *around the best epoch*.

## Augmentation (applied during training only)

Configured under `config['augmentation']`, applied inside the dataloader (`rst_seti.data.dataset.create_dataloaders` / `rst_seti.data.augmentation`):

- **SpecAugment** (`spec_augment`) — randomly masks up to `freq_mask` contiguous frequency rows and `time_mask` contiguous time rows per sample (set to `0.0`).
- **Mixup** (`mixup`) — blends pairs of `(spectrogram, label)` with `λ ~ Beta(α, α)` (`mixup_alpha`), with `λ = max(λ, 1-λ)` so the dominant sample is always weighted ≥ 0.5.
- **`drop_path`** — stochastic depth rate passed to the DeiT backbone (`config['augmentation']['drop_path']`).

## Current configuration

`configs/default.yaml`, `training`/`augmentation` sections:

| Key | Value | Notes |
| --- | --- | --- |
| `mode` | `full` | Single-phase, all params trainable from epoch 1 |
| `lr` / `epochs` | `8.0e-5` / `25` | Used only in `full` mode |
| `batch_size` | `256` | |
| `mixed_precision` | `true` | |
| `gradient_clip` | `1.0` | |
| `weight_averaging` / `wa_n_before` | `true` / `4` | Averages epochs `[best-4, best]` |
| `early_stopping_patience` | `6` | `0` disables |
| `scheduler` / `eta_min` | `warmup_cosine` / `1.0e-7` | |
| `warmup_epochs` / `warmup_start_factor` | `3` / `0.1` | |
| `plateau_patience` / `plateau_factor` | `5` / `0.5` | Used by `plateau`/`plateau_f1` only |
| `max_lr` | `5.0e-5` | Used by `onecycle` only |
| `focal_loss` | `true` | |
| `focal_gamma` / `focal_alpha` | `2.0` / `0.5` | `0.5` = equal FP/FN weighting |
| `label_smoothing` | `0.0` | Inactive while `focal_loss: true` |
| `freq_mask` / `time_mask` | `32` / `8` | SpecAugment mask sizes (rows) |
| `mixup_alpha` | `0.1` | |
| `drop_path` | `0.20` | |

The `progressive` mode's 3-phase schedule (`head_only` → `last_4_blocks` → `full_finetune`) is defined under `training.phases` in the same file, with per-phase `lr`/`epochs`/`scheduler`/`warmup_epochs` overrides.

## Outputs (`save_dir`, default `checkpoints/`)

- `best_model.pth`, `model_wa.pth`, `epoch_NNN.pth`
- `history.npz` — arrays for `train_loss`, `val_loss`, `val_accuracy`, `val_auc`, `val_f1`, `lr`, `phase` (one entry per epoch).
- `training_log.json` — timestamp, duration, full config, `best_epoch`, `best_val_f1`, `best_val_loss`, `final_epoch`, and the same per-epoch history rounded for readability.

## Evaluation

```bash
rst-eval --model checkpoints/best_model.pth --data data/processed/val.npz --find-optimal --plot
```

`rst-eval` delegates to `scripts/evaluate.py` (resolved relative to the repo root), so it requires a full editable checkout — not just `pip install rst-seti`. `--find-optimal` sweeps the classification threshold to find the value maximizing F1 on the given dataset (useful for comparing against the deployed `inference.threshold = 0.9`); `--plot` generates ROC/PR/confusion-matrix plots under `--output` (default `results/eval`).
