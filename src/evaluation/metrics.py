# -*- coding: utf-8 -*-
"""
RST — Evaluation Metrics

Computes metrics for binary classification ETI vs RFI:
- AUC-ROC: Area Under the ROC Curve
- AUC-PR: Area Under the Precision-Recall Curve
- F1-Score, Precision, Recall
- Confusion Matrix
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all binary classification metrics.

    Args:
        labels: Ground truth, array of 0s and 1s, shape (N,).
        probs: Predicted probabilities (sigmoid output), shape (N,).
        threshold: Threshold to binarize probabilities (default 0.5).

    Returns:
        Dictionary with all metrics.
    """
    # Binarize predictions
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    metrics = {}

    # Accuracy
    metrics['accuracy'] = float(np.mean(preds == labels_int))

    # AUC-ROC
    try:
        metrics['auc_roc'] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics['auc_roc'] = 0.0  # Only one class present

    # AUC-PR (Average Precision)
    try:
        metrics['auc_pr'] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics['auc_pr'] = 0.0

    # F1, Precision, Recall
    metrics['f1'] = float(f1_score(labels_int, preds, zero_division=0))
    metrics['precision'] = float(precision_score(labels_int, preds, zero_division=0))
    metrics['recall'] = float(recall_score(labels_int, preds, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels_int, preds)
    if cm.shape == (2, 2):
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])

    return metrics


def get_curves(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute ROC and Precision-Recall curves for visualization.

    Returns:
        Dictionary with:
        - 'roc': (fpr, tpr, thresholds)
        - 'pr': (precision, recall, thresholds)
    """
    fpr, tpr, roc_thresh = roc_curve(labels, probs)
    prec, rec, pr_thresh = precision_recall_curve(labels, probs)

    return {
        'roc': (fpr, tpr, roc_thresh),
        'pr': (prec, rec, pr_thresh),
    }


def print_report(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Print a complete evaluation report."""
    metrics = compute_metrics(labels, probs, threshold)
    preds = (probs >= threshold).astype(int)

    print('\n' + '=' * 50)
    print('  RST — Evaluation Report')
    print('=' * 50)
    print(f'  Accuracy:   {metrics["accuracy"]:.4f}')
    print(f'  AUC-ROC:    {metrics["auc_roc"]:.4f}')
    print(f'  AUC-PR:     {metrics["auc_pr"]:.4f}')
    print(f'  F1-Score:   {metrics["f1"]:.4f}')
    print(f'  Precision:  {metrics["precision"]:.4f}')
    print(f'  Recall:     {metrics["recall"]:.4f}')
    print(f'  Threshold:  {threshold}')
    print()
    print(classification_report(
        labels.astype(int), preds,
        target_names=['RFI/Noise', 'ETI'],
        zero_division=0,
    ))
