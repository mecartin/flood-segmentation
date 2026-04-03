"""
Evaluation metrics for flood segmentation.
Implements IoU, Dice, Precision, Recall, F1, and a MetricTracker helper.
"""

import torch
import numpy as np
from typing import Dict, Tuple



def threshold_predictions(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (pred >= threshold).float()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard Index)."""
    pred_bin = threshold_predictions(pred, threshold)
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Dice coefficient (F1 score over pixels)."""
    pred_bin = threshold_predictions(pred, threshold)
    target = target.float()
    intersection = (pred_bin * target).sum()
    return ((2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


def compute_precision_recall_f1(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1-score purely on GPU to prevent memory leaks."""
    pred_bin = threshold_predictions(pred, threshold)
    target_bin = target.float()
    
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1.0 - target_bin)).sum()
    fn = ((1.0 - pred_bin) * target_bin).sum()
    
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2.0 * (precision * recall) / (precision + recall + eps)
    
    return precision.item(), recall.item(), f1.item()


class MetricTracker:
    """Accumulates per-batch metrics and computes epoch-level averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums: Dict[str, float] = {k: 0.0 for k in ['iou', 'dice', 'precision', 'recall', 'f1']}
        self._count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        self._sums['iou'] += compute_iou(pred, target, threshold)
        self._sums['dice'] += compute_dice(pred, target, threshold)
        p, r, f = compute_precision_recall_f1(pred, target, threshold)
        self._sums['precision'] += p
        self._sums['recall'] += r
        self._sums['f1'] += f
        self._count += 1

    def averages(self) -> Dict[str, float]:
        n = max(self._count, 1)
        return {k: v / n for k, v in self._sums.items()}
