"""
Visualisation and plotting utilities for flood segmentation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from typing import Dict, List, Optional


def _img_to_display(img: np.ndarray) -> np.ndarray:
    """Convert a [C, H, W] image array to a displayable [H, W] or [H, W, 3] array."""
    if img.shape[0] == 1:
        return img[0]
    if img.shape[0] == 2:
        # Sentinel-1: display VV channel as grayscale
        return img[0]
    # 3+ channels: use first 3 as pseudo-RGB
    rgb = img[:3].transpose(1, 2, 0)
    return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)


def visualize_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    title: str = "Flood Segmentation Predictions",
) -> plt.Figure:
    """
    Side-by-side visualisation: satellite image | ground truth | prediction | overlay.

    Args:
        images    : [B, C, H, W] tensor
        masks     : [B, 1, H, W] tensor
        preds     : [B, 1, H, W] tensor (sigmoid probabilities in [0, 1]).
                    Since models now output raw logits, callers must apply
                    torch.sigmoid(model_output) before passing here.
        n_samples : number of samples to visualise
        save_path : if provided, save figure to this path
        title     : figure super-title
    """
    n = min(n_samples, images.size(0))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ['Satellite Image (VV)', 'Ground Truth Mask', 'Prediction (prob)', 'Overlay']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=12, fontweight='bold', pad=8)

    for i in range(n):
        img_np  = images[i].cpu().numpy()
        mask_np = masks[i, 0].cpu().numpy()
        pred_np = preds[i, 0].cpu().numpy()
        pred_bin = (pred_np >= 0.5).astype(float)

        display = _img_to_display(img_np)
        cmap    = 'gray' if display.ndim == 2 else None

        axes[i][0].imshow(display, cmap=cmap, interpolation='bilinear')
        axes[i][0].axis('off')

        axes[i][1].imshow(mask_np, cmap='Blues', vmin=0, vmax=1)
        axes[i][1].axis('off')

        axes[i][2].imshow(pred_np, cmap='Blues', vmin=0, vmax=1)
        axes[i][2].axis('off')

        # Overlay: green = TP, red = FP, orange = FN
        if display.ndim == 2:
            bg = np.stack([display] * 3, axis=-1)
        else:
            bg = display
        bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        overlay = bg.copy()
        tp = (pred_bin == 1) & (mask_np == 1)
        fp = (pred_bin == 1) & (mask_np == 0)
        fn = (pred_bin == 0) & (mask_np == 1)
        overlay[tp] = [0.0, 0.9, 0.0]   # green  — correct flood
        overlay[fp] = [0.9, 0.2, 0.2]   # red    — false alarm
        overlay[fn] = [1.0, 0.6, 0.0]   # orange — missed flood

        axes[i][3].imshow(overlay)
        axes[i][3].axis('off')

    patches = [
        mpatches.Patch(color='#00e600', label='True Positive'),
        mpatches.Patch(color='#e63300', label='False Positive'),
        mpatches.Patch(color='#ff9900', label='False Negative'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_training_history(
    history: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
    title: str = "Training History",
) -> plt.Figure:
    """Plot loss and IoU curves for train and val splits."""
    train_loss = [m['loss'] for m in history['train']]
    val_loss   = [m['loss'] for m in history['val']]
    train_iou  = [m['iou']  for m in history['train']]
    val_iou    = [m['iou']  for m in history['val']]
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label='Train Loss', color='#e74c3c', lw=2)
    ax1.plot(epochs, val_loss,   label='Val Loss',   color='#3498db', lw=2, ls='--')
    ax1.set(xlabel='Epoch', ylabel='BCE+Dice Loss', title='Loss Curves')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_iou, label='Train IoU', color='#2ecc71', lw=2)
    ax2.plot(epochs, val_iou,   label='Val IoU',   color='#9b59b6', lw=2, ls='--')
    ax2.set(xlabel='Epoch', ylabel='IoU', title='IoU Curves', ylim=[0, 1])
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_ablation_results(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing multiple models across segmentation metrics.

    Args:
        results_dict : {'Model A': {'iou': 0.65, 'dice': 0.78, ...}, ...}
        metrics      : list of metric keys to include (default: all 5)
        save_path    : optional path to save the figure
    """
    if metrics is None:
        metrics = ['iou', 'dice', 'precision', 'recall', 'f1']

    model_names = list(results_dict.keys())
    x     = np.arange(len(metrics))
    width = 0.75 / len(model_names)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, scores) in enumerate(results_dict.items()):
        vals   = [scores.get(m, 0.0) for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=name, color=colors[i % len(colors)], alpha=0.87, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set(xlabel='Metric', ylabel='Score',
           title='Ablation Study — Model Comparison',
           xticks=x, xticklabels=[m.upper() for m in metrics], ylim=[0, 1.15])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def save_figure(fig: plt.Figure, path: str):
    """Save a matplotlib figure to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
