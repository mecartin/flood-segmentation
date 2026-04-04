#!/usr/bin/env python3
"""
demo.py

End-to-end demonstration script for the Weather-Aware Flood Segmentation project.
Loads the pre-trained Baseline and Weather-Aware (Multimodal) models, runs inference 
on a selection of test images, and generates a side-by-side visual comparison.

Usage:
    python scripts/demo.py [--samples 4] [--output results/figures/demo_comparison.png]
"""

import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Ensure the root project directory is in the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets import build_datasets, WEATHER_FEATURES
from src.models import get_model
from src.metrics import compute_iou


def main():
    parser = argparse.ArgumentParser(description="Flood Segmentation Demo")
    parser.add_argument('--samples', type=int, default=4, help='Number of test samples to visualize')
    parser.add_argument('--output', type=str, default='results/figures/demo_comparison.png', help='Output image path')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to raw dataset')
    parser.add_argument('--weather_csv', type=str, default='data/processed/weather.csv', help='Path to weather CSV')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running Demo on device: {device}")

    # Validate paths
    baseline_ckpt_path = os.path.join(project_root, 'results', 'saved_models', 'baseline_unet_v2.pth')
    multimodal_ckpt_path = os.path.join(project_root, 'results', 'saved_models', 'multimodal_unet_v2.pth')
    
    if not os.path.exists(baseline_ckpt_path) or not os.path.exists(multimodal_ckpt_path):
        print("[!] Error: Model checkpoints not found.")
        print(f"    Expected:\n    - {baseline_ckpt_path}\n    - {multimodal_ckpt_path}")
        print("    Please run the training notebooks/scripts first, or update the checkpoint paths.")
        sys.exit(1)

    print("[*] Building datasets (using Test Split for demo)...")
    # Using the same seed 42 to ensure consistent splits
    _, _, test_ds = build_datasets(
        data_dir=os.path.join(project_root, args.data_dir),
        weather_csv_path=os.path.join(project_root, args.weather_csv),
        img_size=256,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )

    if len(test_ds) == 0:
        print("[!] Error: Test dataset is empty.")
        sys.exit(1)

    batch_size = min(args.samples, len(test_ds))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Grab one batch
    imgs, weather_feats, masks = next(iter(loader))

    print("[*] Loading Baseline U-Net...")
    model_A = get_model('baseline', img_ch=2, base_ch=64).to(device)
    ckpt_A = torch.load(baseline_ckpt_path, map_location=device)
    model_A.load_state_dict(ckpt_A['model_state_dict'])
    model_A.eval()

    print("[*] Loading Weather-Aware U-Net...")
    model_B = get_model('multimodal', img_ch=2, weather_dim=len(WEATHER_FEATURES), base_ch=64).to(device)
    ckpt_B = torch.load(multimodal_ckpt_path, map_location=device)
    model_B.load_state_dict(ckpt_B['model_state_dict'])
    model_B.eval()

    print("[*] Running inference...")
    with torch.no_grad():
        logits_A = model_A(imgs.to(device))
        logits_B = model_B(imgs.to(device), weather_feats.to(device))
        
        preds_A = torch.sigmoid(logits_A).cpu()
        preds_B = torch.sigmoid(logits_B).cpu()
        
    # Build comparison grid
    print(f"[*] Visualizing {batch_size} samples...")
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    # Handle single sample case
    if batch_size == 1:
        axes = [axes]

    headers = ['Satellite (VV)', 'Ground Truth', 'Model A (Baseline)', 'Model B (Multimodal)']
    for col, h in enumerate(headers):
        axes[0][col].set_title(h, fontsize=12, fontweight='bold', pad=10)

    for i in range(batch_size):
        img_np = imgs[i, 0].numpy()  # Channel 0 is VV
        gt     = masks[i, 0].numpy()
        pa     = (preds_A[i, 0] >= 0.5).float().numpy()
        pb     = (preds_B[i, 0] >= 0.5).float().numpy()

        # Compute metrics for labels
        iou_a  = compute_iou(preds_A[i:i+1], masks[i:i+1])
        iou_b  = compute_iou(preds_B[i:i+1], masks[i:i+1])

        # Plot 1: Source Image
        axes[i][0].imshow(img_np, cmap='gray', interpolation='bilinear')
        axes[i][0].axis('off')

        # Plot 2: Ground Truth
        axes[i][1].imshow(gt, cmap='Blues', vmin=0, vmax=1)
        axes[i][1].axis('off')

        # Plot 3: Baseline
        axes[i][2].imshow(pa, cmap='Blues', vmin=0, vmax=1)
        axes[i][2].set_title(f"IoU: {iou_a:.3f}", fontsize=11, color='black', y=-0.15)
        axes[i][2].axis('off')

        # Plot 4: Multimodal (color IoU green if it improved, red if it degraded vs baseline)
        axes[i][3].imshow(pb, cmap='Blues', vmin=0, vmax=1)
        color = 'green' if iou_b > iou_a else ('red' if iou_b < iou_a else 'black')
        axes[i][3].set_title(f"IoU: {iou_b:.3f}", fontsize=11, color=color, y=-0.15)
        axes[i][3].axis('off')

    plt.suptitle('Flood Segmentation Demo: Baseline vs. Weather-Aware U-Net', 
                 fontsize=16, fontweight='bold', y=0.95 + (0.05 if batch_size == 1 else 0))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[*] Demo completed successfully! Output saved to: {args.output}")

if __name__ == '__main__':
    main()
