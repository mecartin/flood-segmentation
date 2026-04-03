# Weather-Aware Flood Segmentation

**Multimodal deep learning for flood mapping from Sentinel-1 SAR imagery and weather data**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project trains and compares two flood segmentation models on the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset:

| Model | Input | Architecture |
|-------|-------|--------------|
| **Model A — Baseline U-Net** | Sentinel-1 SAR (VV + VH) | 4-level U-Net + CBAM attention |
| **Model B — Weather-Aware U-Net** | SAR + weather vector | U-Net + multi-scale FiLM weather fusion |

**Research question:** Does adding meteorological context (precipitation, temperature, humidity, wind speed, pressure) improve flood segmentation accuracy?

---

## Key Features

- **CBAM Attention** on every decoder skip connection (channel + spatial gating)
- **Multi-scale FiLM fusion** — weather modulates all 4 decoder levels in Model B
- **DropBlock** structured regularisation in the bottleneck
- **SymmetricUnifiedFocalLoss** — state-of-the-art for class-imbalanced binary segmentation (Yeung et al., 2021)
- **AdamW + Cosine Annealing Warm Restarts** with linear LR warmup
- **Gradient accumulation** for effective larger batch sizes on constrained GPU memory
- **Rich SAR augmentations**: speckle noise, grid distortion, cutout, brightness/contrast jitter
- **Test-Time Augmentation (TTA)** for improved evaluation metrics

---

## Project Structure

```
flood_segmentation/
├── data/
│   ├── raw/                  # Sen1Floods11 GeoTIFFs (downloaded separately)
│   │   ├── S1Hand/           # Sentinel-1 images (*_S1Hand.tif)
│   │   └── S1HandLabels/     # Flood masks (*_LabelHand.tif)
│   └── processed/
│       └── weather.csv       # Weather features per image
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: images, masks, class balance
│   ├── 02_baseline_unet.ipynb          # Train & evaluate Model A
│   ├── 03_multimodal_unet.ipynb        # Train & evaluate Model B
│   └── 04_evaluation_visualization.ipynb  # Ablation study & overlays
├── src/
│   ├── datasets.py     # FloodDataset + build_datasets() + augmentations
│   ├── models.py       # BaselineUNet, WeatherAwareUNet, CBAM, FiLM, get_model()
│   ├── losses.py       # DiceLoss, BCEDiceLoss, FocalDiceLoss, TverskyLoss,
│   │                   #   TverskyFocalLoss, SymmetricUnifiedFocalLoss
│   ├── metrics.py      # IoU, Dice, Precision, Recall, F1, MetricTracker
│   ├── train.py        # train_model(), evaluate_model() (with TTA)
│   └── utils.py        # Visualisation helpers
├── scripts/
│   ├── download_data.py      # Sen1Floods11 downloader (gsutil / Python GCS)
│   └── generate_weather.py   # Synthetic or NASA POWER weather CSV
├── tests/
│   └── test_models.py        # pytest unit tests
├── results/
│   ├── metrics_baseline.json
│   ├── metrics_multimodal.json
│   └── ablation_summary.csv
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/flood-segmentation.git
cd flood-segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU:** PyTorch will auto-detect CUDA. Training on CPU is supported but slow.  
> **Google Colab:** Each notebook includes a Colab setup cell.

### 3. Download the Sen1Floods11 dataset (~4 GB)

**Option A — Google Cloud SDK (recommended)**
```bash
gcloud auth login
python scripts/download_data.py --output data/raw --method gsutil
```

**Option B — Python only**
```bash
pip install google-cloud-storage
python scripts/download_data.py --output data/raw --method python
```

Expected layout after download:
```
data/raw/
  S1Hand/        ← ~4800 GeoTIFF files  (*_S1Hand.tif)
  S1HandLabels/  ← ~4800 GeoTIFF files  (*_LabelHand.tif)
```

### 4. Generate weather features

```bash
# Fast synthetic weather (for first run / development):
python scripts/generate_weather.py --mode synthetic \
    --image_dir data/raw/S1Hand \
    --output data/processed/weather.csv

# Real NASA POWER historical data (requires internet):
python scripts/generate_weather.py --mode nasa \
    --image_dir data/raw/S1Hand \
    --output data/processed/weather.csv
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Running Experiments

Run notebooks **in order**:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Visualise images, masks, class balance, weather stats |
| `02_baseline_unet.ipynb` | Train & evaluate Baseline U-Net (Model A) |
| `03_multimodal_unet.ipynb` | Train & evaluate Weather-Aware U-Net (Model B) |
| `04_evaluation_visualization.ipynb` | Ablation study, metric comparison, prediction overlays |

---

## Model Architecture

### Model A — Baseline U-Net (with CBAM)

```
Image [B, 2, H, W]
  → Encoder: DoubleConv × 4  (64→128→256→512)
  → Bottleneck (1024) + DropBlock
  → Decoder: Up × 4 with CBAM-attended skip connections
  → Output [B, 1, H, W]  (raw logits)
```

### Model B — Weather-Aware U-Net (multi-scale FiLM)

```
Image [B, 2, H, W]          Weather [B, 5]
  → Encoder (64→512)           → MLP + LayerNorm + GELU
  → Pool → FiLM(w)              → Embedding [B, 128]
  → Bottleneck + DropBlock         ↓
  → Decoder (CBAM skips)    ← FiLM at EACH of 4 decoder levels
  → Output [B, 1, H, W]  (raw logits)
```

**FiLM modulation:**  `output = features × (1 + γ(w)) + β(w)`  where γ, β are linear projections of the weather embedding.

---

## Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Loss | SymmetricUnifiedFocalLoss (δ=0.6, γ=0.5) | Best for imbalanced binary seg |
| Optimiser | AdamW (wd=1e-4) | Decoupled weight decay |
| LR schedule | CosineAnnealingWarmRestarts (T₀=20) | Avoids plateau stagnation |
| LR warmup | 5 epochs linear | Stabilises early training |
| Peak LR | 3×10⁻⁴ | |
| Effective batch | 8 (4 × 2 accum steps) | Fits 4 GB GPU |
| Early stopping | patience=15 on val IoU | |
| TTA at eval | H-flip + V-flip average | Free ~1-2% IoU boost |

---

## Results

> Results are populated after running the notebooks. The JSON files in `results/` are updated automatically.

| Model | Test IoU | Test Dice | Test F1 |
|-------|----------|-----------|---------|
| Baseline U-Net v2 | — | — | — |
| Weather-Aware U-Net v2 | — | — | — |

---

## References

- **Sen1Floods11**: Bonafilia et al. (2020). [arXiv](https://arxiv.org/abs/2012.04377) · [GitHub](https://github.com/cloudtostreet/Sen1Floods11)
- **U-Net**: Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv](https://arxiv.org/abs/1505.04597)
- **CBAM**: Woo et al. (2018). *CBAM: Convolutional Block Attention Module.* [arXiv](https://arxiv.org/abs/1807.06521)
- **FiLM**: Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* [arXiv](https://arxiv.org/abs/1709.07871)
- **Symmetric Unified Focal Loss**: Yeung et al. (2021). [arXiv](https://arxiv.org/abs/2111.07648)
- **DropBlock**: Ghiasi et al. (2018). *DropBlock: A Regularization Method for Convolutional Networks.* [arXiv](https://arxiv.org/abs/1810.12890)
- **NASA POWER**: https://power.larc.nasa.gov/

---

## License

MIT License. See [LICENSE](LICENSE) for details.
