"""
PyTorch Dataset for Sen1Floods11 + weather features.

Sen1Floods11 file naming convention:
  S1Hand/        <Event>_<id>_S1Hand.tif        (Sentinel-1 SAR, 2-band VV+VH)
  S1HandLabels/  <Event>_<id>_LabelHand.tif     (binary flood mask)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import rasterio
from rasterio.enums import Resampling

WEATHER_FEATURES = ['precipitation', 'temperature', 'humidity', 'wind_speed', 'pressure']


class FloodDataset(Dataset):
    """
    Dataset pairing Sen1Floods11 Sentinel-1 images with flood masks and weather features.

    Args:
        image_paths  : list of paths to Sentinel-1 GeoTIFF files (2-band VV, VH)
        mask_paths   : list of paths to corresponding label GeoTIFF files
        weather_df   : DataFrame indexed by image basename with WEATHER_FEATURES columns
        img_size     : spatial size to resize images/masks (square)
        split        : 'train', 'val', or 'test'
        augment      : apply random augmentations (train only)
        weather_mean : per-feature mean for z-score normalisation (set after split)
        weather_std  : per-feature std  for z-score normalisation (set after split)
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        weather_df: pd.DataFrame,
        img_size: int = 256,
        split: str = 'train',
        augment: bool = True,
        weather_mean: Optional[np.ndarray] = None,
        weather_std: Optional[np.ndarray] = None,
    ):
        assert len(image_paths) == len(mask_paths), "Mismatch between image and mask counts"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.weather_df  = weather_df
        self.img_size    = img_size
        self.split       = split
        self.augment     = augment and (split == 'train')
        self.weather_mean = weather_mean
        self.weather_std  = weather_std

    def set_weather_stats(self, mean: np.ndarray, std: np.ndarray):
        self.weather_mean = mean
        self.weather_std  = std

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """Load and bilinearly resize a multi-band GeoTIFF → float32 [C, H, W]."""
        with rasterio.open(path) as src:
            img = src.read(
                out_shape=(src.count, self.img_size, self.img_size),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """Load and nearest-neighbour resize a label GeoTIFF → float32 [H, W]."""
        with rasterio.open(path) as src:
            mask = src.read(
                1,
                out_shape=(self.img_size, self.img_size),
                resampling=Resampling.nearest,
            ).astype(np.float32)
        mask = np.clip(mask, 0, 1)   # -1 = no-data → treat as non-flood (0)
        return mask

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Per-channel percentile clipping + min-max normalisation for SAR data."""
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            p2, p98 = np.percentile(img[c], (2, 98))
            denom = p98 - p2
            out[c] = np.clip((img[c] - p2) / (denom + 1e-8), 0.0, 1.0) if denom > 0 else 0.0
        return out

    def _get_weather(self, idx: int) -> np.ndarray:
        """Retrieve and z-score normalise weather vector for image idx."""
        basename = os.path.basename(self.image_paths[idx])
        if basename in self.weather_df.index:
            weather = self.weather_df.loc[basename, WEATHER_FEATURES].values.astype(np.float32)
        else:
            weather = self.weather_df[WEATHER_FEATURES].mean().values.astype(np.float32)

        if self.weather_mean is not None and self.weather_std is not None:
            weather = (weather - self.weather_mean) / (self.weather_std + 1e-8)
        return weather

    # ------------------------------------------------------------------
    # Augmentation pipeline (train-only)
    # ------------------------------------------------------------------

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Comprehensive augmentation for SAR flood imagery:
          - Random flips (H + V)
          - Random 90° rotations
          - SAR speckle noise injection
          - Random brightness / contrast jitter per channel
          - Random cutout (patch erase) to prevent co-variate overfitting
          - Random elastic-style grid distortion
        img: [C, H, W]   mask: [H, W]
        """
        img_hw = img.transpose(1, 2, 0)   # → [H, W, C]

        # ── Geometric ──────────────────────────────────────────────────
        if np.random.rand() < 0.5:
            img_hw = np.fliplr(img_hw).copy()
            mask   = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img_hw = np.flipud(img_hw).copy()
            mask   = np.flipud(mask).copy()
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img_hw = np.rot90(img_hw, k=k).copy()
            mask   = np.rot90(mask,   k=k).copy()

        # ── Grid distortion (lightweight elastic) ──────────────────────
        if np.random.rand() < 0.4:
            img_hw, mask = self._grid_distort(img_hw, mask)

        img = img_hw.transpose(2, 0, 1)   # back to [C, H, W]

        # ── Intensity / noise (image only, not mask) ───────────────────
        # SAR speckle: multiplicative log-normal noise
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(0.02, 0.08)
            noise = np.random.lognormal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img * noise, 0.0, 1.0)

        # Per-channel brightness + contrast jitter
        if np.random.rand() < 0.5:
            for c in range(img.shape[0]):
                alpha = np.random.uniform(0.8, 1.2)   # contrast
                beta  = np.random.uniform(-0.1, 0.1)  # brightness
                img[c] = np.clip(alpha * img[c] + beta, 0.0, 1.0)

        # Random cutout: zero out a random rectangular patch
        if np.random.rand() < 0.4:
            H, W = img.shape[1], img.shape[2]
            ph = np.random.randint(H // 8, H // 3)
            pw = np.random.randint(W // 8, W // 3)
            py = np.random.randint(0, H - ph)
            px = np.random.randint(0, W - pw)
            img[:, py:py+ph, px:px+pw] = 0.0

        return img, mask

    def _grid_distort(
        self, img_hw: np.ndarray, mask: np.ndarray, num_steps: int = 5, distort_limit: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lightweight grid distortion via bilinear interpolation of a coarse displacement field."""
        H, W = img_hw.shape[:2]

        # Build coarse grid of displacements
        xs = np.linspace(0, W - 1, num_steps + 1)
        ys = np.linspace(0, H - 1, num_steps + 1)

        # Perturb interior grid points
        cell_w = W / num_steps
        cell_h = H / num_steps
        dx = np.random.uniform(-distort_limit, distort_limit, (num_steps + 1, num_steps + 1)) * cell_w
        dy = np.random.uniform(-distort_limit, distort_limit, (num_steps + 1, num_steps + 1)) * cell_h

        # Build dense map via upsampling
        from scipy.ndimage import map_coordinates, zoom
        dx_dense = zoom(dx, (H / (num_steps + 1), W / (num_steps + 1)), order=1)[:H, :W]
        dy_dense = zoom(dy, (H / (num_steps + 1), W / (num_steps + 1)), order=1)[:H, :W]

        yy, xx = np.mgrid[0:H, 0:W]
        src_y = np.clip(yy + dy_dense, 0, H - 1)
        src_x = np.clip(xx + dx_dense, 0, W - 1)

        coords = [src_y.ravel(), src_x.ravel()]

        out_img = np.stack([
            map_coordinates(img_hw[:, :, c], coords, order=1, mode='reflect').reshape(H, W)
            for c in range(img_hw.shape[2])
        ], axis=-1).astype(np.float32)
        out_mask = map_coordinates(mask, coords, order=0, mode='reflect').reshape(H, W).astype(np.float32)

        return out_img, out_mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img   = self._normalize_image(self._load_image(self.image_paths[idx]))
        mask  = self._load_mask(self.mask_paths[idx])

        if self.augment:
            img, mask = self._augment(img, mask)

        weather = self._get_weather(idx)

        return (
            torch.from_numpy(img),                    # [C, H, W]
            torch.from_numpy(weather),                # [n_features]
            torch.from_numpy(mask).unsqueeze(0),      # [1, H, W]
        )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(
    data_dir: str,
    weather_csv_path: str,
    img_size: int = 256,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int = 42,
) -> Tuple['FloodDataset', 'FloodDataset', 'FloodDataset']:
    """
    Scan data_dir for S1Hand/*.tif / S1HandLabels/*.tif pairs, split into
    train/val/test subsets, compute weather normalisation statistics, and
    return three ready-to-use FloodDataset objects.

    Expected directory layout:
        data_dir/
            S1Hand/           ← Sentinel-1 images  (*_S1Hand.tif)
            S1HandLabels/     ← flood masks         (*_LabelHand.tif)

        weather_csv_path: CSV with columns [filename, precipitation, temperature,
                                            humidity, wind_speed, pressure]
    """
    import random

    s1_dir    = os.path.join(data_dir, 'S1Hand')
    label_dir = os.path.join(data_dir, 'S1HandLabels')

    image_files = sorted(f for f in os.listdir(s1_dir) if f.endswith('.tif'))
    pairs: List[Tuple[str, str]] = []
    for img_file in image_files:
        label_file = img_file.replace('S1Hand', 'LabelHand')
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            pairs.append((os.path.join(s1_dir, img_file), label_path))

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found in {data_dir}")

    random.seed(seed)
    random.shuffle(pairs)
    n        = len(pairs)
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)

    splits: Dict[str, List] = {
        'train': pairs[:n_train],
        'val':   pairs[n_train:n_train + n_val],
        'test':  pairs[n_train + n_val:],
    }

    weather_df = pd.read_csv(weather_csv_path, index_col='filename')

    # Compute normalisation stats on training filenames only
    train_files  = [os.path.basename(p[0]) for p in splits['train']]
    train_weather = weather_df.loc[
        weather_df.index.isin(train_files), WEATHER_FEATURES
    ].values.astype(np.float32)
    weather_mean = train_weather.mean(axis=0)
    weather_std  = train_weather.std(axis=0)

    datasets = {}
    for split, split_pairs in splits.items():
        img_paths  = [p[0] for p in split_pairs]
        mask_paths = [p[1] for p in split_pairs]
        ds = FloodDataset(
            img_paths, mask_paths, weather_df,
            img_size=img_size, split=split,
            augment=(split == 'train'),
            weather_mean=weather_mean,
            weather_std=weather_std,
        )
        datasets[split] = ds

    print(f"Dataset sizes  →  train: {len(datasets['train'])}  "
          f"val: {len(datasets['val'])}  test: {len(datasets['test'])}")

    return datasets['train'], datasets['val'], datasets['test']
