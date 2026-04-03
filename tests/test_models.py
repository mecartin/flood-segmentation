"""
Unit and integration tests for models, losses, and metrics.
Run with:  python -m pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.models  import BaselineUNet, WeatherAwareUNet, get_model
from src.losses  import DiceLoss, BCEDiceLoss, FocalLoss
from src.metrics import compute_iou, compute_dice, compute_precision_recall_f1, MetricTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, H, W = 2, 2, 256, 256   # batch, channels, height, width
N_WEATHER   = 5


@pytest.fixture
def img_batch():
    return torch.rand(B, C, H, W)


@pytest.fixture
def weather_batch():
    return torch.rand(B, N_WEATHER)


@pytest.fixture
def mask_batch():
    m = torch.zeros(B, 1, H, W)
    m[:, :, 80:160, 80:180] = 1.0   # inject a flood region
    return m


@pytest.fixture
def pred_batch(mask_batch):
    pred = mask_batch.clone() + torch.rand_like(mask_batch) * 0.1
    return pred.clamp(0, 1)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestBaselineUNet:
    def test_output_shape(self, img_batch):
        model = BaselineUNet(img_ch=C)
        out = model(img_batch)
        assert out.shape == (B, 1, H, W), f"Expected {(B,1,H,W)}, got {out.shape}"

    def test_output_range(self, img_batch):
        model = BaselineUNet(img_ch=C)
        out = model(img_batch)
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output outside [0,1]"

    def test_weather_arg_ignored(self, img_batch, weather_batch):
        model = BaselineUNet(img_ch=C)
        out1 = model(img_batch, weather=None)
        out2 = model(img_batch, weather=weather_batch)   # should be ignored
        assert torch.allclose(out1, out2)

    def test_factory_baseline(self, img_batch):
        model = get_model('baseline', img_ch=C)
        assert model(img_batch).shape == (B, 1, H, W)


class TestWeatherAwareUNet:
    def test_output_shape(self, img_batch, weather_batch):
        model = WeatherAwareUNet(img_ch=C, weather_dim=N_WEATHER)
        out = model(img_batch, weather_batch)
        assert out.shape == (B, 1, H, W)

    def test_output_range(self, img_batch, weather_batch):
        model = WeatherAwareUNet(img_ch=C, weather_dim=N_WEATHER)
        out = model(img_batch, weather_batch)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_factory_multimodal(self, img_batch, weather_batch):
        model = get_model('multimodal', img_ch=C, weather_dim=N_WEATHER)
        assert model(img_batch, weather_batch).shape == (B, 1, H, W)

    def test_weather_changes_output(self, img_batch, weather_batch):
        model = WeatherAwareUNet(img_ch=C, weather_dim=N_WEATHER)
        out1 = model(img_batch, weather_batch)
        out2 = model(img_batch, torch.zeros_like(weather_batch))
        assert not torch.allclose(out1, out2), "Weather should affect output"


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model('unknown')


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_dice_loss_perfect(self):
        pred   = torch.ones(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        loss   = DiceLoss()(pred, target)
        assert loss.item() < 0.01, "Dice loss should be ~0 for perfect prediction"

    def test_dice_loss_no_overlap(self):
        pred   = torch.ones(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        loss   = DiceLoss()(pred, target)
        assert loss.item() > 0.9, "Dice loss should be ~1 for zero overlap"

    def test_bce_dice_loss_range(self, mask_batch):
        pred = torch.sigmoid(torch.randn_like(mask_batch))
        loss = BCEDiceLoss()(pred, mask_batch)
        assert 0.0 <= loss.item() <= 2.0

    def test_focal_loss_range(self, mask_batch):
        pred = torch.sigmoid(torch.randn_like(mask_batch))
        loss = FocalLoss()(pred, mask_batch)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_iou_perfect(self, mask_batch):
        iou = compute_iou(mask_batch, mask_batch)
        assert abs(iou - 1.0) < 1e-4

    def test_iou_zero(self):
        pred   = torch.ones(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        iou    = compute_iou(pred, target)
        assert iou < 0.01

    def test_dice_perfect(self, mask_batch):
        dice = compute_dice(mask_batch, mask_batch)
        assert abs(dice - 1.0) < 1e-4

    def test_precision_recall_f1_shape(self, pred_batch, mask_batch):
        p, r, f = compute_precision_recall_f1(pred_batch, mask_batch)
        assert all(0.0 <= v <= 1.0 for v in [p, r, f])

    def test_metric_tracker(self, pred_batch, mask_batch):
        tracker = MetricTracker()
        tracker.update(pred_batch, mask_batch)
        tracker.update(pred_batch, mask_batch)
        avgs = tracker.averages()
        assert set(avgs.keys()) == {'iou', 'dice', 'precision', 'recall', 'f1'}
        assert all(0.0 <= v <= 1.0 for v in avgs.values())

    def test_tracker_reset(self, pred_batch, mask_batch):
        tracker = MetricTracker()
        tracker.update(pred_batch, mask_batch)
        tracker.reset()
        avgs = tracker.averages()
        assert all(v == 0.0 for v in avgs.values())


# ---------------------------------------------------------------------------
# Gradient flow check
# ---------------------------------------------------------------------------

def test_gradients_baseline(img_batch, mask_batch):
    model = BaselineUNet(img_ch=C)
    pred  = model(img_batch)
    loss  = BCEDiceLoss()(pred, mask_batch)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_gradients_multimodal(img_batch, weather_batch, mask_batch):
    model = WeatherAwareUNet(img_ch=C, weather_dim=N_WEATHER)
    pred  = model(img_batch, weather_batch)
    loss  = BCEDiceLoss()(pred, mask_batch)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
