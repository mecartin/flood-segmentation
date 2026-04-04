import torch
import sys, os
from src.datasets import build_datasets
from src.models import get_model
from src.train import train_model
from src.losses import BCEDiceLoss

DATA_DIR = 'data/raw'
WEATHER_CSV = 'data/processed/weather.csv'

train_ds, val_ds, test_ds = build_datasets(
    data_dir=DATA_DIR,
    weather_csv_path=WEATHER_CSV,
    img_size=256,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model('baseline', img_ch=2, base_ch=64, drop_prob=0.1)

print("Testing BCEDiceLoss...")
import src.train
src.train.SymmetricUnifiedFocalLoss = lambda delta, gamma: BCEDiceLoss()

history, best_iou = train_model(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=4,
    batch_size=4,
    lr=3e-4,
    patience=3,
    num_workers=0,
    device=device,
    warmup_epochs=1,
    accum_steps=2
)
