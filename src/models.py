"""
Model definitions for flood segmentation.

Models:
  - BaselineUNet     : image-only U-Net with CBAM attention gates (Model A)
  - WeatherAwareUNet : multimodal U-Net with multi-scale FiLM weather fusion (Model B)
  - get_model()      : factory function

Key improvements over v1:
  * CBAM (Channel + Spatial Attention) on every skip connection — lets the
    decoder focus on flood-relevant features before merging.
  * DropBlock regularisation in the bottleneck — structured dropout that
    outperforms standard Dropout for convolutional features.
  * Deeper weather MLP with LayerNorm for stable training.
  * Multi-scale FiLM fusion in WeatherAwareUNet — weather modulates all 4
    decoder levels, not just the bottleneck.
  * Weight initialisation: Kaiming for conv, constant for BN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Utility: weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool then DoubleConv encoder block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """ConvTranspose2d upsample + optional attention on skip + DoubleConv decoder block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = CBAM(skip_ch) if use_attention else nn.Identity()
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.attn(skip)   # attend to relevant skip features
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., 2018)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx  = self.fc(self.max_pool(x).view(B, C))
        scale = torch.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise statistics."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Channel + Spatial attention (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# DropBlock: structured dropout for convolutional features
# ---------------------------------------------------------------------------

class DropBlock2D(nn.Module):
    """
    DropBlock (Ghiasi et al., 2018): drops contiguous spatial blocks instead
    of independent pixels. More effective than Dropout for conv layers.
    """

    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob  = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        B, C, H, W = x.shape
        # Seed mask (before block expansion)
        seed_drop_rate = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones(B, C, H, W, device=x.device) * (1 - seed_drop_rate))
        # Expand seeds into blocks via max-pooling
        mask = 1 - F.max_pool2d(
            1 - mask,
            kernel_size=(self.block_size, self.block_size),
            stride=1,
            padding=self.block_size // 2,
        )[:, :, :H, :W]
        # Normalise to maintain expected activation magnitude
        mask = mask / (mask.mean() + 1e-6)
        return x * mask


# ---------------------------------------------------------------------------
# Model A: Baseline U-Net with CBAM attention (image only)
# ---------------------------------------------------------------------------

class BaselineUNet(nn.Module):
    """
    4-level U-Net for binary flood segmentation from Sentinel-1 SAR imagery.

    Improvements over v1:
      - CBAM attention on each skip connection (channel + spatial gating)
      - DropBlock regularisation at the bottleneck
      - Kaiming weight initialisation

    Args:
        img_ch    : number of input image channels (2 for Sentinel-1 VV+VH)
        base_ch   : base channel width (doubled at each encoder level)
        drop_prob : DropBlock drop probability (0 = disabled)
    Input  : img   [B, img_ch, H, W]
    Output : mask  [B, 1, H, W]  (raw logits — apply sigmoid for probabilities)
    """

    def __init__(self, img_ch: int = 2, base_ch: int = 64, drop_prob: float = 0.1):
        super().__init__()
        c = base_ch
        self.enc1 = DoubleConv(img_ch, c)
        self.enc2 = Down(c, c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)
        self.bottleneck = Down(c * 8, c * 16)
        self.dropblock  = DropBlock2D(block_size=7, drop_prob=drop_prob)

        # Decoder with CBAM on skip connections
        self.up4 = Up(c * 16, c * 8,  c * 8,  use_attention=True)
        self.up3 = Up(c * 8,  c * 4,  c * 4,  use_attention=True)
        self.up2 = Up(c * 4,  c * 2,  c * 2,  use_attention=True)
        self.up1 = Up(c * 2,  c,      c,       use_attention=True)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, img: torch.Tensor, weather: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.dropblock(self.bottleneck(e4))
        return self.out_conv(self.up1(self.up2(self.up3(self.up4(b, e4), e3), e2), e1))


# ---------------------------------------------------------------------------
# Weather MLP branch
# ---------------------------------------------------------------------------

class WeatherMLP(nn.Module):
    """
    Weather feature encoder with LayerNorm for stable training.
    Outputs a rich embedding used for multi-scale FiLM modulation.
    """

    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation: modulates a [B, C, H, W] feature map
    using a [B, emb_dim] weather embedding.
      out = x * (1 + gamma) + beta,  gamma/beta ∈ R^C broadcast spatially.
    """

    def __init__(self, emb_dim: int, num_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(emb_dim, num_channels)
        self.beta_fc  = nn.Linear(emb_dim, num_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(w).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        beta  = self.beta_fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Model B: Weather-Aware U-Net with multi-scale FiLM (multimodal)
# ---------------------------------------------------------------------------

class WeatherAwareUNet(nn.Module):
    """
    Multimodal U-Net: fuses satellite imagery with weather metadata at
    multiple decoder scales via Feature-wise Linear Modulation (FiLM).

    Architecture improvements over v1:
      - FiLM applied at ALL 4 decoder scales (not just bottleneck)
        so weather context propagates throughout the full decoder.
      - CBAM attention on every skip connection.
      - Deeper weather MLP with LayerNorm + GELU.
      - DropBlock at the bottleneck.

    Args:
        img_ch          : input image channels (2 for Sentinel-1)
        weather_dim     : number of weather features
        base_ch         : base channel width
        weather_emb_dim : weather embedding size
        drop_prob       : DropBlock probability
    Input  : img [B, img_ch, H, W], weather [B, weather_dim]
    Output : mask [B, 1, H, W]  (raw logits — apply sigmoid for probabilities)
    """

    def __init__(self, img_ch: int = 2, weather_dim: int = 5,
                 base_ch: int = 64, weather_emb_dim: int = 128,
                 drop_prob: float = 0.1):
        super().__init__()
        c = base_ch
        self.weather_mlp = WeatherMLP(weather_dim, emb_dim=weather_emb_dim)

        self.enc1 = DoubleConv(img_ch, c)
        self.enc2 = Down(c,     c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)

        # Bottleneck
        self.pool          = nn.MaxPool2d(2)
        self.film_bottle   = FiLMBlock(weather_emb_dim, c * 8)      # modulate before bottleneck conv
        self.bottleneck_conv = DoubleConv(c * 8, c * 16)
        self.dropblock     = DropBlock2D(block_size=7, drop_prob=drop_prob)

        # Decoder with CBAM skip attention
        self.up4   = Up(c * 16, c * 8, c * 8,  use_attention=True)
        self.up3   = Up(c * 8,  c * 4, c * 4,  use_attention=True)
        self.up2   = Up(c * 4,  c * 2, c * 2,  use_attention=True)
        self.up1   = Up(c * 2,  c,     c,       use_attention=True)

        # Multi-scale FiLM: weather modulates EACH decoder output
        self.film4 = FiLMBlock(weather_emb_dim, c * 8)
        self.film3 = FiLMBlock(weather_emb_dim, c * 4)
        self.film2 = FiLMBlock(weather_emb_dim, c * 2)
        self.film1 = FiLMBlock(weather_emb_dim, c)

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, img: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        # Encode image
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Encode weather
        w = self.weather_mlp(weather)   # [B, emb_dim]

        # Bottleneck with FiLM before processing
        pooled = self.pool(e4)
        pooled = self.film_bottle(pooled, w)
        b = self.dropblock(self.bottleneck_conv(pooled))

        # Decode with multi-scale weather modulation
        d4 = self.film4(self.up4(b,  e4), w)
        d3 = self.film3(self.up3(d4, e3), w)
        d2 = self.film2(self.up2(d3, e2), w)
        d1 = self.film1(self.up1(d2, e1), w)

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(model_type: str = 'baseline', img_ch: int = 2,
              weather_dim: int = 5, base_ch: int = 64,
              drop_prob: float = 0.1) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_type : 'baseline' or 'multimodal'
        img_ch     : image input channels
        weather_dim: number of weather features (only for multimodal)
        base_ch    : U-Net base channel width
        drop_prob  : DropBlock drop probability in bottleneck (0 = disabled)
    """
    if model_type == 'baseline':
        return BaselineUNet(img_ch=img_ch, base_ch=base_ch, drop_prob=drop_prob)
    elif model_type == 'multimodal':
        return WeatherAwareUNet(img_ch=img_ch, weather_dim=weather_dim,
                                base_ch=base_ch, drop_prob=drop_prob)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'baseline' or 'multimodal'.")
