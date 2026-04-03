"""
Loss functions for flood segmentation.
All losses accept RAW LOGITS (no sigmoid applied beforehand).
This is required for AMP (autocast) safety — F.binary_cross_entropy and
nn.BCELoss are unsafe under autocast; the *_with_logits variants are safe.

Available losses:
  - DiceLoss              : soft Dice, logit-safe
  - BCEDiceLoss           : BCE + Dice combo
  - FocalLoss             : focal loss for class imbalance
  - FocalDiceLoss         : Focal + Dice (previous default)
  - TverskyLoss           : generalised Dice with FP/FN weighting (alpha/beta)
  - TverskyFocalLoss      : Tversky + Focal combo
  - SymmetricUnifiedFocal : best reported for imbalanced binary segmentation
                            (Yeung et al. 2021 – "Unified Focal Loss")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.
    Accepts raw logits; applies sigmoid internally before computing overlap.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)          # logits → probabilities
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice_coeff


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss. Accepts raw logits.
    Uses BCEWithLogitsLoss (AMP-safe) instead of BCELoss.
    L = lambda_bce * BCE + lambda_dice * Dice
    """

    def __init__(self, lambda_bce: float = 0.5, lambda_dice: float = 0.5):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.bce = nn.BCEWithLogitsLoss()   # AMP-safe; includes sigmoid internally
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_bce * self.bce(pred, target) + self.lambda_dice * self.dice(pred, target)


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy negatives to focus on hard flood pixels.
    Accepts raw logits; uses binary_cross_entropy_with_logits (AMP-safe).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # AMP-safe
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss.
    Focal handles class imbalance by downweighting easy negatives.
    Dice improves overlap quality.
    """

    def __init__(self, lambda_focal: float = 0.5, lambda_dice: float = 0.5,
                 alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_focal * self.focal(pred, target) + self.lambda_dice * self.dice(pred, target)


# ---------------------------------------------------------------------------
# Tversky loss — best for highly imbalanced segmentation
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky loss: a generalisation of Dice that separately penalises
    false positives (alpha) and false negatives (beta).

    For flood segmentation (rare positive class), set beta > alpha to
    penalise missed floods more than false alarms.
      Recommended: alpha=0.3, beta=0.7  →  strong FN penalty

    Tversky index = TP / (TP + alpha*FP + beta*FN)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        p    = prob.view(-1)
        t    = target.view(-1)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class TverskyFocalLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan, 2019):
      L = (1 - Tversky)^gamma
    Applies focal modulation on top of Tversky to further focus on hard examples.

    Recommended hyperparameters for flood segmentation:
      alpha=0.3, beta=0.7, gamma=1.33
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma   = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tl = self.tversky(pred, target)
        return tl ** self.gamma


# ---------------------------------------------------------------------------
# Symmetric Unified Focal Loss (Yeung et al., 2021) — state-of-the-art
# for imbalanced binary segmentation; outperforms Focal+Dice on medical/
# remote-sensing benchmarks.
# ---------------------------------------------------------------------------

class SymmetricUnifiedFocalLoss(nn.Module):
    """
    Symmetric Unified Focal Loss (SUF):
      L = delta * L_sym_focal + (1 - delta) * L_sym_focal_dice

    where:
      L_sym_focal      = 0.5 * [Focal(p) + Focal(1-p)] (symmetric focal)
      L_sym_focal_dice = 0.5 * [FocalTversky(p) + FocalTversky(1-p)]

    Symmetric formulation ensures both flood and non-flood classes
    contribute equally to the gradient.

    Args:
        delta : weight between focal and focal-Tversky terms (default 0.6)
        gamma : focal modulation exponent (default 0.5 — mild, per paper)
    """

    def __init__(self, delta: float = 0.6, gamma: float = 0.5):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def _sym_focal(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Symmetric focal cross-entropy."""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p   = torch.sigmoid(pred)
        # positive and negative focal terms
        focal_pos = (1 - p) ** self.gamma * bce
        focal_neg = p       ** self.gamma * F.binary_cross_entropy_with_logits(
            -pred, 1 - target, reduction='none'
        )
        return 0.5 * (focal_pos + focal_neg).mean()

    def _sym_focal_tversky(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Symmetric Focal Tversky term (FP/FN balanced + focal)."""
        p = torch.sigmoid(pred).view(-1)
        t = target.view(-1)
        smooth = 1.0

        def _tversky_coeff(pp, tt):
            tp = (pp * tt).sum()
            fp = (pp * (1 - tt)).sum()
            fn = ((1 - pp) * tt).sum()
            return (tp + smooth) / (tp + 0.5 * fp + 0.5 * fn + smooth)

        tc_pos = _tversky_coeff(p, t)
        tc_neg = _tversky_coeff(1 - p, 1 - t)
        return 0.5 * ((1 - tc_pos) ** self.gamma + (1 - tc_neg) ** self.gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.delta * self._sym_focal(pred, target)
            + (1.0 - self.delta) * self._sym_focal_tversky(pred, target)
        )
