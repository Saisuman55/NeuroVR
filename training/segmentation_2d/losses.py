"""
Loss Functions for 2D Brain Tumor Segmentation.

All losses operate on multi-label segmentation (3 channels: WT, TC, ET)
with sigmoid activation (not softmax) since regions overlap.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_loss(name: str = "dice_ce") -> nn.Module:
    """Factory function for segmentation loss functions.

    Args:
        name: Loss function name. One of:
            "dice"       — Dice loss only (MONAI DiceLoss)
            "dice_ce"    — Dice + Cross-Entropy (recommended)
            "dice_focal" — Dice + Focal Loss (for class imbalance)

    Returns:
        nn.Module loss function. Expects logit inputs (before sigmoid).
    """
    name = name.lower().strip()

    if name == "dice":
        return _DiceLoss()
    elif name == "dice_ce":
        return _DiceCELoss()
    elif name == "dice_focal":
        return _DiceFocalLoss()
    else:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: dice, dice_ce, dice_focal"
        )


class _DiceLoss(nn.Module):
    """MONAI DiceLoss with sigmoid activation for multi-label segmentation."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from monai.losses import DiceLoss
            self._loss = DiceLoss(
                sigmoid=True,
                squared_pred=True,
                reduction="mean",
            )
        except ImportError as exc:
            raise ImportError("MONAI required: pip install monai") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


class _DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (recommended for BraTS).

    Dice loss handles class imbalance well; CE stabilizes training.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        try:
            from monai.losses import DiceCELoss
            self._loss = DiceCELoss(
                sigmoid=True,
                squared_pred=True,
                reduction="mean",
                lambda_dice=dice_weight,
                lambda_ce=ce_weight,
            )
        except ImportError as exc:
            raise ImportError("MONAI required: pip install monai") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


class _DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss for severely imbalanced regions (e.g., ET).

    Focal loss down-weights easy negatives (abundant background) and
    focuses learning on hard, rare positives.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        try:
            from monai.losses import DiceFocalLoss
            self._loss = DiceFocalLoss(
                sigmoid=True,
                squared_pred=True,
                reduction="mean",
                gamma=gamma,
                lambda_dice=dice_weight,
                lambda_focal=focal_weight,
            )
        except ImportError as exc:
            raise ImportError("MONAI required: pip install monai") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)
