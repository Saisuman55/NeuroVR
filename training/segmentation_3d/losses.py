"""
Loss Functions for 3D Brain Tumor Segmentation.

Identical interface to the 2D losses — same factory pattern,
same loss options — but tuned for 3D volumetric predictions.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_loss(name: str = "dice_ce") -> nn.Module:
    """Factory function for 3D segmentation loss functions.

    Args:
        name: "dice" | "dice_ce" | "dice_focal"

    Returns:
        nn.Module. Expects logit inputs (before sigmoid).
    """
    name = name.lower().strip()
    if name == "dice":
        return _DiceLoss3D()
    elif name == "dice_ce":
        return _DiceCELoss3D()
    elif name == "dice_focal":
        return _DiceFocalLoss3D()
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose: dice, dice_ce, dice_focal")


class _DiceLoss3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from monai.losses import DiceLoss
            self._loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        except ImportError as exc:
            raise ImportError("MONAI required: pip install monai") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


class _DiceCELoss3D(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
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


class _DiceFocalLoss3D(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        try:
            from monai.losses import DiceFocalLoss
            self._loss = DiceFocalLoss(
                sigmoid=True,
                squared_pred=True,
                reduction="mean",
                gamma=gamma,
            )
        except ImportError as exc:
            raise ImportError("MONAI required: pip install monai") from exc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)
