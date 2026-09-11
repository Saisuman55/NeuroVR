"""
2D U-Net Model Definition for NeuroVR Segmentation.

Uses MONAI's 2D UNet as the backbone for slice-level multi-label
brain tumor segmentation.

Input : [B, 4, H, W] — 4 MRI modalities (T1, T1ce, T2, FLAIR)
Output: [B, 3, H, W] — 3 binary channel logits (WT, TC, ET)

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def create_2d_unet(
    in_channels: int = 4,
    out_channels: int = 3,
    channels: Sequence[int] = (32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a MONAI 2D U-Net for multi-label brain tumor segmentation.

    Args:
        in_channels: Number of input channels (4 for BraTS modalities).
        out_channels: Number of output channels (3 for WT/TC/ET).
        channels: Feature map sizes for each encoder stage.
        strides: Downsampling strides per stage.
        num_res_units: Residual units per block.
        dropout: Dropout probability.

    Returns:
        MONAI UNet nn.Module (2D spatial dims, not sigmoid-activated).
        Apply torch.sigmoid() on the output for probabilities.
    """
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai") from exc

    assert len(channels) == len(strides) + 1, (
        f"len(channels) must equal len(strides)+1: {len(channels)} vs {len(strides)+1}"
    )

    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
        act="PRELU",
        norm="INSTANCE",
    )

    return model


def get_model_summary(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters.

    Args:
        model: PyTorch nn.Module.

    Returns:
        (total_parameters, trainable_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module, prefix: str = "2D U-Net") -> None:
    """Print a concise model summary."""
    total, trainable = get_model_summary(model)
    print(f"[{prefix}] Total params    : {total:,}")
    print(f"[{prefix}] Trainable params: {trainable:,}")
    print(f"[{prefix}] Non-trainable   : {total - trainable:,}")
