"""
3D Segmentation Model Definitions for NeuroVR.

Provides factory functions for:
  - SegResNet (same architecture as bundled MONAI pretrained model)
  - 3D U-Net (alternative architecture)

Both accept 4-channel MRI input and produce 3-channel binary logits.

SegResNet architecture is intentionally identical to the bundled
brats_mri_segmentation model, enabling direct weight transfer for
fine-tuning / transfer learning.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# SegResNet (matches bundled MONAI pretrained model)
# ──────────────────────────────────────────────────────────────────────────────

def create_segresnet(
    in_channels: int = 4,
    out_channels: int = 3,
    init_filters: int = 16,
    blocks_down: Sequence[int] = (1, 2, 2, 4),
    blocks_up: Sequence[int] = (1, 1, 1),
    dropout_prob: float = 0.2,
) -> nn.Module:
    """Create a MONAI SegResNet matching the bundled pretrained model.

    This architecture is identical to the MONAI BraTS model zoo entry
    (brats_mri_segmentation v0.4.2). Use load_pretrained_weights() to
    initialize from the downloaded bundle for transfer learning.

    Args:
        in_channels: MRI modalities (4 for T1/T1ce/T2/FLAIR).
        out_channels: Tumor regions (3 for WT/TC/ET).
        init_filters: Base feature map count.
        blocks_down: Residual blocks per encoder stage.
        blocks_up: Residual blocks per decoder stage.
        dropout_prob: Dropout probability.

    Returns:
        MONAI SegResNet nn.Module (outputs logits, not probabilities).
    """
    try:
        from monai.networks.nets import SegResNet
    except ImportError as exc:
        raise ImportError("MONAI required: pip install monai") from exc

    return SegResNet(
        blocks_down=list(blocks_down),
        blocks_up=list(blocks_up),
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout_prob,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3D U-Net (alternative, more memory-efficient for small patches)
# ──────────────────────────────────────────────────────────────────────────────

def create_3d_unet(
    in_channels: int = 4,
    out_channels: int = 3,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a MONAI 3D U-Net for volumetric segmentation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output classes.
        channels: Feature maps per encoder stage.
        strides: Downsampling strides per stage.
        num_res_units: Residual units per block.
        dropout: Dropout probability.

    Returns:
        MONAI 3D UNet nn.Module.
    """
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise ImportError("MONAI required: pip install monai") from exc

    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
        act="PRELU",
        norm="INSTANCE",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained weight loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """Load pretrained weights into a model, with graceful key mismatch handling.

    Compatible with:
      - MONAI bundle model.pt files
      - Checkpoints saved by this training pipeline
      - Raw state_dict files

    Args:
        model: Target nn.Module.
        checkpoint_path: Path to .pt checkpoint or model.pt bundle file.
        strict: If False, missing/extra keys are logged but not errors.
        device: Map device (defaults to CPU for safe loading).

    Returns:
        (loaded_keys, total_keys) — number of weights matched.
    """
    map_dev = device or torch.device("cpu")
    checkpoint_path = str(checkpoint_path)

    if not Path(checkpoint_path).exists():
        print(f"[Model3D] Pretrained weights not found: {checkpoint_path}")
        print("[Model3D] Training from scratch (random initialization).")
        return 0, sum(1 for _ in model.parameters())

    state = torch.load(checkpoint_path, map_location=map_dev, weights_only=True)

    # Unwrap common checkpoint wrappers
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in state:
                state = state[key]
                break

    result = model.load_state_dict(state, strict=strict)

    if not strict:
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        if n_missing or n_unexpected:
            print(
                f"[Model3D] Partial load: {n_missing} missing keys, "
                f"{n_unexpected} unexpected keys"
            )

    total_keys = len(list(model.state_dict().keys()))
    loaded_keys = total_keys - len(result.missing_keys)
    print(
        f"[Model3D] ✓ Loaded pretrained weights: "
        f"{loaded_keys}/{total_keys} keys from {Path(checkpoint_path).name}"
    )
    return loaded_keys, total_keys


# ──────────────────────────────────────────────────────────────────────────────
# Model summary
# ──────────────────────────────────────────────────────────────────────────────

def get_model_summary(model: nn.Module) -> Tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module, prefix: str = "3D Model") -> None:
    """Print concise model parameter summary."""
    total, trainable = get_model_summary(model)
    print(f"[{prefix}] Total params    : {total:,}")
    print(f"[{prefix}] Trainable params: {trainable:,}")
