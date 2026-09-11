"""
MONAI-based Training Transform Chains for NeuroVR.

Provides pre-built MONAI Compose transform pipelines for both 2D and 3D
segmentation training. Separated from the inference preprocessor to allow
independent tuning of training augmentations without affecting deployed
inference behaviour.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 3D transform chains
# ──────────────────────────────────────────────────────────────────────────────

def get_3d_train_transforms(
    patch_size: Sequence[int] = (64, 64, 64),
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    pos_samples: int = 1,
    neg_samples: int = 1,
    gaussian_noise_std: float = 0.01,
    intensity_scale_factor: float = 0.1,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
):
    """Build a MONAI Compose for 3D training with augmentation.

    Input dictionary keys expected:
      "image" → list of 4 modality file paths
      "label" → segmentation file path

    Output keys after transforms:
      "image" → [4, ps, ps, ps] float32 tensor (patch)
      "label" → [3, ps, ps, ps] float32 tensor (WT, TC, ET channels)

    Args:
        patch_size: 3D patch size (D, H, W).
        target_spacing: Voxel spacing to resample to (mm).
        pos_samples: Positive (tumor) patches per volume.
        neg_samples: Negative (background) patches per volume.
        gaussian_noise_std: Std for Gaussian noise augmentation.
        intensity_scale_factor: Scale range for intensity augmentation.
        flip_prob: Probability of random flip.
        rotate_prob: Probability of random 90° rotation.

    Returns:
        monai.transforms.Compose object.
    """
    try:
        from monai import transforms as T
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai") from exc

    return T.Compose([
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        # Convert BraTS integer labels → 3-channel binary (WT, TC, ET)
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # Reorient to RAS+ for consistent axis ordering
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Resample to isotropic voxel spacing
        T.Spacingd(
            keys=["image", "label"],
            pixdim=list(target_spacing),
            mode=["bilinear", "nearest"],
        ),
        # Crop to foreground (removes empty background slices)
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        # Z-score normalise per modality channel within non-zero voxels
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Random crop centred on positive (tumour) or negative (background) voxels
        T.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=list(patch_size),
            pos=pos_samples,
            neg=neg_samples,
            num_samples=pos_samples + neg_samples,
            image_key="image",
            image_threshold=0,
        ),
        # ── Augmentations ──────────────────────────────────────────────────
        T.RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),
        T.RandRotate90d(keys=["image", "label"], prob=rotate_prob, max_k=3),
        T.RandGaussianNoised(keys="image", prob=0.15, std=gaussian_noise_std),
        T.RandScaleIntensityd(keys="image", factors=intensity_scale_factor, prob=0.5),
        T.RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
        T.EnsureTyped(keys=["image", "label"]),
    ])


def get_3d_val_transforms(
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
):
    """Build a MONAI Compose for 3D validation (no augmentation, no cropping).

    Full volumes are loaded for sliding-window inference validation.

    Args:
        target_spacing: Voxel spacing to resample to (mm).

    Returns:
        monai.transforms.Compose object.
    """
    try:
        from monai import transforms as T
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai") from exc

    return T.Compose([
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(
            keys=["image", "label"],
            pixdim=list(target_spacing),
            mode=["bilinear", "nearest"],
        ),
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        T.EnsureTyped(keys=["image", "label"]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 2D augmentations (applied in-memory on extracted slices)
# ──────────────────────────────────────────────────────────────────────────────

def get_2d_train_augmentations(
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    noise_prob: float = 0.15,
    noise_std: float = 0.01,
    scale_prob: float = 0.5,
    scale_factor: float = 0.1,
    contrast_prob: float = 0.3,
):
    """Build MONAI 2D augmentation transforms for slice-level training.

    These transforms operate on single-channel or multi-channel 2D tensors.
    Applied AFTER the 3D-to-2D slice extraction.

    Args:
        flip_prob: Probability of random horizontal / vertical flip.
        rotate_prob: Probability of random 90° rotation.
        noise_prob: Probability of adding Gaussian noise.
        noise_std: Std for Gaussian noise.
        scale_prob: Probability of random intensity scaling.
        scale_factor: Intensity scale range (±factor).
        contrast_prob: Probability of contrast adjustment.

    Returns:
        monai.transforms.Compose object for 2D slices.
    """
    try:
        from monai import transforms as T
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai") from exc

    return T.Compose([
        T.RandFlip(prob=flip_prob, spatial_axis=0),
        T.RandFlip(prob=flip_prob, spatial_axis=1),
        T.RandRotate90(prob=rotate_prob, max_k=3),
        T.RandGaussianNoise(prob=noise_prob, std=noise_std),
        T.RandScaleIntensity(factors=scale_factor, prob=scale_prob),
        T.RandAdjustContrast(prob=contrast_prob, gamma=(0.7, 1.5)),
    ])


def get_2d_val_transforms():
    """Identity transform for 2D validation (no augmentation).

    Returns:
        monai.transforms.Compose (identity pass-through).
    """
    try:
        from monai import transforms as T
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai") from exc
    return T.Compose([T.EnsureType()])


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_preprocessed_volume(
    tensor,
    meta: dict,
    max_nan_fraction: float = 0.0,
) -> Tuple[bool, List[str]]:
    """Check a preprocessed tensor for common data quality issues.

    Args:
        tensor: PyTorch tensor [C, H, W, D] or numpy array.
        meta: Metadata dict (must have "patient_id" key).
        max_nan_fraction: Maximum allowable NaN fraction (default 0 = none).

    Returns:
        (is_valid, list_of_error_messages)
    """
    import torch
    errors = []
    pid = meta.get("patient_id", "unknown")

    if isinstance(tensor, torch.Tensor):
        arr = tensor.numpy()
    else:
        import numpy as np
        arr = tensor

    import numpy as np

    # Check dtype
    if arr.dtype not in (np.float32, np.float64):
        errors.append(f"{pid}: Expected float32/64, got {arr.dtype}")

    # Check NaN/Inf
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    total_vox = arr.size
    if nan_count > int(max_nan_fraction * total_vox):
        errors.append(f"{pid}: {nan_count} NaN values ({nan_count/total_vox:.2%})")
    if inf_count > 0:
        errors.append(f"{pid}: {inf_count} Inf values")

    # Check value range (after Z-score norm, should be roughly [-10, 10])
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if abs(vmin) > 100 or abs(vmax) > 100:
        errors.append(f"{pid}: Value range [{vmin:.1f}, {vmax:.1f}] unusually large")

    # Check shape (at least 3D)
    if arr.ndim < 3:
        errors.append(f"{pid}: Expected ≥3D array, got shape {arr.shape}")

    return (len(errors) == 0), errors
