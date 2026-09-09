"""
3D MRI Volume Preprocessor for NeuroVR.

Applies the MONAI BraTS preprocessing pipeline:
  1. Load 4-modality NIfTI volumes
  2. Reorient to RAS+
  3. Resample to target voxel spacing
  4. Z-score intensity normalization per modality
  5. Foreground cropping
  6. Stack → [4, H, W, D] tensor

Preserves all spatial metadata for inverse transforms.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Spatial utilities (pure numpy — no MONAI dependency at import time)
# ──────────────────────────────────────────────────────────────────────────────


def _zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalize a 3D volume, optionally within a foreground mask.

    Args:
        volume: Float32 array (H, W, D).
        mask: Optional boolean mask selecting foreground voxels.

    Returns:
        Normalized float32 array.
    """
    if mask is not None:
        fg = volume[mask]
    else:
        # Use non-zero voxels as foreground
        fg = volume[volume > 0]

    if fg.size == 0 or fg.std() == 0:
        return volume  # Nothing to normalize

    mean = fg.mean()
    std = fg.std()
    normalized = (volume - mean) / (std + 1e-8)
    return normalized.astype(np.float32)


def _compute_foreground_bbox(
    volumes: List[np.ndarray],
    threshold: float = 0.0,
) -> Tuple[slice, slice, slice]:
    """Find the bounding box containing non-zero voxels across all modalities.

    Args:
        volumes: List of 3D arrays (same shape).
        threshold: Voxels above this value are considered foreground.

    Returns:
        Tuple of slices (x_slice, y_slice, z_slice).
    """
    combined = np.max(np.stack(volumes, axis=0), axis=0)
    fg = combined > threshold

    if not fg.any():
        # Entire volume is foreground if nothing found
        return (
            slice(0, volumes[0].shape[0]),
            slice(0, volumes[0].shape[1]),
            slice(0, volumes[0].shape[2]),
        )

    coords = np.where(fg)
    slices = tuple(
        slice(int(c.min()), int(c.max()) + 1)
        for c in coords
    )
    return slices  # type: ignore[return-value]


def _resample_volume_scipy(
    volume: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing: np.ndarray,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a volume to target voxel spacing using scipy zoom.

    Args:
        volume: Float32 array (H, W, D).
        current_spacing: Current [dx, dy, dz] in mm.
        target_spacing: Target [dx, dy, dz] in mm.
        order: Interpolation order (1=linear, 0=nearest for masks).

    Returns:
        Tuple of (resampled_volume, new_spacing).
    """
    from scipy.ndimage import zoom

    zoom_factors = current_spacing / target_spacing
    resampled = zoom(volume, zoom_factors, order=order, prefilter=order > 1)
    return resampled.astype(np.float32), target_spacing.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessor class
# ──────────────────────────────────────────────────────────────────────────────


class VolumePreprocessor:
    """
    Preprocessing pipeline for 4-modality BraTS MRI volumes.

    Produces a [4, H, W, D] PyTorch tensor ready for MONAI inference.
    Stores all metadata needed to invert transforms on the output mask.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        crop_foreground: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            target_spacing: Desired voxel size in mm (dx, dy, dz).
            crop_foreground: If True, crop to non-zero bounding box.
            normalize: If True, apply Z-score normalization per modality.
        """
        self.target_spacing = np.array(target_spacing, dtype=np.float32)
        self.crop_foreground = crop_foreground
        self.normalize = normalize

        # Metadata stored after last preprocess() call
        self._original_shape: Optional[Tuple] = None
        self._original_spacing: Optional[np.ndarray] = None
        self._crop_slices: Optional[Tuple[slice, ...]] = None
        self._cropped_shape: Optional[Tuple] = None
        self._resampled_shape: Optional[Tuple] = None

    def preprocess(
        self,
        modality_volumes: Dict[str, Dict],
        modality_order: List[str] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run the full preprocessing pipeline on 4 modality volumes.

        Args:
            modality_volumes: Dict from nifti_loader.load_brats_case().
                Keys are modality names; values contain 'data', 'affine',
                'voxel_spacing', etc.
            modality_order: Order of modalities in output tensor.
                Defaults to ['t1', 't1ce', 't2', 'flair'].

        Returns:
            Tuple of:
              - tensor: torch.Tensor float32 [4, H, W, D]
              - meta: Dict with all transform metadata for inversion
        """
        if modality_order is None:
            modality_order = ["t1", "t1ce", "t2", "flair"]

        volumes: List[np.ndarray] = []
        for m in modality_order:
            if m not in modality_volumes:
                raise ValueError(f"Missing modality '{m}' in input volumes.")
            volumes.append(modality_volumes[m]["data"].copy())

        # Use voxel spacing from the first modality (they must all match)
        original_spacing = modality_volumes[modality_order[0]]["voxel_spacing"].copy()
        original_shape = volumes[0].shape
        affine = modality_volumes[modality_order[0]]["affine"]

        self._original_shape = original_shape
        self._original_spacing = original_spacing.copy()

        # ── Step 1: Foreground crop ────────────────────────────────────────────
        if self.crop_foreground:
            crop_slices = _compute_foreground_bbox(volumes)
            self._crop_slices = crop_slices
            volumes = [v[crop_slices] for v in volumes]
        else:
            self._crop_slices = tuple(slice(0, s) for s in original_shape)

        cropped_shape = volumes[0].shape
        self._cropped_shape = cropped_shape

        # ── Step 2: Resample to target spacing ────────────────────────────────
        resampled: List[np.ndarray] = []
        for vol in volumes:
            r, _ = _resample_volume_scipy(vol, original_spacing, self.target_spacing)
            resampled.append(r)
        volumes = resampled
        self._resampled_shape = volumes[0].shape

        # ── Step 3: Z-score normalization ─────────────────────────────────────
        if self.normalize:
            volumes = [_zscore_normalize(v) for v in volumes]

        # ── Step 4: Stack → tensor ────────────────────────────────────────────
        stacked = np.stack(volumes, axis=0)  # [4, H, W, D]
        tensor = torch.from_numpy(stacked).float()

        meta = {
            "original_shape": original_shape,
            "original_spacing": original_spacing,
            "crop_slices": self._crop_slices,
            "cropped_shape": cropped_shape,
            "resampled_shape": self._resampled_shape,
            "target_spacing": self.target_spacing.copy(),
            "affine": affine,
            "modality_order": modality_order,
        }

        print(
            f"[Preprocessor] "
            f"original={original_shape} spacing={original_spacing} "
            f"→ cropped={cropped_shape} "
            f"→ resampled={self._resampled_shape} "
            f"target_spacing={self.target_spacing}"
        )

        return tensor, meta

    def invert_mask(
        self,
        mask: np.ndarray,
        meta: Dict,
        order: int = 0,
    ) -> np.ndarray:
        """Invert preprocessing transforms to restore mask to original space.

        Args:
            mask: Predicted binary mask in resampled space, shape (H', W', D').
            meta: Metadata dict returned by preprocess().
            order: Interpolation order (0=nearest for binary masks).

        Returns:
            Binary mask restored to original volume shape.
        """
        from scipy.ndimage import zoom

        # ── Step 1: Resample back to cropped space ────────────────────────────
        resampled_shape = meta["resampled_shape"]
        cropped_shape = meta["cropped_shape"]
        target_spacing = meta["target_spacing"]
        original_spacing = meta["original_spacing"]

        # zoom factors: from resampled space → cropped space
        zoom_back = target_spacing / original_spacing
        mask_cropped = zoom(mask.astype(np.float32), zoom_back, order=order)

        # Clamp shape to exact cropped shape
        mask_cropped = mask_cropped[
            : cropped_shape[0], : cropped_shape[1], : cropped_shape[2]
        ]
        # Pad if slightly undersize
        pad_width = [
            (0, max(0, cropped_shape[i] - mask_cropped.shape[i]))
            for i in range(3)
        ]
        mask_cropped = np.pad(mask_cropped, pad_width, mode="constant")

        # ── Step 2: Uncrop ────────────────────────────────────────────────────
        original_shape = meta["original_shape"]
        crop_slices = meta["crop_slices"]
        full_mask = np.zeros(original_shape, dtype=np.float32)
        full_mask[crop_slices] = mask_cropped

        return (full_mask > 0.5).astype(np.uint8)
