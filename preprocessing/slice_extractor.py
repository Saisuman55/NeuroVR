"""
2D Slice Extractor for NeuroVR Training Pipeline.

Extracts 2D slices from 4-channel 3D MRI volumes for the 2D segmentation
pipeline. Designed to be called AFTER patient-level splitting to prevent
data leakage — this module never sees patients from multiple splits.

Key features:
  - Axial / coronal / sagittal orientations
  - Tumor-focused sampling (configurable ratio)
  - Lazy, on-the-fly extraction (no disk pre-caching needed)
  - Metadata tracking per slice for reproducibility

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SliceMetadata:
    """Metadata for one extracted 2D slice."""
    patient_id: str
    slice_index: int
    orientation: str          # "axial" | "coronal" | "sagittal"
    has_tumor: bool
    wt_voxels: int
    tc_voxels: int
    et_voxels: int
    volume_shape: Tuple[int, int, int]   # (H, W, D) of the source 3D volume


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def _get_slice(
    volume: np.ndarray,
    idx: int,
    orientation: str,
) -> np.ndarray:
    """Extract a single 2D slice from a [C, H, W, D] volume.

    Args:
        volume: Array [C, H, W, D] (float32).
        idx: Slice index along the chosen axis.
        orientation: "axial", "coronal", or "sagittal".

    Returns:
        2D slice [C, H', W'] (float32).
    """
    if orientation == "axial":
        return volume[:, :, :, idx]        # [C, H, W]
    elif orientation == "coronal":
        return volume[:, :, idx, :]        # [C, H, D]
    elif orientation == "sagittal":
        return volume[:, idx, :, :]        # [C, W, D]
    else:
        raise ValueError(f"Unknown orientation: {orientation}. Use axial/coronal/sagittal")


def _get_axis_length(shape: Tuple, orientation: str) -> int:
    """Return the number of slices along the extraction axis.

    Args:
        shape: Volume shape (H, W, D).
        orientation: Orientation string.

    Returns:
        Number of slices.
    """
    h, w, d = shape
    if orientation == "axial":
        return d
    elif orientation == "coronal":
        return w
    elif orientation == "sagittal":
        return h
    else:
        raise ValueError(f"Unknown orientation: {orientation}")


def get_tumor_slice_indices(
    seg_3ch: np.ndarray,
    orientation: str,
) -> List[int]:
    """Get slice indices that contain at least one tumor voxel.

    Args:
        seg_3ch: Binary segmentation array [3, H, W, D] (WT/TC/ET).
        orientation: Orientation string.

    Returns:
        Sorted list of slice indices with tumor.
    """
    any_tumor = seg_3ch.any(axis=0)  # [H, W, D] — True where any region has tumor
    if orientation == "axial":
        axis = 2
    elif orientation == "coronal":
        axis = 1
    else:
        axis = 0

    # Sum over non-slice axes
    axes_to_sum = tuple(i for i in range(3) if i != axis)
    slice_has_tumor = any_tumor.sum(axis=axes_to_sum) > 0  # [N_slices]
    return sorted(np.where(slice_has_tumor)[0].tolist())


def sample_slice_indices(
    n_total: int,
    tumor_indices: List[int],
    tumor_ratio: float = 0.6,
    max_slices: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Sample slice indices with a given tumor/non-tumor ratio.

    IMPORTANT: Patient-level splitting must be done BEFORE calling this.
    This function only operates on slices within a single patient.

    Args:
        n_total: Total number of slices in the volume.
        tumor_indices: Indices of slices containing tumor.
        tumor_ratio: Fraction of sampled slices that should contain tumor.
        max_slices: Maximum slices to return (None = all sampled).
        rng: Optional seeded random.Random instance.

    Returns:
        Sorted list of selected slice indices.
    """
    if rng is None:
        rng = random.Random()

    all_indices = list(range(n_total))
    non_tumor_indices = [i for i in all_indices if i not in set(tumor_indices)]

    if not tumor_indices:
        # No tumor — just return all non-tumor slices
        return all_indices

    n_total_slices = max_slices or n_total
    n_tumor = min(int(n_total_slices * tumor_ratio), len(tumor_indices))
    n_non_tumor = min(n_total_slices - n_tumor, len(non_tumor_indices))

    selected_tumor = rng.sample(tumor_indices, n_tumor)
    selected_non_tumor = rng.sample(non_tumor_indices, n_non_tumor)

    return sorted(set(selected_tumor + selected_non_tumor))


# ──────────────────────────────────────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_slices_from_case(
    image_4ch: np.ndarray,
    seg_3ch: np.ndarray,
    patient_id: str,
    orientation: str = "axial",
    tumor_ratio: float = 0.6,
    max_slices: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    rng: Optional[random.Random] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray, SliceMetadata], None, None]:
    """Yield (image_slice, mask_slice, metadata) tuples for one patient.

    Called AFTER patient-level splitting to prevent data leakage.
    Slices are generated lazily (no full slice list allocated).

    Args:
        image_4ch: Image volume [4, H, W, D] float32 (T1, T1ce, T2, FLAIR).
        seg_3ch: Segmentation masks [3, H, W, D] float32 (WT, TC, ET).
        patient_id: Patient identifier string.
        orientation: "axial", "coronal", or "sagittal".
        tumor_ratio: Fraction of sampled slices containing tumor.
        max_slices: Max slices to yield per patient (None = all).
        target_size: Resize slices to (H, W). If None, no resize.
        rng: Optional seeded random.Random.

    Yields:
        Tuple of:
          - image_slice: [4, H', W'] float32
          - mask_slice: [3, H', W'] float32
          - metadata: SliceMetadata
    """
    vol_shape = image_4ch.shape[1:]  # (H, W, D)
    n_slices = _get_axis_length(vol_shape, orientation)

    tumor_indices = get_tumor_slice_indices(seg_3ch, orientation)
    selected = sample_slice_indices(
        n_slices, tumor_indices, tumor_ratio, max_slices, rng
    )

    for idx in selected:
        img_slice = _get_slice(image_4ch, idx, orientation)  # [4, H', W']
        seg_slice = _get_slice(seg_3ch, idx, orientation)    # [3, H', W']

        # Optional resize
        if target_size is not None:
            img_slice = _resize_slice(img_slice, target_size, order=1)
            seg_slice = _resize_slice(seg_slice, target_size, order=0)

        has_tumor = bool(seg_slice.any())
        wt_vox = int(seg_slice[0].sum())
        tc_vox = int(seg_slice[1].sum())
        et_vox = int(seg_slice[2].sum())

        meta = SliceMetadata(
            patient_id=patient_id,
            slice_index=idx,
            orientation=orientation,
            has_tumor=has_tumor,
            wt_voxels=wt_vox,
            tc_voxels=tc_vox,
            et_voxels=et_vox,
            volume_shape=vol_shape,
        )

        yield img_slice.astype(np.float32), seg_slice.astype(np.float32), meta


def _resize_slice(
    arr: np.ndarray,
    target_size: Tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize a [C, H, W] slice to target (H', W').

    Args:
        arr: [C, H, W] float32 array.
        target_size: Target (H', W').
        order: Interpolation order (1=linear, 0=nearest for masks).

    Returns:
        [C, H', W'] float32 array.
    """
    try:
        from skimage.transform import resize as sk_resize
        out = np.stack([
            sk_resize(arr[c], target_size, order=order,
                      preserve_range=True, anti_aliasing=(order > 0))
            for c in range(arr.shape[0])
        ], axis=0)
        return out.astype(np.float32)
    except ImportError:
        try:
            from scipy.ndimage import zoom
            h, w = arr.shape[1], arr.shape[2]
            th, tw = target_size
            factors = (1.0, th / h, tw / w)
            return zoom(arr, factors, order=order).astype(np.float32)
        except ImportError:
            raise ImportError(
                "scikit-image or scipy required for slice resizing. "
                "pip install scikit-image"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Volume loading helper
# ──────────────────────────────────────────────────────────────────────────────

def load_brats_volume_for_2d(
    patient_dir: Path,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load a BraTS patient directory into 4-channel image + 3-channel mask.

    Args:
        patient_dir: Path to patient directory with modality .nii.gz files.
        normalize: If True, apply per-channel Z-score normalization.

    Returns:
        Tuple of:
          - image [4, H, W, D] float32
          - seg   [3, H, W, D] float32 (WT, TC, ET)
          - meta  dict (affine, voxel_spacing, patient_id, shape)
    """
    try:
        import nibabel as nib
        from data.label_utils import brats_seg_to_regions, enforce_containment
        from preprocessing.nifti_loader import DETECTION_ORDER, _detect_modality
    except ImportError as exc:
        raise ImportError(f"Required module missing: {exc}")

    modality_order = ["t1", "t1ce", "t2", "flair"]
    channels = []
    affine = None
    voxel_spacing = None

    # Locate modality files
    modality_paths: Dict[str, Optional[Path]] = {m: None for m in modality_order}
    for f in sorted(patient_dir.glob("*.nii*")):
        if "seg" in f.name.lower():
            continue
        mod = _detect_modality(f.name)
        if mod and modality_paths.get(mod) is None:
            modality_paths[mod] = f

    for mod in modality_order:
        f = modality_paths[mod]
        if f is None:
            raise FileNotFoundError(
                f"Could not find '{mod}' modality in {patient_dir}"
            )
        img = nib.load(str(f))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        if normalize:
            nz = data != 0
            if nz.any():
                data[nz] = (data[nz] - data[nz].mean()) / (data[nz].std() + 1e-8)

        channels.append(data)
        if affine is None:
            affine = img.affine
            voxel_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)

    image = np.stack(channels, axis=0)  # [4, H, W, D]

    # Load segmentation
    seg_files = [f for f in patient_dir.glob("*.nii*") if "seg" in f.name.lower()]
    if not seg_files:
        raise FileNotFoundError(f"No segmentation file in {patient_dir}")
    seg_int = nib.load(str(seg_files[0])).get_fdata().astype(np.int32)
    wt, tc, et = brats_seg_to_regions(seg_int, enforce=True)
    seg = np.stack([wt, tc, et], axis=0).astype(np.float32)  # [3, H, W, D]

    meta = {
        "patient_id": patient_dir.name,
        "affine": affine,
        "voxel_spacing": voxel_spacing,
        "shape": image.shape[1:],
    }

    return image, seg, meta
