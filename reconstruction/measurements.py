"""
Physical tumor measurements for NeuroVR 3D.

Calculates clinically-relevant dimensional metrics from binary
segmentation masks using physical voxel spacing (mm).

All measurements are AI-derived from segmentation output and are
clearly labeled as research estimates — not clinical measurements.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def calculate_volume_cm3(
    mask: np.ndarray,
    voxel_spacing: np.ndarray,
) -> float:
    """Calculate physical volume in cubic centimetres.

    Args:
        mask: Binary mask (H, W, D).
        voxel_spacing: [dx, dy, dz] in mm.

    Returns:
        Volume in cm³. Returns 0.0 if mask is empty.
    """
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return 0.0

    # Volume per voxel in mm³ → cm³
    voxel_vol_mm3 = float(voxel_spacing[0]) * float(voxel_spacing[1]) * float(voxel_spacing[2])
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    return round(voxel_count * voxel_vol_cm3, 4)


def calculate_bounding_box_mm(
    mask: np.ndarray,
    voxel_spacing: np.ndarray,
) -> Optional[Dict]:
    """Calculate the axis-aligned bounding box in physical mm coordinates.

    Args:
        mask: Binary mask (H, W, D).
        voxel_spacing: [dx, dy, dz] in mm.

    Returns:
        Dict with keys 'width_mm', 'height_mm', 'depth_mm', 'voxel_bbox',
        or None if mask is empty.
    """
    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    extents_vox = max_coords - min_coords + 1  # +1 for inclusive bounds

    extents_mm = extents_vox * voxel_spacing

    return {
        "width_mm":  round(float(extents_mm[0]), 2),
        "height_mm": round(float(extents_mm[1]), 2),
        "depth_mm":  round(float(extents_mm[2]), 2),
        "voxel_bbox": {
            "min": min_coords.tolist(),
            "max": max_coords.tolist(),
            "extent_vox": extents_vox.tolist(),
        },
    }


def calculate_centroid_mm(
    mask: np.ndarray,
    voxel_spacing: np.ndarray,
    affine: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Calculate the centroid of a mask in voxel and physical coordinates.

    Args:
        mask: Binary mask (H, W, D).
        voxel_spacing: [dx, dy, dz] in mm.
        affine: Optional 4×4 affine to transform centroid to world/RAS coords.

    Returns:
        Dict with centroid in voxel and mm coordinates.
    """
    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask)
    centroid_vox = coords.mean(axis=0)
    centroid_mm = centroid_vox * voxel_spacing

    result = {
        "centroid_voxel": [round(float(v), 2) for v in centroid_vox],
        "centroid_mm": [round(float(v), 2) for v in centroid_mm],
    }

    # If affine provided, compute world (RAS) coordinates
    if affine is not None:
        try:
            vox_hom = np.append(centroid_vox, 1.0)
            world = affine @ vox_hom
            result["centroid_ras_mm"] = [round(float(v), 2) for v in world[:3]]
        except Exception:
            pass

    return result


def measure_region(
    mask: np.ndarray,
    voxel_spacing: np.ndarray,
    region_name: str,
    affine: Optional[np.ndarray] = None,
) -> Dict:
    """Compute all measurements for one tumor region.

    Args:
        mask: Binary mask (H, W, D).
        voxel_spacing: [dx, dy, dz] in mm.
        region_name: Descriptive name (e.g., 'Whole Tumor').
        affine: Optional 4×4 affine for world-space coordinates.

    Returns:
        Dict with all measurements for the region.
    """
    voxel_count = int(mask.sum())
    volume_cm3 = calculate_volume_cm3(mask, voxel_spacing)
    bbox = calculate_bounding_box_mm(mask, voxel_spacing)
    centroid = calculate_centroid_mm(mask, voxel_spacing, affine)

    return {
        "region": region_name,
        "detected": voxel_count > 0,
        "voxel_count": voxel_count,
        "volume_cm3": volume_cm3,
        "bounding_box_mm": bbox,
        "centroid": centroid,
    }


def compute_all_measurements(
    tc_mask: np.ndarray,
    wt_mask: np.ndarray,
    et_mask: np.ndarray,
    voxel_spacing: np.ndarray,
    affine: Optional[np.ndarray] = None,
    inference_time_s: float = 0.0,
    model_name: str = "MONAI BraTS MRI Segmentation",
) -> Dict:
    """Compute physical measurements for all three BraTS tumor regions.

    Args:
        tc_mask: Tumor Core binary mask (H, W, D).
        wt_mask: Whole Tumor binary mask (H, W, D).
        et_mask: Enhancing Tumor binary mask (H, W, D).
        voxel_spacing: [dx, dy, dz] in mm.
        affine: Optional 4×4 affine for world-space centroid.
        inference_time_s: Time taken for inference (for UI display).
        model_name: Name of the model used.

    Returns:
        Comprehensive measurements dict suitable for JSON serialization.
    """
    wt_meas = measure_region(wt_mask, voxel_spacing, "Whole Tumor", affine)
    tc_meas = measure_region(tc_mask, voxel_spacing, "Tumor Core", affine)
    et_meas = measure_region(et_mask, voxel_spacing, "Enhancing Tumor", affine)

    tumor_detected = wt_meas["detected"]

    return {
        "disclaimer": (
            "AI-derived research measurements from automated segmentation. "
            "Not for clinical diagnosis. Consult a qualified radiologist."
        ),
        "tumor_detected": tumor_detected,
        "model": model_name,
        "voxel_spacing_mm": [round(float(v), 4) for v in voxel_spacing],
        "inference_time_s": round(inference_time_s, 2),
        "regions": {
            "whole_tumor": wt_meas,
            "tumor_core": tc_meas,
            "enhancing_tumor": et_meas,
        },
        "summary": _format_summary(wt_meas, tc_meas, et_meas),
    }


def _format_summary(wt: Dict, tc: Dict, et: Dict) -> Dict:
    """Create a flat summary for easy display in the frontend."""
    return {
        "whole_tumor_volume_cm3": wt["volume_cm3"],
        "tumor_core_volume_cm3": tc["volume_cm3"],
        "enhancing_tumor_volume_cm3": et["volume_cm3"],
        "whole_tumor_detected": wt["detected"],
        "tumor_core_detected": tc["detected"],
        "enhancing_tumor_detected": et["detected"],
        "wt_bbox_mm": wt["bounding_box_mm"],
        "wt_centroid": wt["centroid"],
    }
