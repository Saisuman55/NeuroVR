"""
Tumor mask post-processing for NeuroVR 3D.

Applies connected-component analysis, morphological cleanup,
and optional minimum-size filtering to binary segmentation masks.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def filter_connected_components(
    mask: np.ndarray,
    min_size: int = 100,
    keep_largest: bool = False,
) -> Tuple[np.ndarray, int]:
    """Remove connected components below a minimum voxel count.

    Args:
        mask: Binary uint8 array (H, W, D).
        min_size: Components with fewer voxels are removed.
        keep_largest: If True, keep only the single largest component.

    Returns:
        Tuple of (filtered_mask, number_of_remaining_components).
    """
    try:
        from skimage.measure import label, regionprops
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required: pip install scikit-image"
        ) from exc

    if mask.sum() == 0:
        return mask.copy(), 0

    labeled = label(mask.astype(np.uint8), connectivity=3)
    regions = regionprops(labeled)

    if keep_largest:
        if not regions:
            return np.zeros_like(mask), 0
        largest = max(regions, key=lambda r: r.area)
        filtered = (labeled == largest.label).astype(np.uint8)
        return filtered, 1

    # Filter by minimum size
    keep_labels = [r.label for r in regions if r.area >= min_size]
    if not keep_labels:
        return np.zeros_like(mask), 0

    filtered = np.isin(labeled, keep_labels).astype(np.uint8)
    return filtered, len(keep_labels)


def morphological_cleanup(
    mask: np.ndarray,
    closing_radius: int = 2,
) -> np.ndarray:
    """Apply morphological closing to fill small holes in the mask.

    Args:
        mask: Binary uint8 array (H, W, D).
        closing_radius: Radius of the structuring element for closing.

    Returns:
        Cleaned binary mask.
    """
    if mask.sum() == 0:
        return mask.copy()

    try:
        from scipy.ndimage import binary_closing, generate_binary_structure, iterate_structure

        struct = generate_binary_structure(3, 1)
        struct = iterate_structure(struct, closing_radius)
        closed = binary_closing(mask.astype(bool), structure=struct)
        return closed.astype(np.uint8)
    except Exception as exc:
        print(f"[MaskProcessing] Morphological cleanup failed: {exc}. Returning original.")
        return mask.copy()


def postprocess_masks(
    tc_mask: np.ndarray,
    wt_mask: np.ndarray,
    et_mask: np.ndarray,
    min_component_size: int = 100,
    apply_closing: bool = True,
    closing_radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply full post-processing pipeline to all three tumor masks.

    Args:
        tc_mask: Tumor Core binary mask (H, W, D).
        wt_mask: Whole Tumor binary mask (H, W, D).
        et_mask: Enhancing Tumor binary mask (H, W, D).
        min_component_size: Minimum voxel count for a connected component.
        apply_closing: Whether to apply morphological closing.
        closing_radius: Structuring element radius for closing.

    Returns:
        Tuple of (tc_clean, wt_clean, et_clean) — all binary uint8 arrays.
    """
    results = []
    for name, mask in [("TC", tc_mask), ("WT", wt_mask), ("ET", et_mask)]:
        if mask.sum() == 0:
            results.append(mask.copy())
            print(f"[MaskProcessing] {name}: empty, skipping")
            continue

        # Morphological cleanup first
        if apply_closing:
            mask = morphological_cleanup(mask, closing_radius)

        # Connected component filtering
        mask, n_components = filter_connected_components(mask, min_size=min_component_size)
        print(f"[MaskProcessing] {name}: {int(mask.sum())} voxels, {n_components} components kept")
        results.append(mask)

    return tuple(results)  # type: ignore[return-value]
