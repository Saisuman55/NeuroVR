"""
BraTS Label Utilities for NeuroVR.

Converts BraTS integer segmentation labels to binary masks for
Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET),
enforces and validates the anatomical containment invariant:

    ET ⊆ TC ⊆ WT

This invariant reflects the biological reality that:
  - Enhancing tumor is a subset of the tumor core
  - Tumor core is a subset of the whole tumor

Research prototype — NOT for clinical diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# BraTS label constants
# ──────────────────────────────────────────────────────────────────────────────

# BraTS 2020 / MSD Task01 integer labels
LABEL_BACKGROUND = 0
LABEL_NCR_NET = 1   # Necrotic and Non-Enhancing Tumor Core
LABEL_EDEMA = 2     # Peritumoral Edema
LABEL_ET = 4        # Enhancing Tumor (label 3 in some older BraTS versions)

# BraTS 2018 compatibility (some datasets use label 3 instead of 4 for ET)
LABEL_ET_COMPAT = 3


class ContainmentViolationError(ValueError):
    """Raised when the ET ⊆ TC ⊆ WT invariant is violated and cannot be corrected."""
    pass


@dataclass
class ContainmentReport:
    """Result of a containment validation check."""
    valid: bool
    et_outside_tc_voxels: int
    tc_outside_wt_voxels: int
    et_voxels: int
    tc_voxels: int
    wt_voxels: int

    @property
    def summary(self) -> str:
        if self.valid:
            return f"Containment OK | ET={self.et_voxels} ⊆ TC={self.tc_voxels} ⊆ WT={self.wt_voxels}"
        return (
            f"Containment VIOLATED | "
            f"ET outside TC: {self.et_outside_tc_voxels} vox | "
            f"TC outside WT: {self.tc_outside_wt_voxels} vox"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Label → binary mask conversion
# ──────────────────────────────────────────────────────────────────────────────

def create_whole_tumor_mask(seg: np.ndarray) -> np.ndarray:
    """Whole Tumor = all non-background labels (NCR + Edema + ET).

    BraTS convention: WT = {1, 2, 4}.

    Args:
        seg: Integer segmentation volume (H, W, D).

    Returns:
        Binary uint8 mask, 1 where any tumor label present.
    """
    return ((seg == LABEL_NCR_NET) |
            (seg == LABEL_EDEMA) |
            (seg == LABEL_ET) |
            (seg == LABEL_ET_COMPAT)).astype(np.uint8)


def create_tumor_core_mask(seg: np.ndarray) -> np.ndarray:
    """Tumor Core = Necrotic core + Enhancing Tumor (excludes edema).

    BraTS convention: TC = {1, 4}.

    Args:
        seg: Integer segmentation volume (H, W, D).

    Returns:
        Binary uint8 mask.
    """
    return ((seg == LABEL_NCR_NET) |
            (seg == LABEL_ET) |
            (seg == LABEL_ET_COMPAT)).astype(np.uint8)


def create_enhancing_tumor_mask(seg: np.ndarray) -> np.ndarray:
    """Enhancing Tumor = actively enhancing gadolinium-enhancing tissue.

    BraTS convention: ET = {4}.

    Args:
        seg: Integer segmentation volume (H, W, D).

    Returns:
        Binary uint8 mask.
    """
    return ((seg == LABEL_ET) | (seg == LABEL_ET_COMPAT)).astype(np.uint8)


def brats_seg_to_regions(
    seg: np.ndarray,
    enforce: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BraTS integer segmentation to three binary region masks.

    Args:
        seg: Integer segmentation volume (H, W, D) with BraTS labels.
        enforce: If True, enforce containment after extraction. This
            ensures ET ⊆ TC ⊆ WT even if the source has minor label errors.

    Returns:
        Tuple of (wt_mask, tc_mask, et_mask) — all binary uint8 (H, W, D).
    """
    wt = create_whole_tumor_mask(seg)
    tc = create_tumor_core_mask(seg)
    et = create_enhancing_tumor_mask(seg)

    if enforce:
        wt, tc, et = enforce_containment(wt, tc, et)

    return wt, tc, et


# ──────────────────────────────────────────────────────────────────────────────
# Containment enforcement & validation
# ──────────────────────────────────────────────────────────────────────────────

def enforce_containment(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enforce the biological containment invariant: ET ⊆ TC ⊆ WT.

    Corrects violations by masking:
      - ET = ET ∩ TC  (any ET outside TC is removed)
      - TC = TC ∪ ET  (TC must include all ET)
      - WT = WT ∪ TC  (WT must include all TC)

    This is the correct biological fix: inner regions can only exist within
    outer regions, not the other way around. WT is never shrunk.

    Args:
        wt: Whole Tumor binary mask.
        tc: Tumor Core binary mask.
        et: Enhancing Tumor binary mask.

    Returns:
        Tuple of corrected (wt, tc, et) binary uint8 masks.
    """
    et_arr = et.astype(bool)
    tc_arr = tc.astype(bool)
    wt_arr = wt.astype(bool)

    # TC must contain ET
    tc_arr = tc_arr | et_arr
    # WT must contain TC
    wt_arr = wt_arr | tc_arr

    # ET must be within TC (already guaranteed above, but make explicit)
    et_arr = et_arr & tc_arr

    return (
        wt_arr.astype(np.uint8),
        tc_arr.astype(np.uint8),
        et_arr.astype(np.uint8),
    )


def validate_containment(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    raise_on_violation: bool = False,
) -> ContainmentReport:
    """Check that ET ⊆ TC ⊆ WT.

    Args:
        wt: Whole Tumor binary mask.
        tc: Tumor Core binary mask.
        et: Enhancing Tumor binary mask.
        raise_on_violation: If True, raise ContainmentViolationError when violated.

    Returns:
        ContainmentReport with violation counts and validity flag.
    """
    et_arr = et.astype(bool)
    tc_arr = tc.astype(bool)
    wt_arr = wt.astype(bool)

    et_outside_tc = int(np.count_nonzero(et_arr & ~tc_arr))
    tc_outside_wt = int(np.count_nonzero(tc_arr & ~wt_arr))

    valid = (et_outside_tc == 0) and (tc_outside_wt == 0)

    report = ContainmentReport(
        valid=valid,
        et_outside_tc_voxels=et_outside_tc,
        tc_outside_wt_voxels=tc_outside_wt,
        et_voxels=int(et_arr.sum()),
        tc_voxels=int(tc_arr.sum()),
        wt_voxels=int(wt_arr.sum()),
    )

    if not valid and raise_on_violation:
        raise ContainmentViolationError(report.summary)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Multi-channel tensor conversion (for model I/O)
# ──────────────────────────────────────────────────────────────────────────────

def masks_to_multichannel(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
) -> np.ndarray:
    """Stack three binary masks into a 3-channel array [3, H, W, D].

    Channel order: [WT, TC, ET] — matches MONAI BraTS convention.

    Args:
        wt: Whole Tumor mask.
        tc: Tumor Core mask.
        et: Enhancing Tumor mask.

    Returns:
        Float32 array [3, H, W, D].
    """
    return np.stack([
        wt.astype(np.float32),
        tc.astype(np.float32),
        et.astype(np.float32),
    ], axis=0)


def multichannel_to_masks(
    multichannel: np.ndarray,
    threshold: float = 0.5,
    enforce: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a 3-channel prediction array back to binary masks.

    Args:
        multichannel: Float32 array [3, H, W, D] (sigmoid probabilities).
        threshold: Binarization threshold.
        enforce: If True, enforce containment after thresholding.

    Returns:
        Tuple of (wt_mask, tc_mask, et_mask) binary uint8 arrays.
    """
    wt = (multichannel[0] > threshold).astype(np.uint8)
    tc = (multichannel[1] > threshold).astype(np.uint8)
    et = (multichannel[2] > threshold).astype(np.uint8)

    if enforce:
        wt, tc, et = enforce_containment(wt, tc, et)

    return wt, tc, et


def regions_to_brats_seg(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
) -> np.ndarray:
    """Reconstruct BraTS integer segmentation from three binary masks.

    Used for evaluation round-trips and saving predictions in BraTS format.

    Args:
        wt: Whole Tumor mask.
        tc: Tumor Core mask.
        et: Enhancing Tumor mask.

    Returns:
        Integer array with BraTS labels (0=bg, 1=NCR, 2=edema, 4=ET).
    """
    seg = np.zeros(wt.shape, dtype=np.uint8)
    # Order matters: apply from outer to inner (WT -> TC -> ET)
    seg[wt.astype(bool)] = LABEL_EDEMA      # Edema fills the WT region
    seg[tc.astype(bool)] = LABEL_NCR_NET    # NCR/NET fills the TC region
    seg[et.astype(bool)] = LABEL_ET         # ET fills the ET region
    return seg
