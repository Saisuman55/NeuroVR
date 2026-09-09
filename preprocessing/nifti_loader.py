"""
NIfTI volume loader for NeuroVR 3D.

Loads and validates NIfTI (.nii / .nii.gz) MRI files for multi-modal
BraTS-style analysis. Preserves affine transform and voxel spacing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

# Expected BraTS modality file suffixes (case-insensitive).
# IMPORTANT: t1ce must be listed and checked BEFORE t1 to prevent
# "case_t1ce.nii.gz" from being matched as t1 (prefix collision).
MODALITY_ALIASES: Dict[str, List[str]] = {
    "t1ce":  ["t1ce", "t1c.", "t1c_", "_t1c.", "t1gd", "t1_ce", "t1_c.", "t1+c", "ce_t1"],
    "t1":    ["_t1.", "_t1_", "_t1n.", "t1.nii", "t1n.nii", "/t1."],
    "t2":    ["_t2.", "_t2_", "t2.nii", "/t2."],
    "flair": ["flair", "_flair.", "t2flair", "t2_flair"],
}

# Detection order: t1ce before t1 to avoid prefix collision
DETECTION_ORDER = ["t1ce", "t1", "t2", "flair"]

REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]
VALID_EXTENSIONS = {".nii", ".gz"}


def _detect_modality(filename: str) -> Optional[str]:
    """Heuristically detect which BraTS modality a filename represents.

    IMPORTANT: Uses DETECTION_ORDER to check t1ce before t1, preventing
    filenames like 'case_t1ce.nii.gz' from being matched as 't1'.
    """
    name = filename.lower()
    # Remove extension for cleaner matching
    stem = name.replace(".nii.gz", "").replace(".nii", "")

    # Iterate in priority order: t1ce first, then t1
    for modality in DETECTION_ORDER:
        aliases = MODALITY_ALIASES[modality]
        for alias in aliases:
            if alias.lower() in stem:
                return modality

    # Fallback: for 'flair' and 't2' which are unambiguous
    if "flair" in stem or "t2flair" in stem:
        return "flair"
    if stem.endswith("_t2") or "_t2_" in stem or stem == "t2":
        return "t2"
    if stem.endswith("_t1ce") or stem.endswith("_t1c") or "t1ce" in stem:
        return "t1ce"
    if stem.endswith("_t1") or stem == "t1":
        return "t1"
    return None


def validate_nifti_file(path: str) -> Tuple[bool, str]:
    """Check that a file is a valid, readable NIfTI volume.

    Args:
        path: Absolute path to the file.

    Returns:
        (is_valid, message)
    """
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    if p.suffix not in VALID_EXTENSIONS and p.suffixes[-2:] != [".nii", ".gz"]:
        return False, f"Not a NIfTI file (expected .nii or .nii.gz): {p.name}"
    try:
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) < 3:
            return False, f"Volume must be at least 3D; got shape {shape}"
        if len(shape) == 4 and shape[3] != 1:
            # 4D volumes are accepted but we only use the first volume
            pass
        return True, "OK"
    except Exception as exc:
        return False, f"Could not load NIfTI: {exc}"


def load_nifti_volume(
    path: str,
    ensure_3d: bool = True,
) -> Dict:
    """Load a single NIfTI file and return its data + metadata.

    Args:
        path: Path to .nii or .nii.gz file.
        ensure_3d: If True, drop any 4th dimension (take first volume).

    Returns:
        Dict with keys:
          - data: np.ndarray, float32, shape (H, W, D)
          - affine: np.ndarray (4×4) world-space transform
          - voxel_spacing: np.ndarray [dx, dy, dz] in mm
          - shape: tuple (H, W, D)
          - path: str
          - filename: str
    """
    valid, msg = validate_nifti_file(path)
    if not valid:
        raise ValueError(f"Invalid NIfTI file: {msg}")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    # Collapse 4D → 3D
    if ensure_3d and data.ndim == 4:
        data = data[..., 0]

    affine = img.affine
    # Voxel sizes from header
    voxel_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)

    return {
        "data": data,
        "affine": affine,
        "voxel_spacing": voxel_spacing,
        "shape": data.shape,
        "path": str(path),
        "filename": os.path.basename(path),
    }


def load_brats_case(
    modality_paths: Dict[str, str],
) -> Dict[str, Dict]:
    """Load all four BraTS modalities from explicit file paths.

    Args:
        modality_paths: Dict mapping modality name to file path.
            Required keys: 't1', 't1ce', 't2', 'flair'.

    Returns:
        Dict mapping modality → volume dict (same structure as load_nifti_volume).

    Raises:
        ValueError: If any required modality is missing or invalid.
    """
    missing = [m for m in REQUIRED_MODALITIES if m not in modality_paths]
    if missing:
        raise ValueError(
            f"Missing required modalities: {missing}. "
            f"Provided: {list(modality_paths.keys())}"
        )

    volumes = {}
    for modality in REQUIRED_MODALITIES:
        path = modality_paths[modality]
        try:
            volumes[modality] = load_nifti_volume(path)
        except Exception as exc:
            raise ValueError(
                f"Failed to load modality '{modality}' from '{path}': {exc}"
            ) from exc

    # Validate that all volumes share the same spatial shape
    shapes = {m: v["shape"] for m, v in volumes.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(
            f"All modalities must have identical spatial shapes. "
            f"Found: {shapes}"
        )

    return volumes


def auto_detect_modalities(directory: str) -> Dict[str, str]:
    """Scan a directory and auto-detect which files correspond to each modality.

    Args:
        directory: Path to directory containing NIfTI files.

    Returns:
        Dict mapping modality name → file path.

    Raises:
        ValueError: If any required modality cannot be found.
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    candidates: List[Path] = []
    for ext in ("*.nii", "*.nii.gz"):
        candidates.extend(dirpath.glob(ext))

    detected: Dict[str, str] = {}
    for fpath in candidates:
        modality = _detect_modality(fpath.name)
        if modality and modality not in detected:
            detected[modality] = str(fpath)

    missing = [m for m in REQUIRED_MODALITIES if m not in detected]
    if missing:
        raise ValueError(
            f"Could not auto-detect modalities: {missing}. "
            f"Files found: {[c.name for c in candidates]}. "
            "Provide explicit modality paths instead."
        )

    return detected
