"""
DICOM-to-NIfTI adapter for NeuroVR 3D.

Converts DICOM series (one modality per series) into NIfTI volumes
that are compatible with the NIfTI loader pipeline.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]


def _check_pydicom() -> None:
    """Raise ImportError with install hint if pydicom is missing."""
    try:
        import pydicom  # noqa
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install pydicom"
        ) from exc


def _check_nibabel() -> None:
    try:
        import nibabel  # noqa
    except ImportError as exc:
        raise ImportError(
            "nibabel is required. pip install nibabel"
        ) from exc


def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a DICOM series directory into a 3D numpy volume.

    Slices are sorted by InstanceNumber or ImagePositionPatient Z.

    Args:
        series_dir: Directory containing .dcm files for a single series.

    Returns:
        Tuple of:
          - volume: np.ndarray float32, shape (H, W, D)
          - affine: np.ndarray (4×4)
          - voxel_spacing: np.ndarray [dx, dy, dz] mm
    """
    _check_pydicom()
    import pydicom

    series_path = Path(series_dir)
    dcm_files = sorted(series_path.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(series_path.glob("*.DCM"))
    if not dcm_files:
        # Some DICOM files have no extension
        dcm_files = [p for p in series_path.iterdir() if p.is_file()]

    if not dcm_files:
        raise ValueError(f"No DICOM files found in: {series_dir}")

    # Read all slices
    slices: List[pydicom.Dataset] = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM slices loaded from: {series_dir}")

    # Sort by ImagePositionPatient Z, fallback to InstanceNumber
    def _sort_key(ds: pydicom.Dataset) -> float:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            try:
                return float(ds.InstanceNumber)
            except Exception:
                return 0.0

    slices.sort(key=_sort_key)

    # Build 3D volume: (H, W, D)
    pixel_arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=-1)  # (H, W, D)

    # Extract voxel spacing
    first = slices[0]
    try:
        px_spacing = [float(v) for v in first.PixelSpacing]
    except Exception:
        px_spacing = [1.0, 1.0]

    if len(slices) > 1:
        try:
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            slice_thickness = abs(z1 - z0)
        except Exception:
            try:
                slice_thickness = float(first.SliceThickness)
            except Exception:
                slice_thickness = 1.0
    else:
        try:
            slice_thickness = float(first.SliceThickness)
        except Exception:
            slice_thickness = 1.0

    voxel_spacing = np.array(
        [px_spacing[0], px_spacing[1], slice_thickness], dtype=np.float32
    )

    # Build a simple affine from ImagePositionPatient + ImageOrientationPatient
    affine = _build_affine(first, voxel_spacing)

    return volume, affine, voxel_spacing


def _build_affine(
    ds,
    voxel_spacing: np.ndarray,
) -> np.ndarray:
    """Construct a 4×4 affine from DICOM position/orientation tags."""
    try:
        import pydicom

        F = [float(x) for x in ds.ImageOrientationPatient]
        T = [float(x) for x in ds.ImagePositionPatient]
        dr, dc = float(voxel_spacing[0]), float(voxel_spacing[1])
        # Row + column cosines scaled by pixel spacing
        F11, F21, F31 = F[0] * dr, F[1] * dr, F[2] * dr
        F12, F22, F32 = F[3] * dc, F[4] * dc, F[5] * dc
        # Normal vector (cross product) × slice thickness
        import math

        n1 = F21 * F32 - F31 * F22
        n2 = F31 * F12 - F11 * F32
        n3 = F11 * F22 - F21 * F12
        length = math.sqrt(n1**2 + n2**2 + n3**2) or 1.0
        dz = float(voxel_spacing[2])
        affine = np.array(
            [
                [F11, F12, n1 / length * dz, T[0]],
                [F21, F22, n2 / length * dz, T[1]],
                [F31, F32, n3 / length * dz, T[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    except Exception:
        # Fallback: identity-scaled affine
        affine = np.diag(
            [float(voxel_spacing[0]), float(voxel_spacing[1]), float(voxel_spacing[2]), 1.0]
        )
    return affine


def dicom_to_nifti(series_dir: str, output_path: Optional[str] = None) -> str:
    """Convert a DICOM series directory to a NIfTI file.

    Args:
        series_dir: Path to DICOM series directory.
        output_path: Where to save the .nii.gz file.
                     Defaults to a temp file.

    Returns:
        Path to the generated .nii.gz file.
    """
    _check_pydicom()
    _check_nibabel()
    import nibabel as nib

    volume, affine, voxel_spacing = load_dicom_series(series_dir)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".nii.gz")
        os.close(fd)

    img = nib.Nifti1Image(volume, affine)
    img.header.set_zooms(voxel_spacing)
    nib.save(img, output_path)
    print(f"[DICOM] Saved NIfTI: {output_path}  shape={volume.shape}  spacing={voxel_spacing}")
    return output_path


def convert_dicom_modalities(
    modality_dirs: Dict[str, str],
    output_dir: str,
) -> Dict[str, str]:
    """Convert multiple DICOM series (one per modality) to NIfTI files.

    Args:
        modality_dirs: Dict mapping modality name → DICOM series directory.
        output_dir: Directory where .nii.gz files will be written.

    Returns:
        Dict mapping modality name → path to .nii.gz file.
    """
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, str] = {}

    for modality, series_dir in modality_dirs.items():
        out_path = os.path.join(output_dir, f"{modality}.nii.gz")
        converted = dicom_to_nifti(series_dir, out_path)
        result[modality] = converted
        print(f"[DICOM] Converted {modality}: {series_dir} → {converted}")

    return result
