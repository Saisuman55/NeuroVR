"""
MRI-to-Three.js Coordinate Mapper for NeuroVR.

Establishes and validates the complete coordinate chain:

    NIfTI voxel (i, j, k)
         ↓  affine matrix (4×4)
    RAS world space (mm)
         ↓  axis remap + unit scale
    Three.js scene (meters, Y-up)

This module makes the coordinate transform auditable: every step is
explicit and can be validated with round-trip tests. The transform
matrix is embedded in GLB exports so the frontend can reconstruct
real-world positions from voxel coordinates.

Coordinate conventions:
  - NIfTI RAS: Right=+X, Anterior=+Y, Superior=+Z (neurological)
  - Three.js:  Right=+X, Up=+Y, Forward=-Z (Y-up right-handed)
  - Scale:     1 mm → 0.001 m (Three.js uses meters)

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Scale factor: NIfTI (mm) → Three.js (meters)
MM_TO_METERS = 0.001

# RAS-to-Three.js axis remap matrix:
# Three.js: X=right(R), Y=up(S), Z=-anterior(-A)
# RAS:      X=right(R), Y=anterior(A), Z=superior(S)
# So:  ThreeX = R = RAS_X
#      ThreeY = S = RAS_Z
#      ThreeZ = -A = -RAS_Y
_RAS_TO_THREEJS_REMAP = np.array([
    [1,  0,  0, 0],   # ThreeX = RAS_X (Right)
    [0,  0,  1, 0],   # ThreeY = RAS_Z (Superior → Up)
    [0, -1,  0, 0],   # ThreeZ = -RAS_Y (Anterior → -Forward)
    [0,  0,  0, 1],
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Main coordinate mapper
# ──────────────────────────────────────────────────────────────────────────────

class RASToThreeJS:
    """Transform voxel coordinates through RAS space to Three.js scene units.

    Usage::

        mapper = RASToThreeJS(affine=nib_image.affine)
        ras_mm = mapper.voxel_to_ras([128, 128, 90])
        threejs_pos = mapper.ras_to_threejs(ras_mm)
        # Or in one step:
        threejs_pos = mapper.voxel_to_threejs([128, 128, 90])
    """

    def __init__(self, affine: np.ndarray) -> None:
        """
        Args:
            affine: NIfTI 4×4 voxel-to-world affine matrix.
        """
        if affine.shape != (4, 4):
            raise ValueError(f"Affine must be 4×4, got {affine.shape}")
        self._affine = affine.astype(np.float64)
        self._affine_inv = np.linalg.inv(self._affine)

        # Precompute the full combined transform
        scale = np.eye(4, dtype=np.float64)
        scale[:3, :3] *= MM_TO_METERS

        # Combined: voxel → RAS_mm → RAS_m → Three.js
        self._full_transform = _RAS_TO_THREEJS_REMAP @ scale @ self._affine

    # ── Forward transforms ───────────────────────────────────────────────────

    def voxel_to_ras(
        self,
        voxel_coords: np.ndarray,
    ) -> np.ndarray:
        """Transform voxel index coordinates to RAS world coordinates (mm).

        Args:
            voxel_coords: Array of shape (3,) or (N, 3) — voxel (i, j, k).

        Returns:
            RAS coordinates in mm, same leading shape.
        """
        coords = np.asarray(voxel_coords, dtype=np.float64)
        scalar = coords.ndim == 1
        if scalar:
            coords = coords[np.newaxis, :]

        # Homogeneous coordinates
        ones = np.ones((len(coords), 1))
        hom = np.concatenate([coords, ones], axis=1)  # (N, 4)
        ras = (self._affine @ hom.T).T[:, :3]          # (N, 3)

        return ras[0] if scalar else ras

    def ras_to_threejs(
        self,
        ras_mm: np.ndarray,
    ) -> np.ndarray:
        """Transform RAS world coordinates (mm) to Three.js scene units (m).

        Applies:
          1. Scale mm → meters
          2. Axis remap (RAS → Three.js Y-up right-handed)

        Args:
            ras_mm: Array (3,) or (N, 3) in mm.

        Returns:
            Three.js coordinates in meters.
        """
        coords = np.asarray(ras_mm, dtype=np.float64)
        scalar = coords.ndim == 1
        if scalar:
            coords = coords[np.newaxis, :]

        # Scale + remap
        scaled = coords * MM_TO_METERS  # mm → m
        ones = np.ones((len(scaled), 1))
        hom = np.concatenate([scaled, ones], axis=1)
        threejs = (_RAS_TO_THREEJS_REMAP @ hom.T).T[:, :3]

        return threejs[0] if scalar else threejs

    def voxel_to_threejs(
        self,
        voxel_coords: np.ndarray,
    ) -> np.ndarray:
        """Transform voxel coordinates directly to Three.js scene units (m).

        Args:
            voxel_coords: Array (3,) or (N, 3) — voxel (i, j, k).

        Returns:
            Three.js coordinates in meters.
        """
        coords = np.asarray(voxel_coords, dtype=np.float64)
        scalar = coords.ndim == 1
        if scalar:
            coords = coords[np.newaxis, :]

        ones = np.ones((len(coords), 1))
        hom = np.concatenate([coords, ones], axis=1)
        threejs = (self._full_transform @ hom.T).T[:, :3]

        return threejs[0] if scalar else threejs

    # ── Inverse transform ────────────────────────────────────────────────────

    def threejs_to_voxel(
        self,
        threejs_coords: np.ndarray,
    ) -> np.ndarray:
        """Inverse: Three.js coordinates (m) → voxel (i, j, k).

        Args:
            threejs_coords: Array (3,) or (N, 3) in meters.

        Returns:
            Voxel coordinates.
        """
        coords = np.asarray(threejs_coords, dtype=np.float64)
        scalar = coords.ndim == 1
        if scalar:
            coords = coords[np.newaxis, :]

        # Invert axis remap (remap matrix is orthogonal, so inverse = transpose)
        remap_inv = _RAS_TO_THREEJS_REMAP[:3, :3].T
        ras_m = (remap_inv @ coords.T).T          # (N, 3) in meters
        ras_mm = ras_m / MM_TO_METERS             # meters → mm

        ones = np.ones((len(ras_mm), 1))
        hom = np.concatenate([ras_mm, ones], axis=1)
        voxels = (self._affine_inv @ hom.T).T[:, :3]

        return voxels[0] if scalar else voxels

    # ── Transform matrix for GPU/shader use ──────────────────────────────────

    def get_transform_matrix(self) -> np.ndarray:
        """Return the full 4×4 voxel→Three.js transform matrix.

        This matrix can be passed to a Three.js vertex shader or stored
        in a GLB file's extras for the frontend to use.

        Returns:
            4×4 float64 matrix.
        """
        return self._full_transform.copy()

    def get_metadata(self) -> dict:
        """Return all coordinate transform metadata as a JSON-serializable dict."""
        return {
            "affine": self._affine.tolist(),
            "full_transform_voxel_to_threejs": self._full_transform.tolist(),
            "mm_to_meters": MM_TO_METERS,
            "coordinate_conventions": {
                "nifti": "RAS (Right=+X, Anterior=+Y, Superior=+Z)",
                "threejs": "Y-up right-handed (Right=+X, Up=+Y, Forward=-Z)",
                "units_input": "mm (from NIfTI header)",
                "units_output": "meters (Three.js convention)",
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate chain validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_coordinate_chain(
    affine: np.ndarray,
    test_voxels: Optional[np.ndarray] = None,
    tolerance_mm: float = 0.1,
) -> dict:
    """Validate the round-trip consistency of the coordinate transform chain.

    Checks: voxel → Three.js → voxel ≈ identity.

    Args:
        affine: NIfTI 4×4 affine.
        test_voxels: Array (N, 3) of voxel coords to test. If None,
            uses a standard set of test points.
        tolerance_mm: Acceptable round-trip error in mm.

    Returns:
        Dict with "valid", "max_error_mm", "test_points", "errors".
    """
    mapper = RASToThreeJS(affine)

    if test_voxels is None:
        # Standard test points: corners + center of a 240³ volume
        test_voxels = np.array([
            [0, 0, 0],
            [240, 0, 0],
            [0, 240, 0],
            [0, 0, 240],
            [120, 120, 120],
            [64, 64, 90],
        ], dtype=np.float64)

    threejs = mapper.voxel_to_threejs(test_voxels)
    recovered = mapper.threejs_to_voxel(threejs)

    errors_vox = np.abs(recovered - test_voxels)
    errors_mm = errors_vox * np.abs(np.diag(affine[:3, :3]))  # vox → mm
    max_error = float(errors_mm.max())

    return {
        "valid": max_error < tolerance_mm,
        "max_error_mm": round(max_error, 6),
        "tolerance_mm": tolerance_mm,
        "n_test_points": len(test_voxels),
        "transform_matrix": mapper.get_transform_matrix().tolist(),
    }


def validate_affine_consistency(
    affine_list: List[np.ndarray],
    modality_names: Optional[List[str]] = None,
    rtol: float = 1e-4,
    atol: float = 1e-2,
) -> dict:
    """Check that multiple modality affines share the same coordinate system.

    All 4 BraTS modalities must be registered — their affines must match.
    This is a prerequisite for valid multi-modal segmentation.

    Args:
        affine_list: List of 4×4 affine matrices (one per modality).
        modality_names: Optional labels for error messages.
        rtol: Relative tolerance for numpy.allclose.
        atol: Absolute tolerance (mm).

    Returns:
        Dict with "valid", "reference_modality", "mismatched_modalities".
    """
    if not affine_list:
        return {"valid": True, "reference_modality": None, "mismatched_modalities": []}

    names = modality_names or [f"modality_{i}" for i in range(len(affine_list))]
    reference = affine_list[0]
    mismatched = []

    for i, (aff, name) in enumerate(zip(affine_list[1:], names[1:]), start=1):
        if not np.allclose(aff, reference, rtol=rtol, atol=atol):
            max_diff = float(np.abs(aff - reference).max())
            mismatched.append({
                "modality": name,
                "index": i,
                "max_affine_difference": round(max_diff, 6),
            })

    return {
        "valid": len(mismatched) == 0,
        "reference_modality": names[0],
        "mismatched_modalities": mismatched,
        "n_checked": len(affine_list),
    }
