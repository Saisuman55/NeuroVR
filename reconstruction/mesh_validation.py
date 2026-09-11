"""
Mesh-Level Containment Validation for NeuroVR.

After 3D mesh generation, validates that the biological containment
invariant (ET ⊆ TC ⊆ WT) is preserved in the mesh geometry:

  1. Bounding box containment: ET bbox ⊆ TC bbox ⊆ WT bbox
  2. Centroid containment: ET centroid lies within TC mesh bounds
  3. Brain containment: all tumor meshes within brain mesh bounds
  4. Mesh statistics: vertex count, face count, volume, surface area

Returns structured validation reports suitable for JSON serialization
and frontend display.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Bounding box utilities
# ──────────────────────────────────────────────────────────────────────────────

def _get_mesh_bbox(mesh) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get (min_bounds, max_bounds) of a trimesh.Trimesh.

    Returns None if the mesh is None or empty.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return None
    bounds = mesh.bounds  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    return bounds[0], bounds[1]


def _bbox_contains_bbox(
    outer_min: np.ndarray,
    outer_max: np.ndarray,
    inner_min: np.ndarray,
    inner_max: np.ndarray,
    tolerance: float = 1e-3,
) -> bool:
    """Check that inner bounding box is contained within outer bounding box.

    Args:
        outer_min, outer_max: Outer box bounds.
        inner_min, inner_max: Inner box bounds.
        tolerance: Allowed overshoot in mesh units (meters).

    Returns:
        True if inner bbox is contained within outer bbox.
    """
    return bool(
        np.all(inner_min >= outer_min - tolerance) and
        np.all(inner_max <= outer_max + tolerance)
    )


def _centroid_in_bbox(
    centroid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    tolerance: float = 1e-3,
) -> bool:
    """Check that a centroid lies within a bounding box."""
    return bool(
        np.all(centroid >= bbox_min - tolerance) and
        np.all(centroid <= bbox_max + tolerance)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mesh statistics
# ──────────────────────────────────────────────────────────────────────────────

def report_mesh_statistics(meshes: Dict[str, Any]) -> Dict[str, Dict]:
    """Compute geometric statistics for all meshes.

    Args:
        meshes: Dict mapping region name → trimesh.Trimesh (or None).

    Returns:
        Dict mapping region name → stats dict.
    """
    stats = {}
    for name, mesh in meshes.items():
        if mesh is None:
            stats[name] = {
                "present": False,
                "vertex_count": 0,
                "face_count": 0,
                "volume_m3": 0.0,
                "surface_area_m2": 0.0,
                "centroid_m": None,
                "bbox_min_m": None,
                "bbox_max_m": None,
            }
            continue

        try:
            vol = float(mesh.volume) if mesh.is_watertight else None
        except Exception:
            vol = None

        try:
            area = float(mesh.area)
        except Exception:
            area = 0.0

        centroid = mesh.centroid.tolist() if len(mesh.vertices) > 0 else None
        bbox = _get_mesh_bbox(mesh)

        stats[name] = {
            "present": True,
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "volume_m3": round(vol, 9) if vol is not None else None,
            "surface_area_m2": round(area, 9),
            "centroid_m": [round(v, 6) for v in centroid] if centroid else None,
            "bbox_min_m": [round(v, 6) for v in bbox[0].tolist()] if bbox else None,
            "bbox_max_m": [round(v, 6) for v in bbox[1].tolist()] if bbox else None,
            "is_watertight": bool(mesh.is_watertight) if mesh else False,
        }

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Containment validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_mesh_containment(
    wt_mesh,
    tc_mesh,
    et_mesh,
    brain_mesh=None,
    tolerance: float = 5e-3,  # 5mm in meters
) -> dict:
    """Validate that tumor mesh bounding boxes satisfy ET ⊆ TC ⊆ WT.

    Also checks that all tumor meshes are inside the brain mesh bounds.

    Args:
        wt_mesh: Whole Tumor trimesh.Trimesh (or None).
        tc_mesh: Tumor Core trimesh.Trimesh (or None).
        et_mesh: Enhancing Tumor trimesh.Trimesh (or None).
        brain_mesh: Brain surface trimesh.Trimesh (or None).
        tolerance: Allowed boundary overshoot in meters.

    Returns:
        Dict with "valid", "violations", "checks", "statistics".
    """
    violations: List[str] = []
    checks: Dict[str, bool] = {}

    wt_bbox = _get_mesh_bbox(wt_mesh)
    tc_bbox = _get_mesh_bbox(tc_mesh)
    et_bbox = _get_mesh_bbox(et_mesh)
    brain_bbox = _get_mesh_bbox(brain_mesh) if brain_mesh is not None else None

    # ── ET ⊆ TC ─────────────────────────────────────────────────────────────
    if et_bbox is not None and tc_bbox is not None:
        et_in_tc = _bbox_contains_bbox(tc_bbox[0], tc_bbox[1],
                                       et_bbox[0], et_bbox[1], tolerance)
        checks["et_bbox_within_tc_bbox"] = et_in_tc
        if not et_in_tc:
            violations.append(
                "ET bounding box extends outside TC bounding box "
                f"(tolerance={tolerance*1000:.0f}mm)"
            )

        # Centroid check
        et_centroid = np.array(et_mesh.centroid) if et_mesh and len(et_mesh.vertices) > 0 else None
        if et_centroid is not None:
            et_centroid_in_tc = _centroid_in_bbox(et_centroid, tc_bbox[0], tc_bbox[1], tolerance)
            checks["et_centroid_within_tc_bbox"] = et_centroid_in_tc
            if not et_centroid_in_tc:
                violations.append("ET centroid lies outside TC bounding box")
    elif et_bbox is not None and tc_bbox is None:
        violations.append("ET mesh present but TC mesh is empty (containment invalid)")
        checks["et_bbox_within_tc_bbox"] = False
    else:
        checks["et_bbox_within_tc_bbox"] = True  # ET is absent — no violation

    # ── TC ⊆ WT ─────────────────────────────────────────────────────────────
    if tc_bbox is not None and wt_bbox is not None:
        tc_in_wt = _bbox_contains_bbox(wt_bbox[0], wt_bbox[1],
                                       tc_bbox[0], tc_bbox[1], tolerance)
        checks["tc_bbox_within_wt_bbox"] = tc_in_wt
        if not tc_in_wt:
            violations.append(
                "TC bounding box extends outside WT bounding box "
                f"(tolerance={tolerance*1000:.0f}mm)"
            )
    elif tc_bbox is not None and wt_bbox is None:
        violations.append("TC mesh present but WT mesh is empty (containment invalid)")
        checks["tc_bbox_within_wt_bbox"] = False
    else:
        checks["tc_bbox_within_wt_bbox"] = True

    # ── Tumor meshes inside brain ────────────────────────────────────────────
    if brain_bbox is not None and wt_bbox is not None:
        wt_in_brain = _bbox_contains_bbox(brain_bbox[0], brain_bbox[1],
                                          wt_bbox[0], wt_bbox[1], tolerance)
        checks["wt_inside_brain"] = wt_in_brain
        if not wt_in_brain:
            violations.append(
                "WT bounding box extends outside brain bounding box "
                "(coordinate misalignment possible)"
            )

    # ── Statistics ───────────────────────────────────────────────────────────
    mesh_stats = report_mesh_statistics({
        "whole_tumor": wt_mesh,
        "tumor_core": tc_mesh,
        "enhancing_tumor": et_mesh,
    })

    valid = len(violations) == 0

    return {
        "valid": valid,
        "violations": violations,
        "checks": checks,
        "mesh_statistics": mesh_stats,
        "disclaimer": (
            "AI-derived mesh geometry — research prototype, "
            "not for clinical diagnosis."
        ),
    }
