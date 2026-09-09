"""
NeuroVR 3D — Anatomical Tumor Localization Engine

Determines anatomical location of the tumor from:
  1. NIfTI affine matrix → world/RAS coordinates
  2. Tumor mask centroid → scanner-space position
  3. MNI-space coordinate ranges → estimated lobe

IMPORTANT HONESTY CONSTRAINTS:
- Left/Right is computed from the actual NIfTI affine + centroid.
- Lobe is ESTIMATED from published MNI152 coordinate bounding boxes.
  It is always labeled "Estimated" — never claimed as clinical diagnosis.
- Tumor TYPE is not determined here (BraTS is not a classifier).
- If data is synthetic (identity affine), side = UNCERTAIN.

Output is cached to localization.json in the session directory.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np


# ─── MNI coordinate bounding boxes for major brain regions ────────────────────
# Sources: Tzourio-Mazoyer et al. 2002 (AAL atlas), MNI152 standard space.
# These are approximate — labeled "Estimated" in the UI.
#
# Coordinate convention: MNI RAS (X: L←0→R, Y: P←0→A, Z: I←0→S)
#
# Format: (x_min, x_max, y_min, y_max, z_min, z_max)
# None means no constraint on that axis.
#
MNI_LOBE_BOXES = [
    # name,         xmin,  xmax,   ymin,  ymax,  zmin,  zmax
    ("Frontal Lobe",     None, None,  -20,   None,   0,   None),
    ("Parietal Lobe",    None, None, -100,   -20,   30,   None),
    ("Temporal Lobe",    None, None,  -65,    10,  -25,     30),
    ("Occipital Lobe",   None, None, -110,   -60,  -10,   None),
    ("Cerebellum",       None, None,  None,  None,  -80,   -20),
    ("Brainstem",         -15,   15,  -50,   -10,  -70,   -20),
]

# Midline threshold (mm): centroid X within ±MIDLINE_THRESH → MIDLINE
MIDLINE_THRESH_MM = 8.0

# Identity affine threshold: if close to eye(4) × any scalar, data is synthetic
SYNTHETIC_THRESH = 0.1


def _is_synthetic_affine(affine: np.ndarray) -> bool:
    """Return True if affine looks like a synthetic/identity matrix."""
    if affine is None:
        return True
    try:
        # Normalise by the diagonal scale factor
        scale = np.abs(np.diag(affine[:3, :3])).mean()
        if scale < 1e-6:
            return True
        norm = affine.copy()
        norm[:3, :3] /= scale
        norm[:3, 3]  /= scale
        diff = np.abs(norm - np.eye(4)).max()
        return diff < SYNTHETIC_THRESH
    except Exception:
        return True


def _centroid_voxel(mask: np.ndarray) -> Optional[np.ndarray]:
    """Compute mask centroid in voxel coordinates."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)


def _voxel_to_ras(centroid_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply affine to voxel centroid → world RAS coordinates (mm)."""
    hom = np.append(centroid_vox.astype(float), 1.0)
    world = affine @ hom
    return world[:3]


def _affine_orientation(affine: np.ndarray) -> str:
    """
    Determine the voxel-to-world orientation from the affine.
    Returns a 3-character code like 'RAS', 'LAS', 'LPS', etc.
    Following nibabel's aff2axcodes convention.
    """
    try:
        # Get the dominant axis directions from the rotation part
        rot = affine[:3, :3]
        codes = []
        for col in range(3):
            axis = rot[:, col]
            dominant = np.argmax(np.abs(axis))
            sign = np.sign(axis[dominant])
            labels = [('R', 'L'), ('A', 'P'), ('S', 'I')]
            codes.append(labels[dominant][0 if sign > 0 else 1])
        return ''.join(codes)
    except Exception:
        return 'RAS'


def _determine_side(
    centroid_ras: np.ndarray,
    affine: np.ndarray,
) -> Tuple[str, str, str]:
    """
    Determine LEFT / RIGHT / MIDLINE from RAS centroid.

    In RAS space:
      X > 0  →  patient RIGHT
      X < 0  →  patient LEFT
      |X| < threshold → MIDLINE

    Returns: (side, confidence, method)
    """
    if _is_synthetic_affine(affine):
        return ("UNCERTAIN", "Low",
                "Synthetic/identity affine — orientation not deterministic")

    x = float(centroid_ras[0])

    if abs(x) < MIDLINE_THRESH_MM:
        return ("MIDLINE", "Moderate",
                f"Centroid X={x:.1f}mm within ±{MIDLINE_THRESH_MM}mm midline zone")

    side = "RIGHT" if x > 0 else "LEFT"
    dist = abs(x)
    confidence = "High" if dist > 25 else "Moderate" if dist > 10 else "Low"
    method = (
        f"NIfTI affine RAS X coordinate: {x:.1f}mm "
        f"({'right' if x>0 else 'left'} of midline)"
    )
    return (side, confidence, method)


def _determine_lobe(
    centroid_ras: np.ndarray,
    affine: np.ndarray,
) -> Tuple[Optional[str], str, str, bool]:
    """
    Estimate anatomical lobe from MNI152 coordinate bounding boxes.

    Returns: (lobe_name, confidence, method, available)
    """
    if _is_synthetic_affine(affine):
        return (None, "Low",
                "Synthetic affine — MNI coordinates unreliable", False)

    x, y, z = float(centroid_ras[0]), float(centroid_ras[1]), float(centroid_ras[2])

    matches = []
    for name, xmin, xmax, ymin, ymax, zmin, zmax in MNI_LOBE_BOXES:
        in_box = True
        if xmin is not None and abs(x) < xmin: in_box = False  # use |x| for paired
        if xmax is not None and abs(x) > xmax: in_box = False
        if ymin is not None and y < ymin:       in_box = False
        if ymax is not None and y > ymax:       in_box = False
        if zmin is not None and z < zmin:       in_box = False
        if zmax is not None and z > zmax:       in_box = False
        if in_box:
            matches.append(name)

    method = (
        f"MNI152 coordinate bounding boxes — "
        f"centroid at RAS [{x:.1f}, {y:.1f}, {z:.1f}] mm. "
        f"ESTIMATED — not based on full atlas registration."
    )

    if len(matches) == 0:
        return (None, "Low",
                method + " No matching region box.", False)

    if len(matches) == 1:
        confidence = "Moderate"
    else:
        # Multiple matches — pick the first (ordered by anatomical priority above)
        confidence = "Low"

    return (matches[0], confidence, method, True)


def localize_tumor(
    wt_mask: np.ndarray,
    affine: np.ndarray,
    voxel_spacing: tuple,
) -> Dict:
    """
    Main localization function. Called once per session after segmentation.

    Args:
        wt_mask:       Whole-tumor binary mask (H, W, D)
        affine:        4×4 NIfTI affine matrix
        voxel_spacing: (dx, dy, dz) in mm

    Returns:
        Localization dict suitable for JSON serialization and frontend display.
    """
    result: Dict = {
        "tumor_detected": False,
        "centroid_voxel": None,
        "centroid_ras_mm": None,
        "side": "UNCERTAIN",
        "side_confidence": "Low",
        "side_method": "No tumor detected",
        "lobe": None,
        "lobe_confidence": "Low",
        "lobe_method": "No tumor detected",
        "lobe_available": False,
        "full_label": "UNCERTAIN",
        "localization_disclaimer": (
            "Anatomical localization is AI/coordinate-estimated from NIfTI spatial "
            "metadata and published MNI152 reference bounding boxes. "
            "This is NOT a clinical diagnosis. Consult a qualified radiologist."
        ),
        "error": None,
    }

    try:
        if wt_mask is None or wt_mask.sum() == 0:
            result["error"] = "No tumor voxels in whole-tumor mask."
            return result

        result["tumor_detected"] = True

        # ── Centroid ──────────────────────────────────────────────────────────
        centroid_vox = _centroid_voxel(wt_mask)
        result["centroid_voxel"] = [round(float(v), 2) for v in centroid_vox]

        if affine is None:
            affine = np.eye(4)
        affine = np.array(affine, dtype=float)

        centroid_ras = _voxel_to_ras(centroid_vox, affine)
        result["centroid_ras_mm"] = [round(float(v), 2) for v in centroid_ras]

        # ── Orientation info ──────────────────────────────────────────────────
        result["affine_orientation"] = _affine_orientation(affine)
        result["is_synthetic_data"]  = bool(_is_synthetic_affine(affine))

        # ── Left / Right ──────────────────────────────────────────────────────
        side, side_conf, side_method = _determine_side(centroid_ras, affine)
        result["side"]            = side
        result["side_confidence"] = side_conf
        result["side_method"]     = side_method

        # ── Lobe ─────────────────────────────────────────────────────────────
        lobe, lobe_conf, lobe_method, lobe_avail = _determine_lobe(centroid_ras, affine)
        result["lobe"]            = lobe
        result["lobe_confidence"] = lobe_conf
        result["lobe_method"]     = lobe_method
        result["lobe_available"]  = lobe_avail

        # ── Full label ────────────────────────────────────────────────────────
        if side not in ("UNCERTAIN", "MIDLINE") and lobe_avail and lobe:
            result["full_label"] = f"{side} {lobe.upper()}"
        elif side == "MIDLINE":
            result["full_label"] = f"MIDLINE / DEEP — {lobe or 'Unknown Region'}"
        elif lobe_avail and lobe:
            result["full_label"] = lobe.upper()
        else:
            result["full_label"] = "LOCATION UNCERTAIN"

    except Exception as exc:
        result["error"] = str(exc)
        result["full_label"] = "LOCATION UNCERTAIN"

    return result


def load_localization(session_dir: str) -> Optional[Dict]:
    """Load cached localization.json from session directory."""
    import os
    path = os.path.join(session_dir, "localization.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
