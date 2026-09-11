"""
Segmentation-to-Mesh Pipeline for NeuroVR.

Converts 3D segmentation masks into renderable 3D meshes for
Three.js visualization:

  Segmentation volume [3, H, W, D]
       ↓  Containment enforcement
  Binary masks (WT, TC, ET)
       ↓  Marching Cubes (skimage / mcubes)
  Triangle mesh (vertices, faces)
       ↓  Mesh smoothing (optional)
  Trimesh object
       ↓  Coordinate transform (voxel → Three.js)
  GLB/GLTF file
       ↓  Containment validation
  Validated mesh with metadata

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Mesh color palette
# ──────────────────────────────────────────────────────────────────────────────

REGION_COLORS = {
    "whole_tumor": [0.2, 0.7, 1.0, 0.35],    # Cyan, semi-transparent
    "tumor_core": [1.0, 0.4, 0.1, 0.55],      # Orange
    "enhancing_tumor": [1.0, 0.95, 0.1, 0.80], # Yellow, opaque
}


# ──────────────────────────────────────────────────────────────────────────────
# Marching Cubes
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_mesh(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.5,
    smooth: bool = True,
    smooth_iterations: int = 3,
    decimate_fraction: float = 0.0,
):
    """Convert a 3D binary mask to a triangle mesh via Marching Cubes.

    Args:
        mask: Binary 3D array (H, W, D) — uint8 or bool.
        spacing: Voxel spacing (mm) in each dimension.
        level: Isosurface level for Marching Cubes.
        smooth: Apply Laplacian smoothing to reduce staircase artifact.
        smooth_iterations: Number of smoothing iterations.
        decimate_fraction: Fraction of faces to remove (0=no decimation).

    Returns:
        trimesh.Trimesh object, or None if mask is empty.
    """
    try:
        import trimesh
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError(
            "trimesh and scikit-image are required: "
            "pip install trimesh scikit-image"
        ) from exc

    mask_arr = mask.astype(np.float32)

    if mask_arr.max() < level:
        return None  # Empty mask

    # Marching Cubes
    verts, faces, normals, _ = marching_cubes(
        mask_arr,
        level=level,
        spacing=spacing,
        method="lewiner",
    )

    if len(verts) < 4 or len(faces) < 1:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Laplacian smoothing
    if smooth and smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
        except Exception:
            pass  # Smoothing is optional

    # Decimation (reduce polygon count for web rendering)
    if decimate_fraction > 0:
        try:
            target_faces = max(100, int(len(mesh.faces) * (1 - decimate_fraction)))
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            pass

    return mesh


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transform
# ──────────────────────────────────────────────────────────────────────────────

def apply_voxel_to_threejs_transform(mesh, mapper) -> None:
    """Transform mesh vertices from voxel space to Three.js space (in-place).

    Args:
        mesh: trimesh.Trimesh with vertices in voxel coordinates.
        mapper: RASToThreeJS instance from reconstruction.coordinate_mapper.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return
    mesh.vertices = mapper.voxel_to_threejs(mesh.vertices)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def segmentation_to_meshes(
    seg_volume: np.ndarray,
    affine: np.ndarray,
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth: bool = True,
    smooth_iterations: int = 3,
    decimate_fraction: float = 0.3,
    validate: bool = True,
) -> Tuple[Dict[str, Any], dict]:
    """Full segmentation-to-mesh pipeline.

    Converts a 3-channel segmentation volume into Three.js-ready meshes
    with coordinate transformation and containment validation.

    Args:
        seg_volume: [3, H, W, D] float32 binary masks (WT, TC, ET)
            OR [H, W, D] int32 BraTS integer segmentation.
        affine: NIfTI 4×4 affine matrix (voxel → RAS mm).
        spacing_mm: Voxel spacing in mm for Marching Cubes.
        smooth: Apply Laplacian mesh smoothing.
        smooth_iterations: Laplacian smoothing iterations.
        decimate_fraction: Fraction of faces to remove for web rendering.
        validate: Run containment validation after mesh generation.

    Returns:
        Tuple of:
          - meshes: Dict mapping region name → trimesh.Trimesh (or None)
          - metadata: Dict with validation results, transform info, statistics
    """
    from data.label_utils import brats_seg_to_regions, enforce_containment
    from reconstruction.coordinate_mapper import RASToThreeJS, validate_coordinate_chain
    from reconstruction.mesh_validation import validate_mesh_containment

    # ── Convert labels ──────────────────────────────────────────────────────
    if seg_volume.ndim == 4 and seg_volume.shape[0] == 3:
        # Multi-channel input [3, H, W, D]
        wt_mask = (seg_volume[0] > 0.5).astype(np.uint8)
        tc_mask = (seg_volume[1] > 0.5).astype(np.uint8)
        et_mask = (seg_volume[2] > 0.5).astype(np.uint8)
    elif seg_volume.ndim == 3:
        # Integer BraTS labels [H, W, D]
        wt_mask, tc_mask, et_mask = brats_seg_to_regions(
            seg_volume.astype(np.int32), enforce=True
        )
    else:
        raise ValueError(
            f"Expected seg_volume shape [3,H,W,D] or [H,W,D], got {seg_volume.shape}"
        )

    # Enforce containment on masks before mesh generation
    wt_mask, tc_mask, et_mask = enforce_containment(wt_mask, tc_mask, et_mask)

    # ── Marching Cubes ───────────────────────────────────────────────────────
    print("[Mesh] Generating meshes via Marching Cubes...")
    wt_mesh = mask_to_mesh(wt_mask, spacing_mm, smooth=smooth,
                           smooth_iterations=smooth_iterations,
                           decimate_fraction=decimate_fraction)
    tc_mesh = mask_to_mesh(tc_mask, spacing_mm, smooth=smooth,
                           smooth_iterations=smooth_iterations,
                           decimate_fraction=decimate_fraction)
    et_mesh = mask_to_mesh(et_mask, spacing_mm, smooth=smooth,
                           smooth_iterations=smooth_iterations,
                           decimate_fraction=decimate_fraction)

    for name, mesh in [("WT", wt_mesh), ("TC", tc_mesh), ("ET", et_mesh)]:
        if mesh is not None:
            print(f"[Mesh]   {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        else:
            print(f"[Mesh]   {name}: empty (no tissue present)")

    # ── Coordinate transform ─────────────────────────────────────────────────
    mapper = RASToThreeJS(affine)
    coord_validation = validate_coordinate_chain(affine)
    print(f"[Mesh] Coordinate chain valid: {coord_validation['valid']}, "
          f"max error: {coord_validation['max_error_mm']:.4f} mm")

    for mesh in (wt_mesh, tc_mesh, et_mesh):
        apply_voxel_to_threejs_transform(mesh, mapper)

    # Apply colors
    try:
        import trimesh
        if wt_mesh:
            c = REGION_COLORS["whole_tumor"]
            wt_mesh.visual.vertex_colors = [int(v * 255) for v in c[:3]] + [int(c[3] * 255)]
        if tc_mesh:
            c = REGION_COLORS["tumor_core"]
            tc_mesh.visual.vertex_colors = [int(v * 255) for v in c[:3]] + [int(c[3] * 255)]
        if et_mesh:
            c = REGION_COLORS["enhancing_tumor"]
            et_mesh.visual.vertex_colors = [int(v * 255) for v in c[:3]] + [int(c[3] * 255)]
    except Exception:
        pass

    # ── Containment validation ───────────────────────────────────────────────
    validation_result = {}
    if validate:
        validation_result = validate_mesh_containment(wt_mesh, tc_mesh, et_mesh)
        if validation_result["valid"]:
            print("[Mesh] ✓ Containment validation passed (ET ⊆ TC ⊆ WT)")
        else:
            print(f"[Mesh] ✗ Containment violations: {validation_result['violations']}")

    meshes = {
        "whole_tumor": wt_mesh,
        "tumor_core": tc_mesh,
        "enhancing_tumor": et_mesh,
    }

    metadata = {
        "coordinate_validation": coord_validation,
        "containment_validation": validation_result,
        "coordinate_metadata": mapper.get_metadata(),
        "disclaimer": (
            "AI-derived mesh — research prototype, "
            "not for clinical diagnosis."
        ),
    }

    return meshes, metadata


def export_meshes_to_glb(
    meshes: Dict[str, Any],
    metadata: dict,
    output_dir: str,
    prefix: str = "tumor",
) -> Dict[str, str]:
    """Export all meshes to GLB files with embedded metadata.

    Args:
        meshes: Dict region name → trimesh.Trimesh (or None).
        metadata: Coordinate and validation metadata to embed.
        output_dir: Directory for output files.
        prefix: Filename prefix.

    Returns:
        Dict mapping region name → output file path.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh required: pip install trimesh") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    for region_name, mesh in meshes.items():
        if mesh is None:
            print(f"[Export] Skipping {region_name}: empty mesh")
            continue

        glb_path = out / f"{prefix}_{region_name}.glb"

        # Embed metadata as GLTF extras
        try:
            scene = trimesh.Scene()
            scene.add_geometry(mesh, geom_name=region_name)
            glb_bytes = scene.export(file_type="glb")
            glb_path.write_bytes(glb_bytes)
        except Exception as exc:
            # Fallback: export mesh directly
            try:
                mesh.export(str(glb_path))
            except Exception as exc2:
                print(f"[Export] Failed to export {region_name}: {exc2}")
                continue

        # Save companion metadata JSON
        meta_path = out / f"{prefix}_{region_name}_meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        paths[region_name] = str(glb_path)
        print(f"[Export] {region_name}: {glb_path}")

    return paths


def run_pipeline(
    seg_path: str,
    output_dir: str,
    affine: Optional[np.ndarray] = None,
    smooth: bool = True,
    decimate_fraction: float = 0.3,
) -> Tuple[Dict[str, str], dict]:
    """High-level entrypoint: NIfTI seg file → GLB files.

    Args:
        seg_path: Path to segmentation NIfTI file (.nii or .nii.gz).
        output_dir: Output directory for GLB files.
        affine: Affine override (uses file affine if None).
        smooth: Apply Laplacian smoothing.
        decimate_fraction: Face decimation fraction.

    Returns:
        (file_paths, metadata)
    """
    import nibabel as nib

    img = nib.load(seg_path)
    seg = img.get_fdata(dtype=np.float32).astype(np.int32)
    aff = affine if affine is not None else img.affine
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])

    meshes, metadata = segmentation_to_meshes(
        seg, aff, spacing_mm=spacing,
        smooth=smooth, decimate_fraction=decimate_fraction,
    )
    file_paths = export_meshes_to_glb(meshes, metadata, output_dir)
    return file_paths, metadata
