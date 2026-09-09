"""
3D Tumor Mesh Generation for NeuroVR.

Converts binary segmentation masks into 3D surface meshes using
Marching Cubes and exports them as GLB/GLTF files for Three.js.

Physical voxel spacing is applied to vertices so the mesh is in
real-world mm coordinates (scaled to meters for Three.js convention).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _check_dependencies() -> None:
    """Verify required packages for mesh generation."""
    try:
        from skimage.measure import marching_cubes  # noqa
    except ImportError:
        raise ImportError("scikit-image is required: pip install scikit-image")
    try:
        import trimesh  # noqa
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")


def mask_to_mesh(
    mask: np.ndarray,
    voxel_spacing: np.ndarray,
    level: float = 0.5,
    smoothing_iterations: int = 3,
    step_size: int = 1,
) -> Optional[object]:
    """Convert a binary mask to a 3D triangle mesh using Marching Cubes.

    Args:
        mask: Binary uint8 array (H, W, D).
        voxel_spacing: Physical voxel size [dx, dy, dz] in mm.
        level: Iso-surface level for Marching Cubes (default 0.5).
        smoothing_iterations: Laplacian smoothing passes (0 = disabled).
        step_size: Marching cubes step size (1 = full resolution).

    Returns:
        trimesh.Trimesh object, or None if mask is empty.
    """
    _check_dependencies()
    from skimage.measure import marching_cubes
    import trimesh

    if mask.sum() == 0:
        return None

    # Pad mask by 1 voxel on all sides to ensure closed surface
    padded = np.pad(mask.astype(np.float32), pad_width=1, mode="constant", constant_values=0)

    try:
        verts, faces, normals, _ = marching_cubes(
            padded,
            level=level,
            spacing=(1.0, 1.0, 1.0),  # Unit spacing; scale below
            step_size=step_size,
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[TumorMesh] Marching Cubes failed: {exc}")
        return None

    # Adjust for padding offset
    verts -= 1.0

    # Apply physical voxel spacing (mm)
    verts *= voxel_spacing  # Scale each axis by its voxel size

    # Convert mm → meters for Three.js (Three.js units = meters)
    verts /= 1000.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Smooth mesh
    if smoothing_iterations > 0 and len(mesh.vertices) > 3:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)
        except Exception as exc:
            print(f"[TumorMesh] Smoothing failed (mesh kept): {exc}")

    return mesh


def export_mesh_glb(mesh, output_path: str) -> bool:
    """Export a trimesh Trimesh as a GLB file.

    Args:
        mesh: trimesh.Trimesh object.
        output_path: Destination .glb file path.

    Returns:
        True if successful, False otherwise.
    """
    import trimesh

    if mesh is None:
        return False
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        mesh.export(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"[TumorMesh] Exported GLB: {output_path} ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"[TumorMesh] GLB export failed: {exc}")
        return False


def generate_tumor_meshes(
    tc_mask: np.ndarray,
    wt_mask: np.ndarray,
    et_mask: np.ndarray,
    voxel_spacing: np.ndarray,
    output_dir: str,
    smoothing_iterations: int = 3,
) -> Dict[str, str]:
    """Generate GLB meshes for all three tumor regions.

    Args:
        tc_mask: Tumor Core binary mask (H, W, D).
        wt_mask: Whole Tumor binary mask (H, W, D).
        et_mask: Enhancing Tumor binary mask (H, W, D).
        voxel_spacing: Physical voxel size [dx, dy, dz] in mm.
        output_dir: Directory where .glb files will be saved.
        smoothing_iterations: Laplacian smoothing passes.

    Returns:
        Dict mapping region name → GLB file path (or None if empty).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    regions = {
        "tumor_whole": wt_mask,
        "tumor_core": tc_mask,
        "tumor_enhancing": et_mask,
    }

    for name, mask in regions.items():
        out_path = os.path.join(output_dir, f"{name}.glb")
        print(f"[TumorMesh] Generating mesh for {name} ({int(mask.sum())} voxels)...")

        mesh = mask_to_mesh(
            mask,
            voxel_spacing=voxel_spacing,
            smoothing_iterations=smoothing_iterations,
        )

        if mesh is not None:
            success = export_mesh_glb(mesh, out_path)
            paths[name] = out_path if success else None
        else:
            print(f"[TumorMesh] {name}: no tumor → skipping GLB")
            paths[name] = None

    return paths


def generate_brain_surface(
    t1_volume: np.ndarray,
    voxel_spacing: np.ndarray,
    output_path: str,
    threshold_percentile: float = 15.0,  # kept for backward compat, unused now
    smoothing_iterations: int = 2,        # 2 passes — preserves gyri/sulci folds
    step_size: int = 1,                   # full resolution — captures cortical detail
) -> Optional[str]:
    """Generate an anatomically realistic brain surface mesh from a T1 MRI volume.

    Uses adaptive Otsu thresholding + morphological cleanup to produce a clean
    brain mask that preserves cortical gyri and sulci detail. This is for
    educational/research visualization only — NOT for clinical use.

    Pipeline:
        1. Otsu threshold on non-zero voxels (adaptive to MRI intensity)
        2. Binary hole-filling (closes CSF gaps inside brain)
        3. Erosion × 2 → Dilation × 2 (removes skull/scalp fragments)
        4. Largest connected component (discards remaining noise)
        5. Marching Cubes at step_size=1 (full resolution — keeps gyral folds)
        6. Laplacian smoothing × 2 (minimal — removes staircase only)

    Args:
        t1_volume: Float32 T1 MRI volume (H, W, D).
        voxel_spacing: Physical voxel size [dx, dy, dz] in mm.
        output_path: Destination .glb path.
        threshold_percentile: Legacy parameter — not used. Otsu is used instead.
        smoothing_iterations: Laplacian smoothing passes (default 2 to preserve folds).
        step_size: Marching Cubes step size (default 1 = full resolution).

    Returns:
        Output GLB path if successful, None otherwise.
    """
    _check_dependencies()
    from scipy import ndimage

    nonzero = t1_volume[t1_volume > 0]
    if nonzero.size == 0:
        print("[BrainSurface] T1 volume has no non-zero voxels — aborting.")
        return None

    # ── Step 1: Adaptive Otsu threshold ─────────────────────────────────────
    # Otsu finds the optimal threshold between background and brain tissue.
    # Using 35% of Otsu value captures full brain parenchyma including WM/GM.
    try:
        from skimage.filters import threshold_otsu
        otsu_val = threshold_otsu(nonzero)
        brain_thresh = float(otsu_val) * 0.35
        print(f"[BrainSurface] Otsu={otsu_val:.1f}, brain_thresh={brain_thresh:.1f}")
    except Exception:
        # Fallback to percentile if skimage not available
        brain_thresh = float(np.percentile(nonzero, 15.0))
        print(f"[BrainSurface] Otsu failed, using percentile fallback: {brain_thresh:.1f}")

    brain_mask = (t1_volume > brain_thresh).astype(np.uint8)
    print(f"[BrainSurface] Initial mask: {int(brain_mask.sum())} voxels")

    # ── Step 2: Fill holes (CSF spaces, ventricles) ─────────────────────────
    # Fills interior cavities so the brain surface is a solid closed volume.
    try:
        brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)
    except Exception as e:
        print(f"[BrainSurface] fill_holes failed (kept): {e}")

    # ── Step 3: Morphological opening (erode→dilate) ────────────────────────
    # Erosion removes thin skull/scalp fragments attached to brain surface.
    # Dilation restores brain volume without reintroducing skull.
    try:
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity kernel
        brain_mask = ndimage.binary_erosion(brain_mask, structure=struct,
                                            iterations=2).astype(np.uint8)
        brain_mask = ndimage.binary_dilation(brain_mask, structure=struct,
                                             iterations=2).astype(np.uint8)
        print(f"[BrainSurface] After morphological opening: {int(brain_mask.sum())} voxels")
    except Exception as e:
        print(f"[BrainSurface] Morphological ops failed (kept): {e}")

    # ── Step 4: Largest connected component ─────────────────────────────────
    # Removes any remaining disconnected noise blobs (skull fragments, etc.).
    try:
        labeled, n_components = ndimage.label(brain_mask)
        if n_components > 1:
            component_sizes = ndimage.sum(brain_mask, labeled, range(1, n_components + 1))
            largest_label = int(np.argmax(component_sizes)) + 1
            brain_mask = (labeled == largest_label).astype(np.uint8)
            print(f"[BrainSurface] Kept largest of {n_components} components "
                  f"({int(brain_mask.sum())} voxels)")
    except Exception as e:
        print(f"[BrainSurface] Connected component filtering failed (kept): {e}")

    if brain_mask.sum() == 0:
        print("[BrainSurface] Brain mask empty after cleanup — aborting.")
        return None

    # ── Step 5+6: Marching Cubes + controlled smoothing ─────────────────────
    # step_size=1 → full voxel resolution → gyri and sulci are preserved.
    # smoothing_iterations=2 → removes only marching-cubes staircase artefacts.
    mesh = mask_to_mesh(
        brain_mask,
        voxel_spacing=voxel_spacing,
        level=0.5,
        smoothing_iterations=smoothing_iterations,
        step_size=step_size,
    )

    if mesh is None:
        print("[BrainSurface] Marching Cubes returned no mesh.")
        return None

    print(f"[BrainSurface] Brain mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.faces):,} faces")

    success = export_mesh_glb(mesh, output_path)
    return output_path if success else None
