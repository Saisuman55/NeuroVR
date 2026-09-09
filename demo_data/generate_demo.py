"""
Synthetic BraTS-Compatible Demo Data Generator for NeuroVR 3D.

Creates realistic synthetic MRI volumes with a programmatic tumor.
This is strictly for demonstrating the 3D pipeline — no real patient data.

Usage:
    python demo_data/generate_demo.py [--output-dir demo_data/case001] [--shape 64 64 64]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def generate_synthetic_brats_case(
    output_dir: str,
    shape: Tuple[int, int, int] = (64, 64, 64),
    spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    seed: int = 42,
) -> dict:
    """Generate 4 synthetic NIfTI modalities mimicking a BraTS case.

    The tumor is a synthetic ellipsoid centered in the volume.
    Each modality has realistic relative signal characteristics:

      T1:    low signal in edema, moderate in core
      T1ce:  bright enhancement ring (enhancing tumor)
      T2:    high signal in whole tumor
      FLAIR: very high signal in edema + core

    Args:
        output_dir: Where to save the .nii.gz files.
        shape: Volume shape (H, W, D).
        spacing: Voxel size in mm.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping modality → output path.

    IMPORTANT: These volumes are SYNTHETIC. The pipeline treats them
    identically to real MRI — the same MONAI model runs on them.
    Label all outputs as DEMO when presenting to users.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = {}

    H, W, D = shape
    cx, cy, cz = H // 2, W // 2, D // 2

    # Spatial distance arrays from center
    zz, yy, xx = np.mgrid[0:H, 0:W, 0:D]
    dist = np.sqrt(
        ((xx - cx) / 8) ** 2 +
        ((yy - cy) / 10) ** 2 +
        ((zz - cz) / 9) ** 2
    )

    # Define tumor sub-regions by distance
    enhancing_tumor   = dist < 0.6   # Small bright core
    tumor_core        = dist < 1.0   # Core including necrosis
    whole_tumor       = dist < 1.4   # Core + edema

    # Brain mask: ellipsoid covering most of the volume
    brain_dist = np.sqrt(
        ((xx - cx) / (H * 0.42)) ** 2 +
        ((yy - cy) / (W * 0.42)) ** 2 +
        ((zz - cz) / (D * 0.42)) ** 2
    )
    brain_mask = brain_dist < 1.0

    # Affine (diagonal = spacing)
    affine = np.diag([*spacing, 1.0])

    def _save(volume: np.ndarray, modality: str) -> str:
        volume = volume * brain_mask  # Zero outside brain
        img = nib.Nifti1Image(volume.astype(np.float32), affine)
        img.header.set_zooms(spacing)
        p = os.path.join(output_dir, f"{modality}.nii.gz")
        nib.save(img, p)
        return p

    # ── T1: Grey matter ~0.5, white matter ~0.7, tumor core moderate ──────────
    t1 = rng.normal(0.55, 0.08, shape).astype(np.float32)
    t1[whole_tumor]      = rng.normal(0.35, 0.05, whole_tumor.sum())   # Edema: low
    t1[tumor_core]       = rng.normal(0.50, 0.06, tumor_core.sum())    # Core: moderate
    t1[enhancing_tumor]  = rng.normal(0.60, 0.04, enhancing_tumor.sum())
    paths["t1"] = _save(np.clip(t1, 0, 1), "t1")

    # ── T1ce: Enhancing tumor bright, rest similar to T1 ─────────────────────
    t1ce = t1.copy()
    t1ce[enhancing_tumor] = rng.normal(0.92, 0.04, enhancing_tumor.sum())
    paths["t1ce"] = _save(np.clip(t1ce, 0, 1), "t1ce")

    # ── T2: Edema bright, core variable ──────────────────────────────────────
    t2 = rng.normal(0.4, 0.07, shape).astype(np.float32)
    t2[whole_tumor]     = rng.normal(0.82, 0.05, whole_tumor.sum())
    t2[tumor_core]      = rng.normal(0.65, 0.06, tumor_core.sum())
    t2[enhancing_tumor] = rng.normal(0.55, 0.05, enhancing_tumor.sum())
    paths["t2"] = _save(np.clip(t2, 0, 1), "t2")

    # ── FLAIR: Edema very bright, CSF dark ───────────────────────────────────
    flair = rng.normal(0.3, 0.06, shape).astype(np.float32)
    flair[whole_tumor]     = rng.normal(0.90, 0.04, whole_tumor.sum())
    flair[tumor_core]      = rng.normal(0.75, 0.05, tumor_core.sum())
    flair[enhancing_tumor] = rng.normal(0.60, 0.04, enhancing_tumor.sum())
    paths["flair"] = _save(np.clip(flair, 0, 1), "flair")

    print(f"[DemoData] Generated synthetic BraTS case at: {output_dir}")
    print(f"  Shape: {shape}  Spacing: {spacing} mm")
    print(f"  Tumor (whole): {whole_tumor.sum()} voxels")
    print(f"  Tumor core:   {tumor_core.sum()} voxels")
    print(f"  Enhancing:    {enhancing_tumor.sum()} voxels")
    print("  NOTE: SYNTHETIC DATA — not from a real patient.")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic BraTS-compatible demo NIfTI volumes."
    )
    parser.add_argument("--output-dir", default="demo_data/case001",
                        help="Output directory")
    parser.add_argument("--shape", nargs=3, type=int, default=[64, 64, 64],
                        metavar=("H", "W", "D"), help="Volume shape")
    parser.add_argument("--spacing", nargs=3, type=float, default=[2.0, 2.0, 2.0],
                        metavar=("dx", "dy", "dz"), help="Voxel spacing in mm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = generate_synthetic_brats_case(
        output_dir=args.output_dir,
        shape=tuple(args.shape),
        spacing=tuple(args.spacing),
        seed=args.seed,
    )
    print("\nGenerated files:")
    for mod, p in paths.items():
        print(f"  {mod}: {p}")
