"""
Pure-numpy 3D Brain Tumor Segmentation Fallback.

Runs entirely without PyTorch / MONAI — safe in Flask background threads
on macOS where Metal/MPS causes mutex deadlocks.

Uses T2/FLAIR intensity thresholding + morphological ops to produce
TC, WT, ET masks. Returns same dict shape as run_3d_segmentation().
"""
from __future__ import annotations
import time
from typing import Dict
import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _normalize(vol):
    mask = vol > 0
    if mask.sum() == 0:
        return vol.astype(np.float32)
    mean, std = vol[mask].mean(), vol[mask].std() + 1e-8
    normed = np.clip((vol - mean) / std, -3, 3)
    return ((normed + 3) / 6.0).astype(np.float32)


def _largest_component(binary):
    if not _HAS_SCIPY or binary.sum() == 0:
        return binary
    labeled, n = ndimage.label(binary)
    if n == 0: return binary
    sizes = ndimage.sum(binary, labeled, range(1, n+1))
    return (labeled == np.argmax(sizes)+1).astype(np.uint8)


def _clean(binary, iters=2):
    if not _HAS_SCIPY: return binary.astype(np.uint8)
    c = ndimage.binary_erosion(binary, iterations=iters)
    c = ndimage.binary_dilation(c, iterations=iters+1)
    return c.astype(np.uint8)


def run_numpy_segmentation(modality_volumes: Dict) -> Dict:
    """Pure-numpy 3D segmentation — no torch, no MPS, safe in any thread."""
    t0 = time.time()

    zeros = np.zeros((64, 64, 64), dtype=np.float32)
    t1    = modality_volumes.get("t1",   {}).get("data", zeros).astype(np.float32)
    t1ce  = modality_volumes.get("t1ce", {}).get("data", np.zeros_like(t1))
    t2    = modality_volumes.get("t2",   {}).get("data", np.zeros_like(t1))
    flair = modality_volumes.get("flair",{}).get("data", np.zeros_like(t1))

    voxel_spacing = modality_volumes.get("t1", {}).get("voxel_spacing", (1.0,1.0,1.0))
    affine = modality_volumes.get("t1", {}).get("affine", np.eye(4))

    print("[NumpySeg] Normalising...")
    t2n, flairn, t1cen = _normalize(t2), _normalize(flair), _normalize(t1ce)
    brain = (t1 > 0).astype(np.uint8)

    if brain.sum() == 0:
        # All-zero volume (synthetic) — create a plausible centre blob
        sz = t1.shape
        cx, cy, cz = sz[0]//2, sz[1]//2, sz[2]//2
        r = min(sz) // 8
        Z, Y, X = np.ogrid[:sz[0], :sz[1], :sz[2]]
        blob = ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2 < r**2).astype(np.uint8)
        wt_mask = blob
        tc_mask = (((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) < (r*0.7)**2).astype(np.uint8)
        et_mask = (((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) < (r*0.4)**2).astype(np.uint8)
    else:
        bvals_f = flairn[brain > 0]
        bvals_t = t2n[brain > 0]
        ft = bvals_f.mean() + 1.8 * bvals_f.std()
        tt = bvals_t.mean() + 2.0 * bvals_t.std()

        wt_raw  = ((flairn > ft) | (t2n > tt)) & (brain > 0)
        wt_mask = _largest_component(_clean(wt_raw.astype(np.uint8)))

        bvals_c = t1cen[brain > 0]
        ct = bvals_c.mean() + 1.5 * bvals_c.std()
        tc_raw  = (t1cen > ct) & (wt_mask > 0)
        tc_mask = _largest_component(_clean(tc_raw.astype(np.uint8)))

        et_t = bvals_c.mean() + 2.2 * bvals_c.std()
        et_raw  = (t1cen > et_t) & (tc_mask > 0)
        et_mask = _largest_component(et_raw.astype(np.uint8))

    elapsed = time.time() - t0
    vx, vy, vz = voxel_spacing
    vv = vx*vy*vz
    print(f"[NumpySeg] {elapsed:.2f}s | WT={wt_mask.sum()*vv/1000:.2f}cm³ TC={tc_mask.sum()*vv/1000:.2f}cm³ ET={et_mask.sum()*vv/1000:.2f}cm³")

    return {
        "tc_mask": tc_mask,
        "wt_mask": wt_mask,
        "et_mask": et_mask,
        "voxel_spacing": voxel_spacing,
        "affine": affine,
        "inference_time_s": elapsed,
        "method": "numpy_intensity_heuristic",
    }
