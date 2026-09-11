"""
Evaluation Metrics for 3D Brain Tumor Segmentation.

Extends the 2D metrics with 3D-aware HD95 computation and
containment validation on volumetric predictions.

Per-region (WT, TC, ET):
  - Dice Score
  - IoU / Jaccard
  - Precision, Recall/Sensitivity, Specificity
  - HD95 (Hausdorff Distance 95th percentile, in mm)
  - Containment validation: ET ⊆ TC ⊆ WT

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REGION_NAMES = ["wt", "tc", "et"]


# ──────────────────────────────────────────────────────────────────────────────
# 3D HD95
# ──────────────────────────────────────────────────────────────────────────────

def compute_hd95_3d(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing_mm: Optional[Tuple[float, float, float]] = None,
) -> float:
    """Compute HD95 between two 3D binary masks.

    Args:
        pred_mask: Predicted binary mask (H, W, D) — bool or uint8.
        target_mask: Ground truth binary mask — same shape.
        spacing_mm: Physical voxel spacing (mm) [dH, dW, dD].
            Defaults to isotropic 1mm.

    Returns:
        HD95 in mm. Returns 0.0 if both empty, 374.0 if one is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return 374.0  # ~max diagonal of 240mm brain at 1mm spacing

    if spacing_mm is None:
        spacing_mm = (1.0, 1.0, 1.0)

    dt_target = distance_transform_edt(~target, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~pred, sampling=spacing_mm)

    # Surface voxels: foreground voxels adjacent to background
    from scipy.ndimage import binary_erosion
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred & ~binary_erosion(pred, structure=struct, border_value=1)
    target_surface = target & ~binary_erosion(target, structure=struct, border_value=1)

    if not pred_surface.any() or not target_surface.any():
        pred_surface = pred
        target_surface = target

    d1 = dt_target[pred_surface]
    d2 = dt_pred[target_surface]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────────────

def _dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p, t = pred.astype(bool), target.astype(bool)
    inter = (p & t).sum()
    union = p.sum() + t.sum()
    return 1.0 if union == 0 else float((2 * inter + eps) / (union + eps))


def _iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p, t = pred.astype(bool), target.astype(bool)
    inter = (p & t).sum()
    union = (p | t).sum()
    return 1.0 if union == 0 else float((inter + eps) / (union + eps))


def _precision_recall(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    p, t = pred.astype(bool), target.astype(bool)
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    return float((tp + eps) / (tp + fp + eps)), float((tp + eps) / (tp + fn + eps))


def _specificity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p, t = pred.astype(bool), target.astype(bool)
    tn = (~p & ~t).sum()
    fp = (p & ~t).sum()
    return float((tn + eps) / (tn + fp + eps))


def validate_prediction_containment(
    pred_wt: np.ndarray,
    pred_tc: np.ndarray,
    pred_et: np.ndarray,
) -> Dict[str, int]:
    """Check ET ⊆ TC ⊆ WT for a single prediction volume.

    Returns violation counts (0 = valid).
    """
    et_outside_tc = int(np.count_nonzero(pred_et.astype(bool) & ~pred_tc.astype(bool)))
    tc_outside_wt = int(np.count_nonzero(pred_tc.astype(bool) & ~pred_wt.astype(bool)))
    return {"et_outside_tc": et_outside_tc, "tc_outside_wt": tc_outside_wt}


def compute_metrics_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    spacing_mm: Optional[Tuple[float, float, float]] = None,
    compute_hd95_flag: bool = True,
) -> Dict[str, float]:
    """Compute all segmentation metrics for one 3D prediction.

    Args:
        pred: [3, H, W, D] sigmoid probabilities.
        target: [3, H, W, D] binary ground truth.
        threshold: Binarization threshold.
        spacing_mm: Voxel spacing (mm) for HD95.
        compute_hd95_flag: Skip HD95 if False (much faster per batch).

    Returns:
        Dict with per-region and mean metrics.
    """
    from data.label_utils import enforce_containment

    pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target

    # Binarize predictions with containment enforcement
    p_wt = (pred_np[0] > threshold).astype(np.uint8)
    p_tc = (pred_np[1] > threshold).astype(np.uint8)
    p_et = (pred_np[2] > threshold).astype(np.uint8)
    p_wt, p_tc, p_et = enforce_containment(p_wt, p_tc, p_et)
    preds = [p_wt, p_tc, p_et]

    t_wt = (target_np[0] > 0.5).astype(np.uint8)
    t_tc = (target_np[1] > 0.5).astype(np.uint8)
    t_et = (target_np[2] > 0.5).astype(np.uint8)
    targets = [t_wt, t_tc, t_et]

    result: Dict[str, float] = {}
    dice_values = []

    for i, region in enumerate(REGION_NAMES):
        p_r, t_r = preds[i], targets[i]
        dice = _dice(p_r, t_r)
        iou = _iou(p_r, t_r)
        precision, recall = _precision_recall(p_r, t_r)
        spec = _specificity(p_r, t_r)

        result[f"{region}_dice"] = dice
        result[f"{region}_iou"] = iou
        result[f"{region}_precision"] = precision
        result[f"{region}_recall"] = recall
        result[f"{region}_specificity"] = spec
        dice_values.append(dice)

        if compute_hd95_flag:
            hd95 = compute_hd95_3d(p_r, t_r, spacing_mm)
            result[f"{region}_hd95"] = hd95

    result["mean_dice"] = float(np.mean(dice_values))
    if compute_hd95_flag:
        result["mean_hd95"] = float(np.mean([result[f"{r}_hd95"] for r in REGION_NAMES]))

    # Containment validation
    violations = validate_prediction_containment(p_wt, p_tc, p_et)
    result["containment_et_outside_tc"] = float(violations["et_outside_tc"])
    result["containment_tc_outside_wt"] = float(violations["tc_outside_wt"])

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Metrics tracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker3D:
    """Accumulate per-case 3D metrics across a validation epoch."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if v is not None and not np.isnan(v):
                self._sums[k] = self._sums.get(k, 0.0) + v
                self._counts[k] = self._counts.get(k, 0) + 1

    def compute(self) -> Dict[str, float]:
        return {
            k: (self._sums[k] / self._counts[k] if self._counts[k] > 0 else 0.0)
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def format_summary(self, metrics: Optional[Dict] = None) -> str:
        m = metrics or self.compute()
        parts = []
        for region in REGION_NAMES:
            dice = m.get(f"{region}_dice", float("nan"))
            hd95 = m.get(f"{region}_hd95")
            s = f"{region.upper()} Dice={dice:.4f}"
            if hd95 is not None:
                s += f" HD95={hd95:.1f}mm"
            parts.append(s)
        mean_dice = m.get("mean_dice", float("nan"))
        return f"Mean={mean_dice:.4f} | " + " | ".join(parts)
