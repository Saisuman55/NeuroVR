"""
Evaluation Metrics for 2D Brain Tumor Segmentation.

Computes per-region (WT, TC, ET) metrics:
  - Dice Score (F1)
  - IoU / Jaccard Index
  - Precision
  - Recall / Sensitivity
  - Specificity
  - HD95 (Hausdorff Distance 95th percentile) in mm

HD95 is the standard BraTS challenge evaluation metric alongside Dice.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Region names — index 0=WT, 1=TC, 2=ET
REGION_NAMES = ["wt", "tc", "et"]
REGION_LABELS = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]


# ──────────────────────────────────────────────────────────────────────────────
# HD95 computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_hd95(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing_mm: Optional[Tuple[float, float]] = None,
) -> float:
    """Compute Hausdorff Distance 95th percentile (HD95) between two binary masks.

    HD95 measures the 95th percentile of the directed Hausdorff distances,
    which is more robust to outliers than the full Hausdorff distance.

    Args:
        pred_mask: Binary predicted mask (H, W) or (H, W, D) — bool or uint8.
        target_mask: Binary ground truth mask — same shape.
        spacing_mm: Physical voxel spacing (mm per voxel) for each dimension.
            If None, uses isotropic spacing of 1.0.

    Returns:
        HD95 in mm. Returns 0.0 if both masks are empty,
        or a large value (374 mm, ~max diagonal of 240³ @ 1mm) if
        one mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    # Both empty → perfect match
    if not pred.any() and not target.any():
        return 0.0

    # One empty → maximum possible distance
    if not pred.any() or not target.any():
        return 374.0  # ~max diagonal of 240mm brain volume

    if spacing_mm is None:
        spacing_mm = tuple(1.0 for _ in pred.shape)

    # Distance transform: distance from each voxel to nearest mask boundary
    # using physical spacing for correct mm distances
    dt_pred = distance_transform_edt(~pred, sampling=spacing_mm)
    dt_target = distance_transform_edt(~target, sampling=spacing_mm)

    # Directed Hausdorff: for each surface voxel of pred, find nearest in target
    pred_surface = pred & (distance_transform_edt(pred, sampling=spacing_mm) == 1)
    target_surface = target & (distance_transform_edt(target, sampling=spacing_mm) == 1)

    if not pred_surface.any() or not target_surface.any():
        # Fallback if surface detection fails (very small masks)
        pred_surface = pred
        target_surface = target

    # Collect surface-to-surface distances
    d_pred_to_target = dt_target[pred_surface]
    d_target_to_pred = dt_pred[target_surface]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


# ──────────────────────────────────────────────────────────────────────────────
# Per-region metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute Dice score between two binary arrays."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    intersection = (pred_b & target_b).sum()
    union = pred_b.sum() + target_b.sum()
    if union == 0:
        return 1.0  # Both empty — perfect score
    return float((2 * intersection + eps) / (union + eps))


def compute_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute IoU / Jaccard index between two binary arrays."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    intersection = (pred_b & target_b).sum()
    union = (pred_b | target_b).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def compute_precision_recall(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """Compute precision and recall."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = (pred_b & target_b).sum()
    fp = (pred_b & ~target_b).sum()
    fn = (~pred_b & target_b).sum()
    tn = (~pred_b & ~target_b).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(precision), float(recall)


def compute_specificity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute specificity (true negative rate)."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tn = (~pred_b & ~target_b).sum()
    fp = (pred_b & ~target_b).sum()
    return float((tn + eps) / (tn + fp + eps))


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    spacing_mm: Optional[Tuple] = None,
    compute_hd95_flag: bool = True,
) -> Dict[str, float]:
    """Compute all segmentation metrics for a single (pred, target) pair.

    Args:
        pred: Predicted tensor [3, H, W] — sigmoid probabilities.
        target: Ground truth tensor [3, H, W] — binary (0/1).
        threshold: Binarization threshold for predictions.
        spacing_mm: Physical spacing (mm per pixel) for HD95.
        compute_hd95_flag: If False, skip HD95 (much faster).

    Returns:
        Dict with keys like "wt_dice", "tc_iou", "et_hd95", "mean_dice", etc.
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target

    # Enforce containment on predictions before computing metrics
    from data.label_utils import enforce_containment
    p_wt = (pred_np[0] > threshold).astype(np.uint8)
    p_tc = (pred_np[1] > threshold).astype(np.uint8)
    p_et = (pred_np[2] > threshold).astype(np.uint8)
    p_wt, p_tc, p_et = enforce_containment(p_wt, p_tc, p_et)
    pred_bin = [p_wt, p_tc, p_et]

    t_wt = (target_np[0] > 0.5).astype(np.uint8)
    t_tc = (target_np[1] > 0.5).astype(np.uint8)
    t_et = (target_np[2] > 0.5).astype(np.uint8)
    target_bin = [t_wt, t_tc, t_et]

    result: Dict[str, float] = {}
    dice_values = []

    for i, region in enumerate(REGION_NAMES):
        p = pred_bin[i]
        t = target_bin[i]

        dice = compute_dice(p, t)
        iou = compute_iou(p, t)
        precision, recall = compute_precision_recall(p, t)
        specificity = compute_specificity(p, t)

        result[f"{region}_dice"] = dice
        result[f"{region}_iou"] = iou
        result[f"{region}_precision"] = precision
        result[f"{region}_recall"] = recall
        result[f"{region}_specificity"] = specificity
        dice_values.append(dice)

        if compute_hd95_flag:
            hd95 = compute_hd95(p, t, spacing_mm)
            result[f"{region}_hd95"] = hd95

    result["mean_dice"] = float(np.mean(dice_values))

    if compute_hd95_flag:
        result["mean_hd95"] = float(np.mean([result[f"{r}_hd95"] for r in REGION_NAMES]))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Metrics accumulator
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulate per-batch metrics and compute epoch averages.

    Example::

        tracker = MetricsTracker()
        for batch in val_loader:
            metrics = compute_metrics(pred, target)
            tracker.update(metrics)
        epoch_metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Accumulate one batch/sample of metrics."""
        for k, v in metrics.items():
            if k not in self._sums:
                self._sums[k] = 0.0
                self._counts[k] = 0
            if v is not None and not np.isnan(v):
                self._sums[k] += v
                self._counts[k] += 1

    def compute(self) -> Dict[str, float]:
        """Compute epoch-averaged metrics."""
        return {
            k: (self._sums[k] / self._counts[k] if self._counts[k] > 0 else 0.0)
            for k in self._sums
        }

    def reset(self) -> None:
        """Reset all accumulators for the next epoch."""
        self._sums.clear()
        self._counts.clear()

    def format_summary(self, metrics: Optional[Dict] = None) -> str:
        """Format metrics as a compact string for console logging."""
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
        return f"Mean Dice={mean_dice:.4f} | " + " | ".join(parts)
