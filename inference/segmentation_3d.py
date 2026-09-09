"""
3D MRI Segmentation Inference for NeuroVR.

Runs the MONAI BraTS segmentation model on preprocessed 4-channel
MRI tensors using sliding-window inference. Produces three tumor masks:

  - Tumor Core (TC)
  - Whole Tumor (WT)
  - Enhancing Tumor (ET)

The masks are reconstructed into the original MRI coordinate space
using the preprocessing inverse transforms.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.model_loader_3d import get_model
from preprocessing.volume_preprocessor import VolumePreprocessor


# BraTS channel assignment in model output [B, 3, H, W, D]
TC_CHANNEL = 0  # Tumor Core
WT_CHANNEL = 1  # Whole Tumor
ET_CHANNEL = 2  # Enhancing Tumor

DEFAULT_ROI_SIZE = (128, 128, 128)
DEFAULT_OVERLAP = 0.5
DEFAULT_SW_BATCH_SIZE = 1
DEFAULT_THRESHOLD = 0.5


def _sliding_window_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    """Run sliding-window inference, with MONAI fallback to manual tiling.

    Args:
        model: Loaded segmentation model.
        inputs: Input tensor [1, 4, H, W, D] on CPU.
        roi_size: Patch size for sliding window.
        sw_batch_size: Number of patches per forward pass.
        overlap: Fraction of overlap between patches (0–1).
        device: Compute device.

    Returns:
        Prediction tensor [1, 3, H, W, D] on CPU.
    """
    # Move input to device
    inputs = inputs.to(device)

    try:
        from monai.inferers import sliding_window_inference as monai_swi

        with torch.inference_mode():
            output = monai_swi(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        return output.cpu()

    except ImportError:
        # Fallback: single forward pass on full volume (may OOM on large volumes)
        print("[Inference] MONAI not available for sliding-window. Using full-volume inference.")
        with torch.inference_mode():
            # Attempt to fit in memory; if not, raise informative error
            try:
                output = model(inputs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        "GPU out of memory for full-volume inference. "
                        "Install MONAI (pip install monai) to enable sliding-window."
                    ) from e
                raise
        return output.cpu()


def run_3d_segmentation(
    modality_volumes: Dict[str, Dict],
    device_str: str = "auto",
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sw_batch_size: int = DEFAULT_SW_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict:
    """
    Run the full 3D segmentation pipeline on four MRI modalities.

    Args:
        modality_volumes: Dict from nifti_loader.load_brats_case().
        device_str: Device preference ('auto'/'cuda'/'mps'/'cpu').
        roi_size: Sliding-window ROI size.
        overlap: Sliding-window overlap fraction.
        sw_batch_size: Patches per forward pass.
        threshold: Probability threshold for binary mask.

    Returns:
        Dict with keys:
          - tc_mask: np.ndarray uint8 (H, W, D) — Tumor Core
          - wt_mask: np.ndarray uint8 (H, W, D) — Whole Tumor
          - et_mask: np.ndarray uint8 (H, W, D) — Enhancing Tumor
          - tc_prob: np.ndarray float32 — raw TC probabilities (original space)
          - wt_prob: np.ndarray float32 — raw WT probabilities
          - et_prob: np.ndarray float32 — raw ET probabilities
          - affine: np.ndarray (4×4)
          - voxel_spacing: np.ndarray [dx, dy, dz]
          - inference_time_s: float
          - preprocessed_shape: tuple
    """
    t0 = time.time()

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, device = get_model(device_str=device_str)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    preprocessor = VolumePreprocessor(
        target_spacing=(1.0, 1.0, 1.0),
        crop_foreground=True,
        normalize=True,
    )
    tensor, meta = preprocessor.preprocess(modality_volumes)

    # Add batch dimension: [4, H, W, D] → [1, 4, H, W, D]
    tensor = tensor.unsqueeze(0)
    print(f"[Inference] Input tensor shape: {tensor.shape}")

    # ── 3. Sliding-window inference ───────────────────────────────────────────
    # Adjust ROI to not exceed input size
    vol_shape = tensor.shape[2:]
    roi_actual = tuple(min(roi_size[i], vol_shape[i]) for i in range(3))

    print(
        f"[Inference] ROI={roi_actual} overlap={overlap} "
        f"sw_batch={sw_batch_size} device={device}"
    )

    raw_output = _sliding_window_inference(
        model=model,
        inputs=tensor,
        roi_size=roi_actual,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        device=device,
    )  # [1, 3, H', W', D']

    # Apply sigmoid to get probabilities (model may output logits)
    probs = torch.sigmoid(raw_output).squeeze(0).numpy()  # [3, H', W', D']

    # ── 4. Threshold → binary masks in preprocessed space ────────────────────
    tc_prob_pp = probs[TC_CHANNEL]  # (H', W', D')
    wt_prob_pp = probs[WT_CHANNEL]
    et_prob_pp = probs[ET_CHANNEL]

    tc_mask_pp = (tc_prob_pp > threshold).astype(np.uint8)
    wt_mask_pp = (wt_prob_pp > threshold).astype(np.uint8)
    et_mask_pp = (et_prob_pp > threshold).astype(np.uint8)

    # ── 5. Invert transforms → original space ────────────────────────────────
    tc_mask = preprocessor.invert_mask(tc_mask_pp, meta, order=0)
    wt_mask = preprocessor.invert_mask(wt_mask_pp, meta, order=0)
    et_mask = preprocessor.invert_mask(et_mask_pp, meta, order=0)

    # Also invert probability maps (for visualization/heatmaps)
    tc_prob = preprocessor.invert_mask(tc_prob_pp, meta, order=1).astype(np.float32)
    wt_prob = preprocessor.invert_mask(wt_prob_pp, meta, order=1).astype(np.float32)
    et_prob = preprocessor.invert_mask(et_prob_pp, meta, order=1).astype(np.float32)

    t1 = time.time()
    inference_time = t1 - t0

    first_modality = list(modality_volumes.keys())[0]
    affine = modality_volumes[first_modality]["affine"]
    voxel_spacing = modality_volumes[first_modality]["voxel_spacing"]

    result = {
        "tc_mask": tc_mask,
        "wt_mask": wt_mask,
        "et_mask": et_mask,
        "tc_prob": tc_prob,
        "wt_prob": wt_prob,
        "et_prob": et_prob,
        "affine": affine,
        "voxel_spacing": voxel_spacing,
        "inference_time_s": inference_time,
        "preprocessed_shape": tuple(vol_shape),
    }

    tc_vox = int(tc_mask.sum())
    wt_vox = int(wt_mask.sum())
    et_vox = int(et_mask.sum())

    print(
        f"[Inference] ✔ Done in {inference_time:.1f}s | "
        f"TC={tc_vox} vox | WT={wt_vox} vox | ET={et_vox} vox"
    )

    if wt_vox == 0:
        print("[Inference] No tumor detected (all masks empty).")

    return result
