"""
MONAI BraTS MRI Segmentation Model Loader for NeuroVR 3D.

Downloads, caches, verifies, and loads the official MONAI BraTS
segmentation model. The model is loaded once and cached as a module-level
singleton to avoid repeated loading during multi-request sessions.

Model: brats_mri_segmentation (MONAI Model Zoo)
Input: 4-channel MRI [T1, T1ce, T2, FLAIR], shape [B, 4, H, W, D]
Output: 3-channel probability maps [TC, WT, ET], shape [B, 3, H, W, D]
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Disable MPS at module import time — before torch is loaded.
# Prevents Metal framework mutex deadlock in Flask background threads (macOS Python 3.9).
os.environ["DISABLE_MPS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
MODEL_CACHE_DIR = BASE_DIR / "models" / "monai" / "brats_mri_segmentation"

MONAI_BUNDLE_NAME = "brats_mri_segmentation"
MONAI_BUNDLE_VERSION = "0.4.2"

# Module-level singleton
_cached_model: Optional[nn.Module] = None
_cached_device: Optional[torch.device] = None


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────


def get_device(device_str: str = "auto") -> torch.device:
    """Select the best available compute device.

    NOTE: MPS is intentionally skipped on macOS to avoid a Metal framework
    mutex deadlock ([mutex.cc] Lock blocking) in Flask background threads
    on Python 3.9. CPU is used instead. On Linux/CUDA servers 'auto' selects CUDA.

    Args:
        device_str: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device
    """
    # Always use CPU if DISABLE_MPS is set (Flask threading safety)
    if os.environ.get("DISABLE_MPS") == "1":
        device = torch.device("cpu")
        print("[ModelLoader] Device: CPU (MPS disabled for Flask thread safety)")
        return device

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[ModelLoader] Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            # Skip MPS — causes mutex deadlock in Flask threads on macOS Python 3.9
            device = torch.device("cpu")
            print("[ModelLoader] Device: CPU (MPS skipped — Flask threading safety)")
    else:
        device = torch.device(device_str)
        print(f"[ModelLoader] Device: {device} (user-specified)")

    return device


# ──────────────────────────────────────────────────────────────────────────────
# Model download and verification
# ──────────────────────────────────────────────────────────────────────────────


def _model_dir_exists() -> bool:
    """Check if the cached MONAI bundle directory exists and has content."""
    if not MODEL_CACHE_DIR.exists():
        return False
    # Check for at least a configs/ or models/ subdir (MONAI bundle structure)
    has_content = any(MODEL_CACHE_DIR.iterdir())
    return has_content


def download_monai_bundle(force: bool = False) -> Path:
    """Download the MONAI BraTS segmentation bundle if not cached.

    Args:
        force: If True, re-download even if already cached.

    Returns:
        Path to the bundle directory.
    """
    if not force and _model_dir_exists():
        print(f"[ModelLoader] Bundle cached at: {MODEL_CACHE_DIR}")
        return MODEL_CACHE_DIR

    try:
        from monai.bundle import download
    except ImportError as exc:
        raise ImportError(
            "MONAI is required. Install with: pip install monai[all]"
        ) from exc

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[ModelLoader] Downloading MONAI bundle '{MONAI_BUNDLE_NAME}' "
        f"v{MONAI_BUNDLE_VERSION} → {MODEL_CACHE_DIR} ..."
    )
    print("[ModelLoader] This may take several minutes (~500 MB).")

    download(
        name=MONAI_BUNDLE_NAME,
        version=MONAI_BUNDLE_VERSION,
        bundle_dir=str(MODEL_CACHE_DIR.parent),
        source="monaihosting",
    )

    print(f"[ModelLoader] ✔ Bundle downloaded: {MODEL_CACHE_DIR}")
    return MODEL_CACHE_DIR


def _load_bundle_model(bundle_dir: Path, device: torch.device) -> nn.Module:
    """Instantiate and load a MONAI bundle model from its config.

    Args:
        bundle_dir: Path to the bundle directory (contains configs/).
        device: Target device.

    Returns:
        Loaded nn.Module in eval mode.
    """
    try:
        from monai.bundle import ConfigParser
    except ImportError as exc:
        raise ImportError("MONAI is required: pip install monai[all]") from exc

    config_path = bundle_dir / "configs" / "inference.json"
    if not config_path.exists():
        # Try yaml variant
        config_path = bundle_dir / "configs" / "inference.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot find inference config in bundle: {bundle_dir}. "
            "Try re-downloading the bundle."
        )

    parser = ConfigParser()
    parser.read_config(str(config_path))

    # Override device in config
    parser["device"] = str(device)

    # Build network definition
    net = parser.get_parsed_content("network_def", instantiate=True)

    # Load pretrained weights
    weights_path = bundle_dir / "models" / "model.pt"
    if not weights_path.exists():
        weights_path = bundle_dir / "models" / "model.pth"
    if not weights_path.exists():
        # Check for any .pt or .pth in models/
        models_dir = bundle_dir / "models"
        pts = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
        if pts:
            weights_path = pts[0]
        else:
            raise FileNotFoundError(
                f"No model weights found in {models_dir}. "
                "Try re-downloading the bundle."
            )

    print(f"[ModelLoader] Loading weights from: {weights_path}")
    state = torch.load(str(weights_path), map_location=device, weights_only=True)

    # Unwrap common checkpoint wrappers
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in state:
                state = state[key]
                break

    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    print(f"[ModelLoader] ✔ Model loaded on {device}")
    return net


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_model(
    device_str: str = "auto",
    force_download: bool = False,
) -> Tuple[nn.Module, torch.device]:
    """Get (or create) the singleton MONAI BraTS segmentation model.

    Downloads the bundle on first call, loads once, caches in memory.

    Args:
        device_str: Device preference ('auto', 'cuda', 'mps', 'cpu').
        force_download: Re-download bundle even if cached.

    Returns:
        Tuple of (model, device).
    """
    global _cached_model, _cached_device

    device = get_device(device_str)

    # Return cached model if device matches
    if _cached_model is not None and _cached_device == device:
        return _cached_model, _cached_device

    # Download if needed
    bundle_dir = download_monai_bundle(force=force_download)

    # Load model
    model = _load_bundle_model(bundle_dir, device)

    _cached_model = model
    _cached_device = device
    return _cached_model, _cached_device


def clear_model_cache() -> None:
    """Release the cached model from memory."""
    global _cached_model, _cached_device
    _cached_model = None
    _cached_device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[ModelLoader] Model cache cleared.")
