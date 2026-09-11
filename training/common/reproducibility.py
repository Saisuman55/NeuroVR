"""
Reproducibility Logging for NeuroVR Training.

Captures a complete snapshot of every training run:
  - Git commit hash + dirty state
  - Python, PyTorch, MONAI versions
  - Full hyperparameter namespace
  - Dataset fingerprint and patient splits
  - Model architecture summary
  - Environment (pip packages)

Every training run saves a manifest.json that can reproduce
the exact experiment from scratch.

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Manifest dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingManifest:
    """Complete reproducibility snapshot for one training run."""

    # Run identity
    run_id: str = ""
    timestamp: str = ""
    pipeline: str = ""               # "2d_segmentation" | "3d_segmentation"

    # Git provenance
    git_hash: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    git_remote: str = ""

    # Environment
    python_version: str = ""
    torch_version: str = ""
    monai_version: str = ""
    cuda_version: str = ""
    mps_available: bool = False
    hostname: str = ""
    os_info: str = ""

    # Hyperparameters (full CLI args dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Dataset
    dataset_root: str = ""
    dataset_fingerprint: str = ""
    dataset_split_counts: Dict[str, int] = field(default_factory=dict)
    dataset_split_patients: Dict[str, List[str]] = field(default_factory=dict)

    # Model
    model_architecture: str = ""
    total_parameters: int = 0
    trainable_parameters: int = 0

    # Results (filled in at end of training)
    best_epoch: int = 0
    best_val_metrics: Dict[str, float] = field(default_factory=dict)

    # Disclaimer
    disclaimer: str = (
        "Research use only — not clinically validated. "
        "NeuroVR is a B.Tech educational prototype."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return False


def _git_remote() -> str:
    try:
        return subprocess.check_output(
            ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Version helpers
# ──────────────────────────────────────────────────────────────────────────────

def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def _monai_version() -> str:
    try:
        import monai
        return monai.__version__
    except ImportError:
        return "not installed"


def _cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "N/A"
    except Exception:
        return "N/A"


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Model summary
# ──────────────────────────────────────────────────────────────────────────────

def get_model_param_counts(model: Any) -> tuple[int, int]:
    """Return (total_params, trainable_params).

    Args:
        model: PyTorch nn.Module.

    Returns:
        (total, trainable) parameter counts.
    """
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    except Exception:
        return 0, 0


# ──────────────────────────────────────────────────────────────────────────────
# Environment snapshot
# ──────────────────────────────────────────────────────────────────────────────

def log_environment() -> Dict[str, str]:
    """Capture all installed pip packages and versions.

    Returns:
        Dict mapping package_name → version string.
    """
    try:
        import importlib.metadata
        packages = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            version = dist.metadata["Version"]
            if name:
                packages[name.lower()] = version
        return packages
    except Exception:
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                stderr=subprocess.DEVNULL
            ).decode()
            pkgs = json.loads(out)
            return {p["name"].lower(): p["version"] for p in pkgs}
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# Manifest construction
# ──────────────────────────────────────────────────────────────────────────────

def build_manifest(
    pipeline: str,
    hyperparameters: Dict[str, Any],
    dataset_fingerprint: str = "",
    dataset_root: str = "",
    split_counts: Optional[Dict[str, int]] = None,
    split_patients: Optional[Dict[str, List[str]]] = None,
    model: Optional[Any] = None,
) -> TrainingManifest:
    """Build a TrainingManifest from the current environment.

    Args:
        pipeline: "2d_segmentation" or "3d_segmentation".
        hyperparameters: Full CLI args as a dict.
        dataset_fingerprint: SHA256 fingerprint from dataset_manager.
        dataset_root: Path to dataset root.
        split_counts: {"train": N, "val": N, "test": N}.
        split_patients: {"train": [...ids], "val": [...ids], "test": [...ids]}.
        model: Optional nn.Module for parameter counting.

    Returns:
        TrainingManifest (not yet saved to disk).
    """
    import datetime
    import uuid as _uuid

    run_id = _uuid.uuid4().hex[:12]
    timestamp = datetime.datetime.now().isoformat()

    total_params, trainable_params = (0, 0)
    arch_str = ""
    if model is not None:
        total_params, trainable_params = get_model_param_counts(model)
        try:
            arch_str = repr(model)[:2000]  # Truncate very long reprs
        except Exception:
            arch_str = type(model).__name__

    return TrainingManifest(
        run_id=run_id,
        timestamp=timestamp,
        pipeline=pipeline,
        git_hash=_git_hash(),
        git_branch=_git_branch(),
        git_dirty=_git_dirty(),
        git_remote=_git_remote(),
        python_version=sys.version,
        torch_version=_torch_version(),
        monai_version=_monai_version(),
        cuda_version=_cuda_version(),
        mps_available=_mps_available(),
        hostname=platform.node(),
        os_info=platform.platform(),
        hyperparameters=hyperparameters,
        dataset_root=str(dataset_root),
        dataset_fingerprint=dataset_fingerprint,
        dataset_split_counts=split_counts or {},
        dataset_split_patients=split_patients or {},
        model_architecture=arch_str,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
    )


def save_manifest(manifest: TrainingManifest, path: str) -> None:
    """Serialize and save a TrainingManifest to JSON.

    Args:
        manifest: The manifest to save.
        path: Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, default=str)
    print(f"[Reproducibility] Manifest saved: {path}")


def load_manifest(path: str) -> TrainingManifest:
    """Load a TrainingManifest from JSON.

    Args:
        path: Path to manifest.json.

    Returns:
        TrainingManifest dataclass.
    """
    with open(path) as f:
        data = json.load(f)
    return TrainingManifest(**data)


def print_manifest_summary(manifest: TrainingManifest) -> None:
    """Print a human-readable summary of a TrainingManifest."""
    print(f"\n{'─'*55}")
    print(f"  Run ID       : {manifest.run_id}")
    print(f"  Pipeline     : {manifest.pipeline}")
    print(f"  Timestamp    : {manifest.timestamp}")
    print(f"  Git hash     : {manifest.git_hash[:12]} ({'dirty' if manifest.git_dirty else 'clean'})")
    print(f"  PyTorch      : {manifest.torch_version}")
    print(f"  MONAI        : {manifest.monai_version}")
    print(f"  Dataset FP   : {manifest.dataset_fingerprint[:16]}...")
    print(f"  Splits       : {manifest.dataset_split_counts}")
    print(f"  Parameters   : {manifest.trainable_parameters:,} trainable / {manifest.total_parameters:,} total")
    print(f"  Seed         : {manifest.hyperparameters.get('seed', 'N/A')}")
    print(f"{'─'*55}\n")
