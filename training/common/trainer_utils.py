"""
Shared Training Utilities for NeuroVR.

Provides seeding, device selection, checkpointing, early stopping,
and training curve visualization for both 2D and 3D pipelines.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic seeding
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CuDNN (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device(prefer: str = "auto") -> torch.device:
    """Select the best available compute device.

    Args:
        prefer: "auto" | "cuda" | "mps" | "cpu"

    Returns:
        torch.device
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Trainer] Device: CUDA ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available() and os.environ.get("DISABLE_MPS") != "1":
            device = torch.device("mps")
            print("[Trainer] Device: MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("[Trainer] Device: CPU")
    else:
        device = torch.device(prefer)
        print(f"[Trainer] Device: {device} (user-specified)")

    return device


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Validation metrics dict.
        path: Output file path (.pt).
        scheduler: Optional LR scheduler.
        extra: Optional extra metadata to embed.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "extra": extra or {},
        "disclaimer": "Research use only — not clinically validated.",
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional LR scheduler to restore.
        device: Target device.

    Returns:
        (epoch, metrics) from the checkpoint.
    """
    map_device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"[Trainer] Loaded checkpoint from epoch {epoch}: {path}")
    return epoch, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitor a metric and signal when training should stop.

    Supports "max" mode (e.g., Dice) and "min" mode (e.g., loss).

    Example::

        es = EarlyStopping(patience=10, mode="max")
        for epoch in ...:
            val_dice = ...
            if es(val_dice):
                print("Early stopping triggered")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        """
        Args:
            patience: Epochs to wait after last improvement.
            min_delta: Minimum improvement to count as improvement.
            mode: "max" for metrics that should increase, "min" for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        """Update and check if training should stop.

        Args:
            metric: Current epoch metric value.

        Returns:
            True if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                print(
                    f"[EarlyStopping] Triggered after {self.counter} epochs "
                    f"without improvement (best={self.best:.4f})"
                )
                return True

        return False

    def state_dict(self) -> Dict:
        return {
            "counter": self.counter,
            "best": self.best,
            "triggered": self.triggered,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.counter = state["counter"]
        self.best = state["best"]
        self.triggered = state["triggered"]


# ──────────────────────────────────────────────────────────────────────────────
# Training curve visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: List[Dict[str, float]],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Generate and save training/validation plots.

    Args:
        history: List of per-epoch metric dicts with keys like
            "train_loss", "val_loss", "val_wt_dice", etc.
        output_dir: Directory to save PNG files.
        prefix: Optional prefix for file names.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Trainer] matplotlib not available — skipping plots")
        return

    if not history:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

    def _plot_metric(key: str, ylabel: str, filename: str) -> None:
        values = [h.get(key) for h in history]
        if all(v is None for v in values):
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, values, "o-", linewidth=2, markersize=4, label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix} {ylabel}" if prefix else ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    # Loss
    _plot_metric("train_loss", "Loss", f"{prefix}_loss.png" if prefix else "loss.png")

    # Dice scores
    for region in ("wt", "tc", "et", "mean"):
        key = f"val_{region}_dice"
        _plot_metric(key, f"Dice ({region.upper()})", f"{prefix}_{key}.png" if prefix else f"{key}.png")

    # HD95
    for region in ("wt", "tc", "et", "mean"):
        key = f"val_{region}_hd95"
        _plot_metric(key, f"HD95 mm ({region.upper()})", f"{prefix}_{key}.png" if prefix else f"{key}.png")

    # Combined Dice plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for region, color in [("wt", "#0ea5e9"), ("tc", "#ef4444"), ("et", "#f59e0b")]:
        key = f"val_{region}_dice"
        values = [h.get(key) for h in history]
        if any(v is not None for v in values):
            ax.plot(epochs, values, "o-", linewidth=2, markersize=4,
                    label=f"{region.upper()} Dice", color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title(f"{prefix} Validation Dice" if prefix else "Validation Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fname = f"{prefix}_val_dice_combined.png" if prefix else "val_dice_combined.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)

    print(f"[Trainer] Training curves saved to {out_dir}")


def save_history_csv(history: List[Dict], path: str) -> None:
    """Save training history to a CSV file.

    Args:
        history: List of per-epoch metric dicts.
        path: Output CSV path.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(history)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except ImportError:
        # Fallback: write CSV manually
        if not history:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        keys = list(history[0].keys())
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in history:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
