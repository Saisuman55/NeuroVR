"""
2D Brain Tumor Segmentation Training Script for NeuroVR.

Trains a 2D U-Net on BraTS MRI slices with patient-level splits,
MONAI augmentations, Dice+CE loss, and full metrics including HD95.

Usage::

    # Train on BraTS dataset
    python -m training.segmentation_2d.train --data_dir data/raw/brats

    # Dry run with synthetic data (no real dataset needed)
    python -m training.segmentation_2d.train --dry_run

    # Resume from checkpoint
    python -m training.segmentation_2d.train --data_dir data/raw/brats \\
        --resume results/2d/checkpoints/last.pt

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project root
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroVR 2D Segmentation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--data_dir", type=str, default="data/raw/brats",
                   help="Path to BraTS dataset root (or BRATS_DATASET_PATH env var)")
    p.add_argument("--orientation", default="axial",
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--patch_size", type=int, default=240,
                   help="Square 2D patch size (H=W)")
    p.add_argument("--tumor_ratio", type=float, default=0.6,
                   help="Fraction of slices containing tumor per patient")
    p.add_argument("--max_slices", type=int, default=None,
                   help="Max slices per patient (None=all)")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="dice_ce",
                   choices=["dice", "dice_ce", "dice_focal"])
    p.add_argument("--early_stopping", type=int, default=10,
                   help="Early stopping patience (0=disabled)")

    # Dataset split
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--num_res_units", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="results/2d")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save periodic checkpoint every N epochs")

    # DataLoader
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])

    # Dry run
    p.add_argument("--dry_run", action="store_true",
                   help="Run 1 epoch with synthetic data to validate pipeline")
    p.add_argument("--hd95", action="store_true", default=True,
                   help="Compute HD95 during validation (slower)")
    p.add_argument("--no_hd95", action="store_false", dest="hd95")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data for dry-run
# ──────────────────────────────────────────────────────────────────────────────

class _SyntheticDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for pipeline smoke testing."""

    def __init__(self, n: int = 50, patch_size: int = 64) -> None:
        self.n = n
        self.patch_size = patch_size
        rng = np.random.default_rng(42)
        self.images = rng.standard_normal((n, 4, patch_size, patch_size)).astype(np.float32)
        masks = np.zeros((n, 3, patch_size, patch_size), dtype=np.float32)
        # Add synthetic tumor blobs
        for i in range(n):
            cx, cy = patch_size // 2, patch_size // 2
            r = patch_size // 6
            yy, xx = np.mgrid[:patch_size, :patch_size]
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            masks[i, 0] = (d < r * 2.5).astype(np.float32)  # WT
            masks[i, 1] = (d < r * 1.5).astype(np.float32)  # TC
            masks[i, 2] = (d < r).astype(np.float32)        # ET
        self.masks = masks

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.images[idx]), torch.from_numpy(self.masks[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Training & validation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    compute_hd95_flag: bool = True,
) -> Dict[str, float]:
    """Run validation and return metrics dict."""
    from training.segmentation_2d.metrics import MetricsTracker, compute_metrics

    model.eval()
    tracker = MetricsTracker()
    total_loss = 0.0
    n_batches = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += float(loss)
            n_batches += 1

            # Compute metrics on CPU
            probs = torch.sigmoid(logits).cpu()
            targets = masks.cpu()

            for i in range(probs.shape[0]):  # Iterate over batch
                m = compute_metrics(
                    probs[i], targets[i],
                    compute_hd95_flag=compute_hd95_flag,
                )
                tracker.update(m)

    metrics = tracker.compute()
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Setup
    from training.common.trainer_utils import (
        set_seed, get_device, save_checkpoint, load_checkpoint,
        EarlyStopping, plot_training_curves, save_history_csv,
    )
    from training.segmentation_2d.model import create_2d_unet, print_model_summary
    from training.segmentation_2d.losses import get_loss

    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NeuroVR 2D Segmentation Training")
    print(f"  DISCLAIMER: Research prototype — NOT for clinical use")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[Train] DRY RUN — using synthetic data")
        train_ds = _SyntheticDataset(n=80, patch_size=args.patch_size)
        val_ds = _SyntheticDataset(n=20, patch_size=args.patch_size)
        patient_dirs_per_split = {"train": [], "val": [], "test": []}
        dataset_fingerprint = "dry_run_synthetic"
    else:
        from data.dataset_manager import BraTSDatasetManager

        mgr = BraTSDatasetManager(dataset_root=args.data_dir)
        if not mgr.check_availability():
            print(
                "\n[Train] ERROR: Dataset not found. Please download BraTS dataset.\n"
                "  See: data/README_DATASETS.md\n"
                f"  Expected path: {args.data_dir}\n"
                "  Or set env var: BRATS_DATASET_PATH=/path/to/BraTS\n"
            )
            sys.exit(1)

        valid_patients, _ = mgr.validate_structure()
        print(f"[Train] {len(valid_patients)} valid patients found")

        splits = mgr.create_splits(
            train=args.train_frac,
            val=args.val_frac,
            test=args.test_frac,
            seed=args.seed,
            save_path=str(out_dir / "splits.json"),
        )
        dataset_fingerprint = mgr.compute_dataset_fingerprint()

        from training.segmentation_2d.dataset import BraTS2DSliceDataset
        train_ds = BraTS2DSliceDataset(
            splits["train"], train=True,
            orientation=args.orientation,
            patch_size=(args.patch_size, args.patch_size),
            tumor_ratio=args.tumor_ratio,
            max_slices_per_patient=args.max_slices,
            seed=args.seed,
        )
        val_ds = BraTS2DSliceDataset(
            splits["val"], train=False,
            orientation=args.orientation,
            patch_size=(args.patch_size, args.patch_size),
            tumor_ratio=args.tumor_ratio,
            max_slices_per_patient=args.max_slices,
            seed=args.seed,
        )
        patient_dirs_per_split = {k: [str(p) for p in v] for k, v in splits.items()}

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[Train] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    strides = [2] * (len(args.channels) - 1)
    model = create_2d_unet(
        in_channels=4,
        out_channels=3,
        channels=args.channels,
        strides=strides,
        num_res_units=args.num_res_units,
        dropout=args.dropout,
    ).to(device)
    print_model_summary(model)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn = get_loss(args.loss)
    early_stopping = EarlyStopping(patience=args.early_stopping, mode="max")

    # ── Reproducibility manifest ───────────────────────────────────────────────
    from training.common.reproducibility import build_manifest, save_manifest, print_manifest_summary
    manifest = build_manifest(
        pipeline="2d_segmentation",
        hyperparameters=vars(args),
        dataset_fingerprint=dataset_fingerprint,
        dataset_root=args.data_dir,
        split_counts={k: len(v) for k, v in patient_dirs_per_split.items()},
        split_patients=patient_dirs_per_split,
        model=model,
    )
    print_manifest_summary(manifest)
    save_manifest(manifest, str(out_dir / "manifest.json"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_dice = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch, last_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        best_dice = last_metrics.get("mean_dice", 0.0)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    print(f"\n[Train] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, args.grad_clip
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device,
            compute_hd95_flag=args.hd95,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": round(lr_now, 8),
            **{k: round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(row)

        mean_dice = val_metrics.get("mean_dice", 0.0)
        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"loss={train_loss:.4f} | {_format_val(val_metrics)} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(ckpt_dir / "best.pt"), scheduler,
                extra={"manifest_run_id": manifest.run_id},
            )
            print(f"  ✓ New best model saved (mean_dice={best_dice:.4f})")

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            str(ckpt_dir / "last.pt"), scheduler,
        )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(ckpt_dir / f"epoch_{epoch:04d}.pt"), scheduler,
            )

        # Early stopping
        if args.early_stopping > 0 and early_stopping(mean_dice):
            print(f"[Train] Early stopping at epoch {epoch}")
            break

    # ── Save results ─────────────────────────────────────────────────────────
    save_history_csv(history, str(out_dir / "training_log.csv"))
    plot_training_curves(history, str(out_dir / "plots"), prefix="2d")

    # Update manifest with final results
    best_row = max(history, key=lambda r: r.get("mean_dice", 0.0))
    manifest.best_epoch = best_row["epoch"]
    manifest.best_val_metrics = {
        k: v for k, v in best_row.items() if k not in ("epoch", "lr", "train_loss")
    }
    save_manifest(manifest, str(out_dir / "manifest.json"))

    final_report = {
        "disclaimer": "Research use only — not clinically validated.",
        "best_epoch": best_row["epoch"],
        "best_mean_dice": best_row.get("mean_dice", 0.0),
        "best_val_metrics": manifest.best_val_metrics,
        "total_epochs": len(history),
        "model_params": manifest.trainable_parameters,
        "run_id": manifest.run_id,
        "git_hash": manifest.git_hash,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final_report, indent=2))

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best epoch : {best_row['epoch']} | Mean Dice : {best_row.get('mean_dice', 0.0):.4f}")
    print(f"  Results    : {out_dir}")
    print(f"{'='*60}\n")


def _format_val(metrics: Dict) -> str:
    wt = metrics.get("wt_dice", float("nan"))
    tc = metrics.get("tc_dice", float("nan"))
    et = metrics.get("et_dice", float("nan"))
    hd = metrics.get("mean_hd95")
    s = f"WT={wt:.3f} TC={tc:.3f} ET={et:.3f}"
    if hd is not None:
        s += f" HD95={hd:.1f}mm"
    return s


if __name__ == "__main__":
    main()
