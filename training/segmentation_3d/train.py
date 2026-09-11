"""
3D Brain Tumor Segmentation Training Script for NeuroVR.

Trains a 3D SegResNet or U-Net on full BraTS volumes with patch-based
sampling, MONAI augmentations, sliding-window validation, and HD95 metrics.

Usage::

    # Train from scratch
    python -m training.segmentation_3d.train --data_dir data/raw/brats

    # Fine-tune from MONAI pretrained bundle
    python -m training.segmentation_3d.train --data_dir data/raw/brats \\
        --pretrained models/monai/brats_mri_segmentation/models/model.pt

    # Dry run with synthetic data
    python -m training.segmentation_3d.train --dry_run

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroVR 3D Segmentation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--data_dir", type=str, default="data/raw/brats")
    p.add_argument("--patch_size", type=str, default="64,64,64",
                   help="Patch size as D,H,W (e.g. 64,64,64 or 96,96,96)")
    p.add_argument("--pos_samples", type=int, default=1)
    p.add_argument("--neg_samples", type=int, default=1)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--loss", default="dice_ce",
                   choices=["dice", "dice_ce", "dice_focal"])
    p.add_argument("--early_stopping", type=int, default=15)

    # Dataset split
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--architecture", default="segresnet",
                   choices=["segresnet", "unet3d"])
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to pretrained weights (MONAI bundle or checkpoint)")

    # Inference
    p.add_argument("--sw_roi_size", type=str, default="64,64,64",
                   help="Sliding-window ROI size for validation")
    p.add_argument("--sw_overlap", type=float, default=0.25)

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="results/3d")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_every", type=int, default=5)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])

    # Misc
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--hd95", action="store_true", default=True)
    p.add_argument("--no_hd95", action="store_false", dest="hd95")
    p.add_argument("--cache_dataset", action="store_true",
                   help="Use MONAI CacheDataset (faster but more RAM)")

    return p.parse_args()


def _parse_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(","))


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data for dry-run
# ──────────────────────────────────────────────────────────────────────────────

class _SyntheticDataset3D(torch.utils.data.Dataset):
    """Tiny synthetic 3D dataset for smoke testing."""

    def __init__(self, n: int = 10, patch: int = 32) -> None:
        self.n = n
        self.patch = patch
        rng = np.random.default_rng(42)
        self.images = rng.standard_normal((n, 4, patch, patch, patch)).astype(np.float32)
        masks = np.zeros((n, 3, patch, patch, patch), dtype=np.float32)
        for i in range(n):
            c = patch // 2
            r = patch // 5
            zz, yy, xx = np.mgrid[:patch, :patch, :patch]
            d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)
            masks[i, 0] = (d < r * 2.5).astype(np.float32)
            masks[i, 1] = (d < r * 1.5).astype(np.float32)
            masks[i, 2] = (d < r).astype(np.float32)
        self.masks = masks

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {"image": torch.from_numpy(self.images[idx]),
                "label": torch.from_numpy(self.masks[idx])}


# ──────────────────────────────────────────────────────────────────────────────
# Training & validation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss)
        n += 1
    return total_loss / max(n, 1)


def validate_sliding_window(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    roi_size: Tuple[int, ...],
    overlap: float,
    compute_hd95_flag: bool,
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    from monai.inferers import sliding_window_inference
    from training.segmentation_3d.metrics import MetricsTracker3D, compute_metrics_3d

    model.eval()
    tracker = MetricsTracker3D()
    total_loss = 0.0
    n = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["label"]

            output = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
            )
            loss = loss_fn(output, masks.to(device))
            total_loss += float(loss)
            n += 1

            probs = torch.sigmoid(output).cpu()
            for i in range(probs.shape[0]):
                m = compute_metrics_3d(
                    probs[i], masks[i],
                    spacing_mm=spacing_mm,
                    compute_hd95_flag=compute_hd95_flag,
                )
                tracker.update(m)

    metrics = tracker.compute()
    metrics["val_loss"] = total_loss / max(n, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    patch_size = _parse_tuple(args.patch_size)
    roi_size = _parse_tuple(args.sw_roi_size)

    from training.common.trainer_utils import (
        set_seed, get_device, save_checkpoint, load_checkpoint,
        EarlyStopping, plot_training_curves, save_history_csv,
    )
    from training.segmentation_3d.model import (
        create_segresnet, create_3d_unet,
        load_pretrained_weights, print_model_summary,
    )
    from training.segmentation_3d.losses import get_loss

    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NeuroVR 3D Segmentation Training")
    print(f"  DISCLAIMER: Research prototype — NOT for clinical use")
    print(f"{'='*60}\n")
    print(f"  Architecture : {args.architecture}")
    print(f"  Patch size   : {patch_size}")
    print(f"  Device       : {device}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[Train3D] DRY RUN — using synthetic data")
        train_ds = _SyntheticDataset3D(n=12, patch=patch_size[0])
        val_ds = _SyntheticDataset3D(n=4, patch=patch_size[0])
        patient_dirs_per_split = {"train": [], "val": [], "test": []}
        dataset_fingerprint = "dry_run_synthetic"
    else:
        from data.dataset_manager import BraTSDatasetManager

        mgr = BraTSDatasetManager(dataset_root=args.data_dir)
        if not mgr.check_availability():
            print(
                "\n[Train3D] ERROR: Dataset not found.\n"
                "  See: data/README_DATASETS.md\n"
                f"  Expected: {args.data_dir}\n"
            )
            sys.exit(1)

        mgr.validate_structure()
        splits = mgr.create_splits(
            train=args.train_frac, val=args.val_frac, test=args.test_frac,
            seed=args.seed, save_path=str(out_dir / "splits.json"),
        )
        dataset_fingerprint = mgr.compute_dataset_fingerprint()

        from training.segmentation_3d.dataset import BraTS3DVolumeDataset
        train_ds = BraTS3DVolumeDataset(
            splits["train"], patch_size=patch_size,
            train=True,
            pos_samples=args.pos_samples,
            neg_samples=args.neg_samples,
            cache=args.cache_dataset,
            num_workers=args.num_workers,
        )
        val_ds = BraTS3DVolumeDataset(
            splits["val"], patch_size=patch_size,
            train=False,
            cache=False,
        )
        patient_dirs_per_split = {k: [str(p) for p in v] for k, v in splits.items()}

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.architecture == "segresnet":
        model = create_segresnet(in_channels=4, out_channels=3).to(device)
    else:
        model = create_3d_unet(in_channels=4, out_channels=3).to(device)
    print_model_summary(model, prefix=f"3D {args.architecture.upper()}")

    if args.pretrained:
        load_pretrained_weights(model, args.pretrained, strict=False, device=device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs // 4 or 10, T_mult=2, eta_min=args.lr * 0.01
    )
    loss_fn = get_loss(args.loss)
    early_stopping = EarlyStopping(patience=args.early_stopping, mode="max")

    # ── Reproducibility manifest ───────────────────────────────────────────────
    from training.common.reproducibility import build_manifest, save_manifest, print_manifest_summary
    manifest = build_manifest(
        pipeline="3d_segmentation",
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
    print(f"\n[Train3D] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        val_metrics = validate_sliding_window(
            model, val_loader, loss_fn, device,
            roi_size=roi_size,
            overlap=args.sw_overlap,
            compute_hd95_flag=args.hd95,
        )
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        mean_dice = val_metrics.get("mean_dice", 0.0)
        wt = val_metrics.get("wt_dice", 0.0)
        tc = val_metrics.get("tc_dice", 0.0)
        et = val_metrics.get("et_dice", 0.0)
        hd = val_metrics.get("mean_hd95")

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": round(lr_now, 8),
            **{k: round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(row)

        hd_str = f" HD95={hd:.1f}mm" if hd is not None else ""
        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"loss={train_loss:.4f} | "
            f"WT={wt:.3f} TC={tc:.3f} ET={et:.3f}{hd_str} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(ckpt_dir / "best.pt"), scheduler,
                extra={"manifest_run_id": manifest.run_id},
            )
            print(f"  ✓ Best model saved (mean_dice={best_dice:.4f})")

        save_checkpoint(model, optimizer, epoch, val_metrics,
                        str(ckpt_dir / "last.pt"), scheduler)

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            str(ckpt_dir / f"epoch_{epoch:04d}.pt"), scheduler)

        if args.early_stopping > 0 and early_stopping(mean_dice):
            print(f"[Train3D] Early stopping at epoch {epoch}")
            break

    # ── Save results ─────────────────────────────────────────────────────────
    save_history_csv(history, str(out_dir / "training_log.csv"))
    plot_training_curves(history, str(out_dir / "plots"), prefix="3d")

    best_row = max(history, key=lambda r: r.get("mean_dice", 0.0))
    manifest.best_epoch = best_row["epoch"]
    manifest.best_val_metrics = {
        k: v for k, v in best_row.items()
        if k not in ("epoch", "lr", "train_loss")
    }
    save_manifest(manifest, str(out_dir / "manifest.json"))

    report = {
        "disclaimer": "Research use only — not clinically validated.",
        "best_epoch": best_row["epoch"],
        "best_mean_dice": best_row.get("mean_dice", 0.0),
        "best_wt_dice": best_row.get("wt_dice", 0.0),
        "best_tc_dice": best_row.get("tc_dice", 0.0),
        "best_et_dice": best_row.get("et_dice", 0.0),
        "best_mean_hd95": best_row.get("mean_hd95"),
        "run_id": manifest.run_id,
        "git_hash": manifest.git_hash,
        "dataset_fingerprint": dataset_fingerprint,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best epoch  : {best_row['epoch']} | Mean Dice : {best_row.get('mean_dice', 0.0):.4f}")
    print(f"  Results     : {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
