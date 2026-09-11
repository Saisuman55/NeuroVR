#!/usr/bin/env python3
"""
NeuroVR Training Launcher — convenience script.

Validates the dataset, then trains both pipelines sequentially.

Usage:
    # Download dataset first (see data/README_DATASETS.md), then:
    python train_all.py --data_dir data/raw/brats

    # Dry run to validate pipeline (no real data needed)
    python train_all.py --dry_run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="NeuroVR — Train 2D and 3D pipelines")
    p.add_argument("--data_dir", type=str, default="data/raw/brats")
    p.add_argument("--epochs_2d", type=int, default=30)
    p.add_argument("--epochs_3d", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--only_2d", action="store_true")
    p.add_argument("--only_3d", action="store_true")
    args = p.parse_args()

    # macOS: disable MPS for background threads
    if sys.platform == "darwin":
        os.environ.setdefault("DISABLE_MPS", "1")

    base = Path(__file__).parent

    def run(cmd):
        print(f"\n{'='*60}")
        print(f"  Running: {' '.join(str(c) for c in cmd)}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd, cwd=str(base))
        if result.returncode != 0:
            print(f"\n[Launcher] ERROR: Command failed with code {result.returncode}")
            sys.exit(result.returncode)

    common_args = [
        f"--data_dir={args.data_dir}",
        f"--seed={args.seed}",
        f"--device={args.device}",
    ]
    if args.dry_run:
        common_args.append("--dry_run")

    # ── Validate dataset ──────────────────────────────────────────────────────
    if not args.dry_run:
        print("\n[Launcher] Validating dataset...")
        run([
            sys.executable, "-m", "data.dataset_manager",
            "--validate", "--stats",
            f"--data_dir={args.data_dir}",
        ])

    # ── 2D Training ───────────────────────────────────────────────────────────
    if not args.only_3d:
        run([
            sys.executable, "-m", "training.segmentation_2d.train",
            f"--epochs={args.epochs_2d}",
            "--output_dir=results/2d",
            *common_args,
        ])

    # ── 3D Training ───────────────────────────────────────────────────────────
    if not args.only_2d:
        run([
            sys.executable, "-m", "training.segmentation_3d.train",
            f"--epochs={args.epochs_3d}",
            "--output_dir=results/3d",
            "--architecture=segresnet",
            *common_args,
        ])

    print("\n[Launcher] All training pipelines complete!")
    print("  2D results: results/2d/")
    print("  3D results: results/3d/")
    print("  DISCLAIMER: Research prototype — NOT for clinical use.")


if __name__ == "__main__":
    main()
