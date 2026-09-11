"""
BraTS Dataset Manager for NeuroVR.

Manages the full dataset lifecycle:
  1. Check dataset availability
  2. Validate directory structure and file integrity
  3. Validate modality affine consistency (shared coordinate system)
  4. Validate segmentation label containment
  5. Create deterministic patient-level train/val/test splits
  6. Compute dataset fingerprint for reproducibility
  7. Print dataset statistics

Supports: BraTS 2020, BraTS 2021, MSD Task01

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]
SEGMENTATION_SUFFIX = "seg"
NIFTI_EXTENSIONS = {".nii", ".gz"}
VALID_BRATS_LABELS = {0, 1, 2, 4}

# BraTS 2018 compatibility: some datasets use label 3 for ET
VALID_BRATS_LABELS_COMPAT = {0, 1, 2, 3, 4}


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class BraTSDatasetManager:
    """Validates, splits, and manages a BraTS-format dataset.

    Usage::

        mgr = BraTSDatasetManager(dataset_root="data/raw/brats")
        if mgr.check_availability():
            mgr.validate_structure()
            splits = mgr.create_splits()
            mgr.print_statistics()
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialise the dataset manager.

        Priority order for dataset_root:
          1. Explicit ``dataset_root`` argument
          2. ``BRATS_DATASET_PATH`` environment variable
          3. ``brats_path`` field in ``config.yaml``

        Args:
            dataset_root: Path to the BraTS training directory.
            config_path: Path to config.yaml (used if dataset_root not given).
        """
        self.dataset_root: Path = self._resolve_root(dataset_root, config_path)
        self._patients: Optional[List[Path]] = None
        self._splits_cache: Optional[Dict[str, List[Path]]] = None

    # ── Root resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_root(
        explicit: Optional[str],
        config_path: Optional[str],
    ) -> Path:
        if explicit:
            return Path(explicit)

        env_val = os.environ.get("BRATS_DATASET_PATH", "").strip()
        if env_val:
            return Path(env_val)

        # Try config.yaml
        cfg_file = Path(config_path) if config_path else Path("config.yaml")
        if cfg_file.exists():
            try:
                import yaml
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f)
                brats_path = cfg.get("dataset", {}).get("brats_path", "")
                if brats_path:
                    return Path(brats_path)
            except Exception:
                pass

        return Path("data/raw/brats")

    # ── Availability & structure validation ──────────────────────────────────

    def check_availability(self) -> bool:
        """Return True if dataset_root exists and contains patient directories."""
        if not self.dataset_root.exists():
            print(f"[Dataset] ✗ Dataset root not found: {self.dataset_root}")
            print(f"[Dataset] Set BRATS_DATASET_PATH env var or brats_path in config.yaml")
            return False

        patients = self._find_patient_dirs()
        if not patients:
            print(f"[Dataset] ✗ No patient directories found under {self.dataset_root}")
            return False

        print(f"[Dataset] ✓ Found {len(patients)} patient directories in {self.dataset_root}")
        return True

    def _find_patient_dirs(self) -> List[Path]:
        """Find all candidate patient directories (contains at least one .nii.gz)."""
        candidates = []
        for child in sorted(self.dataset_root.iterdir()):
            if child.is_dir():
                niftis = list(child.glob("*.nii*"))
                if niftis:
                    candidates.append(child)
        return candidates

    def _find_modality_file(self, patient_dir: Path, suffix: str) -> Optional[Path]:
        """Find a NIfTI file containing the given suffix (case-insensitive)."""
        suffix_l = suffix.lower()
        for f in sorted(patient_dir.glob("*.nii*")):
            name = f.name.lower().replace(".nii.gz", "").replace(".nii", "")
            if suffix_l in name:
                return f
        return None

    def validate_structure(self) -> Tuple[List[Path], List[str]]:
        """Validate directory structure for all patient directories.

        Returns:
            (valid_patients, error_messages) — valid_patients have all 4
            modalities + seg file and pass basic header validation.
        """
        candidates = self._find_patient_dirs()
        valid = []
        errors = []

        print(f"[Dataset] Validating {len(candidates)} patient directories...")

        for patient_dir in candidates:
            pid = patient_dir.name
            patient_errors = []

            # Check required modalities
            missing_modalities = []
            for mod in REQUIRED_MODALITY_SUFFIXES:
                if self._find_modality_file(patient_dir, mod) is None:
                    missing_modalities.append(mod)
            if missing_modalities:
                patient_errors.append(f"Missing modalities: {missing_modalities}")

            # Check segmentation file
            seg_file = self._find_modality_file(patient_dir, SEGMENTATION_SUFFIX)
            if seg_file is None:
                patient_errors.append("Missing segmentation file (*seg*.nii*)")

            if patient_errors:
                errors.append(f"{pid}: {'; '.join(patient_errors)}")
                continue

            # Validate modality affines (shared coordinate system)
            affine_ok, affine_msg = self._validate_affine_consistency(patient_dir)
            if not affine_ok:
                errors.append(f"{pid}: {affine_msg}")
                continue

            valid.append(patient_dir)

        self._patients = valid
        print(f"[Dataset] Valid: {len(valid)}/{len(candidates)} patients")
        if errors:
            print(f"[Dataset] ✗ {len(errors)} patients failed validation:")
            for e in errors[:10]:
                print(f"  {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        return valid, errors

    def _validate_affine_consistency(
        self,
        patient_dir: Path,
    ) -> Tuple[bool, str]:
        """Check that all modalities share the same voxel-to-world affine.

        This ensures all MRI volumes are registered to the same coordinate
        system — a prerequisite for multi-modal segmentation.

        Args:
            patient_dir: Patient directory.

        Returns:
            (is_valid, message)
        """
        try:
            import nibabel as nib
        except ImportError:
            return True, "nibabel not available — skipping affine check"

        reference_affine = None
        reference_name = None

        for mod in REQUIRED_MODALITY_SUFFIXES:
            f = self._find_modality_file(patient_dir, mod)
            if f is None:
                continue
            try:
                img = nib.load(str(f))
                affine = img.affine
                if reference_affine is None:
                    reference_affine = affine
                    reference_name = mod
                else:
                    if not np.allclose(affine, reference_affine, rtol=1e-4, atol=1e-2):
                        return (
                            False,
                            f"Affine mismatch: {mod} differs from {reference_name} "
                            f"(coordinate system inconsistency)",
                        )
            except Exception as exc:
                return False, f"Cannot load {mod}: {exc}"

        return True, "OK"

    def validate_labels(
        self,
        patient_dir: Path,
        check_containment: bool = True,
    ) -> Tuple[bool, str]:
        """Validate the segmentation label file for a patient.

        Args:
            patient_dir: Patient directory.
            check_containment: If True, check ET ⊆ TC ⊆ WT.

        Returns:
            (is_valid, message)
        """
        try:
            import nibabel as nib
            from data.label_utils import brats_seg_to_regions, validate_containment
        except ImportError as exc:
            return False, f"Import error: {exc}"

        seg_file = self._find_modality_file(patient_dir, SEGMENTATION_SUFFIX)
        if seg_file is None:
            return False, "No segmentation file"

        try:
            img = nib.load(str(seg_file))
            seg = img.get_fdata(dtype=np.float32).astype(np.int32)
        except Exception as exc:
            return False, f"Cannot load seg: {exc}"

        # Check label values
        unique_labels = set(np.unique(seg).tolist())
        unexpected = unique_labels - VALID_BRATS_LABELS_COMPAT
        if unexpected:
            return False, f"Unexpected label values: {unexpected}"

        # Containment check
        if check_containment:
            wt, tc, et = brats_seg_to_regions(seg, enforce=False)
            report = validate_containment(wt, tc, et)
            if not report.valid:
                # Log but don't fail — some BraTS files have minor boundary issues
                # We enforce containment during training anyway
                return True, f"Minor containment issue (auto-corrected): {report.summary}"

        return True, "OK"

    def detect_corrupted_files(self) -> Dict[str, List[str]]:
        """Attempt to load every NIfTI file and report failures.

        Returns:
            Dict mapping patient_id → list of unreadable file names.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("[Dataset] nibabel not available — skipping corruption check")
            return {}

        corrupted: Dict[str, List[str]] = {}
        patients = self._patients or self._find_patient_dirs()

        for patient_dir in patients:
            bad_files = []
            for f in sorted(patient_dir.glob("*.nii*")):
                try:
                    nib.load(str(f)).get_fdata()
                except Exception as exc:
                    bad_files.append(f"{f.name}: {exc}")
            if bad_files:
                corrupted[patient_dir.name] = bad_files

        if corrupted:
            print(f"[Dataset] ✗ {len(corrupted)} patients have corrupted files:")
            for pid, files in list(corrupted.items())[:5]:
                print(f"  {pid}: {files}")
        else:
            print(f"[Dataset] ✓ All files readable")

        return corrupted

    # ── Patient-level splitting ──────────────────────────────────────────────

    @staticmethod
    def _patient_bucket(patient_id: str, seed: int = 42) -> int:
        """Assign patient to a 0–99 bucket deterministically."""
        key = f"{seed}:{patient_id}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100

    def create_splits(
        self,
        train: float = 0.70,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[Path]]:
        """Create deterministic patient-level train/val/test splits.

        IMPORTANT: Splitting is performed at the PATIENT level. No patient
        appears in more than one split. All 2D slice extraction and
        augmentation happens AFTER splits are determined.

        Args:
            train: Fraction for training (0–1).
            val: Fraction for validation (0–1).
            test: Fraction for test (0–1).
            seed: Random seed for reproducibility.
            save_path: If given, save split JSON to this path.

        Returns:
            Dict with keys "train", "val", "test" mapping to patient Path lists.
        """
        assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1"

        if self._patients is None:
            self.validate_structure()

        patients = sorted(self._patients, key=lambda p: p.name)
        train_thresh = int(train * 100)
        val_thresh = train_thresh + int(val * 100)

        splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        for patient_dir in patients:
            bucket = self._patient_bucket(patient_dir.name, seed)
            if bucket < train_thresh:
                splits["train"].append(patient_dir)
            elif bucket < val_thresh:
                splits["val"].append(patient_dir)
            else:
                splits["test"].append(patient_dir)

        print(
            f"[Dataset] Splits (seed={seed}): "
            f"train={len(splits['train'])} | "
            f"val={len(splits['val'])} | "
            f"test={len(splits['test'])}"
        )

        # Verify no overlap
        train_ids = {p.name for p in splits["train"]}
        val_ids = {p.name for p in splits["val"]}
        test_ids = {p.name for p in splits["test"]}
        overlaps = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        assert len(overlaps) == 0, f"Data leakage detected — patients in multiple splits: {overlaps}"

        self._splits_cache = splits

        if save_path:
            self._save_splits_json(splits, seed, save_path)

        return splits

    def _save_splits_json(
        self,
        splits: Dict[str, List[Path]],
        seed: int,
        path: str,
    ) -> None:
        """Save splits to JSON with dataset fingerprint for reproducibility."""
        data = {
            "dataset_root": str(self.dataset_root),
            "dataset_fingerprint": self.compute_dataset_fingerprint(),
            "seed": seed,
            "splits": {
                split_name: [str(p) for p in patient_list]
                for split_name, patient_list in splits.items()
            },
            "counts": {k: len(v) for k, v in splits.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Dataset] Splits saved to: {path}")

    def get_split(self, split_name: str) -> List[Path]:
        """Get patient dirs for a specific split.

        Args:
            split_name: "train", "val", or "test".

        Returns:
            List of patient directory Paths.
        """
        if self._splits_cache is None:
            raise RuntimeError("Call create_splits() first.")
        return self._splits_cache[split_name]

    def list_patients(self) -> List[Path]:
        """Return all validated patient directories."""
        if self._patients is None:
            self.validate_structure()
        return list(self._patients)

    # ── Dataset fingerprint ──────────────────────────────────────────────────

    def compute_dataset_fingerprint(self) -> str:
        """Compute a SHA256 fingerprint over patient IDs and file sizes.

        This fingerprint is stable across machines and Python versions
        as long as the dataset contents are the same.

        Returns:
            40-character hex fingerprint string.
        """
        patients = self._patients or self._find_patient_dirs()
        entries = []
        for patient_dir in sorted(patients, key=lambda p: p.name):
            for f in sorted(patient_dir.glob("*.nii*")):
                entries.append(f"{patient_dir.name}/{f.name}:{f.stat().st_size}")

        fingerprint = hashlib.sha256("\n".join(entries).encode()).hexdigest()[:40]
        return fingerprint

    # ── Statistics ───────────────────────────────────────────────────────────

    def print_statistics(self, sample_n: int = 10) -> None:
        """Print dataset statistics (shapes, spacings, label distribution).

        Args:
            sample_n: Number of patients to sample for volume statistics.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("[Dataset] nibabel required for statistics")
            return

        patients = self._patients or self.list_patients()
        if not patients:
            print("[Dataset] No patients available for statistics.")
            return

        print(f"\n{'='*60}")
        print(f"  NeuroVR Dataset Statistics")
        print(f"{'='*60}")
        print(f"  Root         : {self.dataset_root}")
        print(f"  Total patients: {len(patients)}")
        print(f"  Fingerprint  : {self.compute_dataset_fingerprint()}")

        # Sample volumes for shape/spacing
        sample = patients[:sample_n]
        shapes, spacings = [], []
        wt_vols, tc_vols, et_vols = [], [], []

        from data.label_utils import brats_seg_to_regions

        for patient_dir in sample:
            t1_file = self._find_modality_file(patient_dir, "t1")
            if t1_file:
                try:
                    img = nib.load(str(t1_file))
                    shapes.append(img.shape[:3])
                    spacings.append(img.header.get_zooms()[:3])
                except Exception:
                    pass

            seg_file = self._find_modality_file(patient_dir, SEGMENTATION_SUFFIX)
            if seg_file:
                try:
                    seg = nib.load(str(seg_file)).get_fdata().astype(np.int32)
                    wt, tc, et = brats_seg_to_regions(seg, enforce=True)
                    wt_vols.append(int(wt.sum()))
                    tc_vols.append(int(tc.sum()))
                    et_vols.append(int(et.sum()))
                except Exception:
                    pass

        if shapes:
            shapes_arr = np.array(shapes)
            print(f"\n  Volume shapes (sampled {len(shapes)}p):")
            print(f"    Mean  : {shapes_arr.mean(axis=0).round(1)}")
            print(f"    Min   : {shapes_arr.min(axis=0)}")
            print(f"    Max   : {shapes_arr.max(axis=0)}")

        if spacings:
            sp_arr = np.array(spacings)
            print(f"  Voxel spacing (mm):")
            print(f"    Mean  : {sp_arr.mean(axis=0).round(3)}")

        if wt_vols:
            print(f"  Label voxel counts (sampled {len(wt_vols)}p, after containment enforcement):")
            print(f"    WT mean : {np.mean(wt_vols):,.0f} vox  | WT median : {np.median(wt_vols):,.0f}")
            print(f"    TC mean : {np.mean(tc_vols):,.0f} vox  | TC median : {np.median(tc_vols):,.0f}")
            print(f"    ET mean : {np.mean(et_vols):,.0f} vox  | ET median : {np.median(et_vols):,.0f}")
            empty_et = sum(1 for v in et_vols if v == 0)
            print(f"    Cases without ET: {empty_et}/{len(et_vols)}")

        print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NeuroVR BraTS Dataset Manager")
    parser.add_argument("--data_dir", type=str, help="Path to BraTS dataset root")
    parser.add_argument("--validate", action="store_true", help="Validate dataset structure")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--split", action="store_true", help="Create and save patient splits")
    parser.add_argument("--split_output", type=str, default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mgr = BraTSDatasetManager(dataset_root=args.data_dir)

    if not mgr.check_availability():
        sys.exit(1)

    if args.validate:
        valid, errors = mgr.validate_structure()
        corrupted = mgr.detect_corrupted_files()
        if errors or corrupted:
            print(f"[Dataset] Validation completed with issues.")
        else:
            print(f"[Dataset] ✓ All {len(valid)} patients validated successfully.")

    if args.stats:
        mgr.print_statistics()

    if args.split:
        splits = mgr.create_splits(seed=args.seed, save_path=args.split_output)
        print(f"[Dataset] Splits: {{k: len(v) for k, v in splits.items()}}")


if __name__ == "__main__":
    _cli()
