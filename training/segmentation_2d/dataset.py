"""
2D Slice Dataset for NeuroVR Segmentation Training.

Wraps a list of patient directories (from a single split) and yields
(image, mask) pairs as 2D slices extracted on-the-fly from 3D volumes.

IMPORTANT: This dataset must receive patient directories from ONLY ONE
split (train/val/test). Patient-level splitting must be done first via
BraTSDatasetManager.create_splits() before instantiating this class.

Input : 4-channel MRI slice [4, H, W] — (T1, T1ce, T2, FLAIR)
Output: 3-channel binary mask [3, H, W] — (WT, TC, ET)

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS2DSliceDataset(Dataset):
    """PyTorch Dataset yielding 2D slices from BraTS 3D volumes.

    Slices are extracted lazily from 3D volumes during __getitem__.
    A patient-indexed slice manifest is built during __init__ so that
    standard DataLoader indexing works correctly.

    Example::

        mgr = BraTSDatasetManager(dataset_root="data/raw/brats")
        splits = mgr.create_splits()
        train_ds = BraTS2DSliceDataset(splits["train"], train=True)
        val_ds   = BraTS2DSliceDataset(splits["val"],   train=False)
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        train: bool = True,
        orientation: str = "axial",
        patch_size: Tuple[int, int] = (240, 240),
        tumor_ratio: float = 0.6,
        max_slices_per_patient: Optional[int] = None,
        seed: int = 42,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            patient_dirs: List of patient directory Paths (from one split ONLY).
            train: If True, apply augmentations.
            orientation: "axial", "coronal", or "sagittal".
            patch_size: Output 2D slice size (H, W). Volumes resized to this.
            tumor_ratio: Fraction of slices per patient that must contain tumor.
            max_slices_per_patient: Cap on slices per patient (None = all).
            seed: Random seed for reproducible slice sampling.
            normalize: Apply Z-score normalization to image channels.
        """
        self.patient_dirs = list(patient_dirs)
        self.train = train
        self.orientation = orientation
        self.patch_size = patch_size
        self.tumor_ratio = tumor_ratio
        self.max_slices_per_patient = max_slices_per_patient
        self.seed = seed
        self.normalize = normalize

        # Build slice manifest: list of (patient_dir, slice_index)
        self._manifest: List[Tuple[Path, int]] = []
        self._build_manifest()

        # Augmentation transforms (MONAI 2D)
        self._augment = None
        if self.train:
            try:
                from preprocessing.training_preprocessor import get_2d_train_augmentations
                self._augment = get_2d_train_augmentations()
            except Exception:
                pass  # Graceful fallback — no augmentation

    def _build_manifest(self) -> None:
        """Build the (patient_dir, slice_index) manifest for all patients."""
        from preprocessing.slice_extractor import (
            load_brats_volume_for_2d,
            get_tumor_slice_indices,
            sample_slice_indices,
            _get_axis_length,
        )

        rng = random.Random(self.seed)
        total_slices = 0

        for patient_dir in self.patient_dirs:
            try:
                # Load only the segmentation to find tumor slices (fast)
                import nibabel as nib
                seg_files = [
                    f for f in patient_dir.glob("*.nii*")
                    if "seg" in f.name.lower()
                ]
                if not seg_files:
                    continue

                from data.label_utils import brats_seg_to_regions
                seg_int = nib.load(str(seg_files[0])).get_fdata().astype(np.int32)
                wt, tc, et = brats_seg_to_regions(seg_int, enforce=True)
                seg_3ch = np.stack([wt, tc, et], axis=0)

                n_slices = _get_axis_length(seg_3ch.shape[1:], self.orientation)
                tumor_indices = self._get_tumor_indices(seg_3ch, n_slices)

                selected = sample_slice_indices(
                    n_slices,
                    tumor_indices,
                    self.tumor_ratio,
                    self.max_slices_per_patient,
                    rng,
                )

                for idx in selected:
                    self._manifest.append((patient_dir, idx))

                total_slices += len(selected)

            except Exception as exc:
                print(f"[Dataset2D] Warning: skipping {patient_dir.name}: {exc}")
                continue

        print(
            f"[Dataset2D] Built manifest: {len(self.patient_dirs)} patients, "
            f"{total_slices} slices ({self.orientation})"
        )

    def _get_tumor_indices(self, seg_3ch: np.ndarray, n_slices: int) -> List[int]:
        """Get slice indices containing tumor."""
        from preprocessing.slice_extractor import get_tumor_slice_indices
        return get_tumor_slice_indices(seg_3ch, self.orientation)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one (image_slice, mask_slice) pair.

        Args:
            idx: Index into the manifest.

        Returns:
            Tuple of:
              - image: torch.Tensor [4, H, W] float32
              - mask:  torch.Tensor [3, H, W] float32
        """
        patient_dir, slice_idx = self._manifest[idx]

        from preprocessing.slice_extractor import (
            load_brats_volume_for_2d,
            _get_slice,
            _resize_slice,
        )

        # Load full 3D volume for this patient
        image_4ch, seg_3ch, _ = load_brats_volume_for_2d(
            patient_dir, normalize=self.normalize
        )

        # Extract the requested 2D slice
        img_slice = _get_slice(image_4ch, slice_idx, self.orientation)  # [4, H, W]
        seg_slice = _get_slice(seg_3ch, slice_idx, self.orientation)    # [3, H, W]

        # Resize to target patch size
        if img_slice.shape[1:] != self.patch_size:
            img_slice = _resize_slice(img_slice, self.patch_size, order=1)
            seg_slice = _resize_slice(seg_slice, self.patch_size, order=0)

        # Convert to tensors
        image_t = torch.from_numpy(img_slice).float()
        mask_t = torch.from_numpy(seg_slice).float()

        # Apply augmentations (training only)
        if self.train and self._augment is not None:
            try:
                image_t = self._augment(image_t)
            except Exception:
                pass  # Graceful fallback

        return image_t, mask_t

    @property
    def n_patients(self) -> int:
        """Number of unique patients in this dataset."""
        return len(self.patient_dirs)

    @property
    def n_slices(self) -> int:
        """Total number of slices in this dataset."""
        return len(self._manifest)
