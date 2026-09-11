"""
3D MONAI Dataset for BraTS Brain Tumor Segmentation Training.

Wraps BraTS patient directories into a MONAI CacheDataset-compatible
format using dictionary transforms. Patient-level splitting must be
performed BEFORE instantiating this dataset.

Input : 4-channel NIfTI volumes [T1, T1ce, T2, FLAIR]
Output: 3-channel binary masks [WT, TC, ET] (containment-enforced)

Research prototype — NOT for clinical use.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _build_data_list(patient_dirs: List[Path]) -> List[dict]:
    """Convert patient directory list into MONAI data dict list.

    Each dict has:
      "image": [t1_path, t1ce_path, t2_path, flair_path]
      "label": seg_path
      "patient_id": str

    Args:
        patient_dirs: List of BraTS patient directories.

    Returns:
        List of dicts for MONAI transforms.
    """
    from preprocessing.nifti_loader import _detect_modality

    modality_order = ["t1", "t1ce", "t2", "flair"]
    data_list = []

    for patient_dir in patient_dirs:
        modality_paths = {m: None for m in modality_order}
        seg_path = None

        for f in sorted(patient_dir.glob("*.nii*")):
            name_l = f.name.lower()
            if "seg" in name_l:
                seg_path = f
                continue
            mod = _detect_modality(f.name)
            if mod and modality_paths.get(mod) is None:
                modality_paths[mod] = f

        missing = [m for m in modality_order if modality_paths[m] is None]
        if missing or seg_path is None:
            print(f"[Dataset3D] Skipping {patient_dir.name}: missing {missing or 'seg'}")
            continue

        data_list.append({
            "image": [str(modality_paths[m]) for m in modality_order],
            "label": str(seg_path),
            "patient_id": patient_dir.name,
        })

    return data_list


class BraTS3DVolumeDataset:
    """MONAI-based 3D dataset for BraTS brain tumor segmentation.

    Wraps MONAI's Dataset or CacheDataset with full training/validation
    transform pipelines including BraTS label conversion with containment
    enforcement.

    Example::

        mgr = BraTSDatasetManager("data/raw/brats")
        splits = mgr.create_splits()
        train_ds = BraTS3DVolumeDataset(splits["train"], patch_size=(64,64,64), train=True)
        val_ds   = BraTS3DVolumeDataset(splits["val"], patch_size=(64,64,64), train=False)
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        train: bool = True,
        pos_samples: int = 1,
        neg_samples: int = 1,
        cache: bool = False,
        num_workers: int = 0,
    ) -> None:
        """
        Args:
            patient_dirs: Patient directories from ONE split only.
            patch_size: 3D patch size for training crops.
            target_spacing: Resampling target spacing (mm).
            train: If True, apply augmentations and random cropping.
            pos_samples: Number of positive (tumor) patches per volume.
            neg_samples: Number of negative patches per volume.
            cache: If True, use MONAI CacheDataset (faster but uses more RAM).
            num_workers: Workers for CacheDataset loading.
        """
        try:
            from monai.data import Dataset, CacheDataset
            from preprocessing.training_preprocessor import (
                get_3d_train_transforms,
                get_3d_val_transforms,
            )
        except ImportError as exc:
            raise ImportError(f"MONAI required: pip install monai — {exc}") from exc

        self.patient_dirs = list(patient_dirs)
        self.train = train

        # Build data list
        data_list = _build_data_list(patient_dirs)
        if not data_list:
            raise RuntimeError(
                f"No valid patients found in {len(patient_dirs)} directories."
            )

        # Build transforms
        if train:
            transforms = get_3d_train_transforms(
                patch_size=patch_size,
                target_spacing=target_spacing,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
            )
        else:
            transforms = get_3d_val_transforms(target_spacing=target_spacing)

        # Wrap in MONAI Dataset
        if cache:
            self._dataset = CacheDataset(
                data=data_list,
                transform=transforms,
                num_workers=num_workers,
                progress=True,
            )
        else:
            self._dataset = Dataset(data=data_list, transform=transforms)

        print(
            f"[Dataset3D] {'Train' if train else 'Val'}: "
            f"{len(data_list)} patients | patch={patch_size}"
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]

    @property
    def n_patients(self) -> int:
        return len(self.patient_dirs)
