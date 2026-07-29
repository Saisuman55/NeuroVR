"""Data loading and preprocessing utilities for classification and segmentation."""
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries.

    Args:
        seed: Integer seed value.
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassificationDataset(Dataset):
    """PyTorch Dataset for brain tumor classification.

    Expects a directory structure where each class has its own subfolder
    containing image files (JPG/PNG).
    """

    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        img_size: int = 224,
        transform: Optional[A.Compose] = None,
        phase: str = "train",
    ):
        """Initialize the classification dataset.

        Args:
            root_dir: Root directory containing class subfolders.
            class_names: Ordered list of class names.
            img_size: Target image size (square).
            transform: Albumentations composition to apply.
            phase: Dataset phase identifier (train/val/test).
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.img_size = img_size
        self.transform = transform
        self.phase = phase
        self.samples: List[Tuple[str, int]] = []
        self._build_samples()

    def _build_samples(self) -> None:
        """Populate the samples list by scanning class directories."""
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retrieve a single image-label pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of transformed image tensor and integer label.
        """
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class SegmentationDataset(Dataset):
    """PyTorch Dataset for brain tumor segmentation.

    Recursively scans a directory for images and pairs them with masks
    identified by a configurable suffix (e.g., image_mask.png).
    """

    def __init__(
        self,
        image_dir: str,
        mask_suffix: str = "_mask",
        img_size: int = 128,
        transform: Optional[A.Compose] = None,
    ):
        """Initialize the segmentation dataset.

        Args:
            image_dir: Root directory containing images and masks.
            mask_suffix: Suffix identifying mask files.
            img_size: Target image size (square).
            transform: Albumentations composition to apply.
        """
        self.image_dir = image_dir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.transform = transform
        self.pairs: List[Tuple[str, str]] = []
        self._build_pairs()

    def _build_pairs(self) -> None:
        """Populate image-mask pairs by scanning the directory tree."""
        for root, _, files in os.walk(self.image_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                if self.mask_suffix in fname:
                    continue
                base, ext = os.path.splitext(fname)
                mask_name = f"{base}{self.mask_suffix}{ext}"
                mask_path = os.path.join(root, mask_name)
                if os.path.exists(mask_path):
                    self.pairs.append((os.path.join(root, fname), mask_path))

    def __len__(self) -> int:
        """Return the number of image-mask pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single image-mask pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of transformed image tensor and binary mask tensor.
        """
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_classification_transforms(
    img_size: int, augmentation: dict
) -> Tuple[A.Compose, A.Compose]:
    """Create Albumentations transforms for classification.

    Args:
        img_size: Target square image size.
        augmentation: Augmentation parameters from config.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    extra = []
    if augmentation.get("elastic_transform", 0) > 0:
        extra.append(A.ElasticTransform(
            alpha=1, sigma=50, p=0.5
        ))
    if augmentation.get("grid_distortion", 0) > 0:
        extra.append(A.GridDistortion(distort_limit=augmentation["grid_distortion"], p=0.5))
    if augmentation.get("optical_distortion", 0) > 0:
        extra.append(A.OpticalDistortion(
            distort_limit=augmentation["optical_distortion"], p=0.5
        ))
    if augmentation.get("gaussian_noise", 0) > 0:
        extra.append(A.GaussNoise(std_range=(0.04, 0.20), p=0.5))
    if augmentation.get("cutout", 0) > 0:
        extra.append(A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(0.0, 0.1), hole_width_range=(0.0, 0.1),
            p=0.5
        ))

    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5)
            if augmentation.get("random_flip") == "horizontal"
            else A.NoOp(),
            A.Rotate(
                limit=int(augmentation["random_rotation"] * 180), p=0.5
            ),
            A.RandomScale(
                scale_limit=augmentation["random_zoom"], p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=augmentation["random_contrast"], p=0.5
            ),
        ]
        + extra
        + [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def get_segmentation_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    """Create Albumentations transforms for segmentation.

    Args:
        img_size: Target square image size.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=0.1, p=0.5
            ),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def get_classification_loaders(
    config: dict, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/validation/test data loaders for classification.

    Args:
        config: Loaded configuration dictionary.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    set_seed(config["seed"])

    data_dir = config["paths"]["data_classification"]
    class_names = config["classification"]["class_names"]
    img_size = config["classification"]["img_size"]
    batch_size = config["classification"]["batch_size"]
    aug = config["classification"]["augmentation"]
    val_split = config["classification"].get("val_split", 0.1)

    train_transform, val_transform = get_classification_transforms(
        img_size, aug
    )

    full_train = ClassificationDataset(
        os.path.join(data_dir, "Training"),
        class_names,
        img_size,
        train_transform,
        "train",
    )
    test_dataset = ClassificationDataset(
        os.path.join(data_dir, "Testing"),
        class_names,
        img_size,
        val_transform,
        "test",
    )

    indices = list(range(len(full_train)))
    labels = [label for _, label in full_train.samples]
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=config["seed"],
        stratify=labels,
    )

    train_dataset = Subset(full_train, train_indices)
    val_dataset = ClassificationDataset(
        os.path.join(data_dir, "Training"),
        class_names,
        img_size,
        val_transform,
        "val",
    )
    val_dataset.samples = [full_train.samples[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


def get_segmentation_loaders(
    config: dict, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation data loaders for segmentation.

    Args:
        config: Loaded configuration dictionary.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    set_seed(config["seed"])

    data_dir = config["paths"]["data_segmentation"]
    img_size = config["segmentation"]["img_size"]
    batch_size = config["segmentation"]["batch_size"]
    mask_suffix = config["segmentation"].get("mask_suffix", "_mask")
    val_split = config["segmentation"].get("val_split", 0.2)

    train_transform, val_transform = get_segmentation_transforms(img_size)
    full_dataset = SegmentationDataset(
        data_dir, mask_suffix=mask_suffix, img_size=img_size, transform=train_transform
    )

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=config["seed"]
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = SegmentationDataset(
        data_dir,
        mask_suffix=mask_suffix,
        img_size=img_size,
        transform=val_transform,
    )
    val_dataset.pairs = [full_dataset.pairs[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
