# 📦 Datasets & Training Resources Used
## Brain Tumor AI Detection & Segmentation System

---

## 🗂️ DATASET 1: Classification Dataset

### Name: Brain Tumor MRI Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Purpose:** Train the EfficientNet-B4 classifier to identify tumor type

### Folder Structure (Actual on Disk)
```
data/classification/
├── Training/
│   ├── glioma/       → 1,401 MRI images
│   ├── meningioma/   → 1,401 MRI images
│   ├── notumor/      → 1,401 MRI images
│   └── pituitary/    → 1,401 MRI images
│
└── Testing/
    ├── glioma/       → 401 MRI images
    ├── meningioma/   → 401 MRI images
    ├── notumor/      → 401 MRI images
    └── pituitary/    → 401 MRI images
```

### Actual Numbers (Verified from Disk)
| Split | Per Class | Total Images |
|---|---|---|
| Training | ~1,401 | **5,604 images** |
| Testing | ~401 | **1,604 images** |
| **Grand Total** | | **~7,208 MRI scans** |

### Classes Explained

| Class | Full Name | Description | Severity |
|---|---|---|---|
| `glioma` | Glioma | Cancer originating in glial cells (astrocytes, oligodendrocytes). Most aggressive brain tumor. | High |
| `meningioma` | Meningioma | Arises from the meninges (brain/spinal cord lining). Usually benign but can compress the brain. | Medium |
| `pituitary` | Pituitary Tumor | Forms in the pituitary gland. Affects hormone production. | Low–Medium |
| `notumor` | No Tumor | Healthy brain MRI scan — no abnormality present. | None |

### Image Properties
| Property | Value |
|---|---|
| Format | JPEG (.jpg) |
| Color | RGB (3 channels) |
| Resolution | Variable (resized to 380×380 during training) |
| Source | Real clinical MRI scans from hospital databases |

---

## 🗂️ DATASET 2: Segmentation Dataset

### Name: Brain MRI Segmentation (LGG — Lower Grade Glioma)
- **Source:** Kaggle — https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **Original Source:** The Cancer Genome Atlas (TCGA) Lower-Grade Glioma collection
- **Purpose:** Train the U-Net model to generate pixel-level tumor masks

### Folder Structure (Actual on Disk)
```
data/segmentation/
└── kaggle_3m/
    ├── TCGA_CS_4941_19960909/    ← Patient folder (MRI + mask pairs)
    ├── TCGA_CS_4942_19970222/
    ├── TCGA_CS_4943_20000902/
    ├── TCGA_CS_4944_20010208/
    ├── TCGA_CS_5393_19990606/
    ├── TCGA_CS_5395_19981004/
    ... (113 patient folders total)
    └── README.md
```

### Actual Numbers (Verified from Disk)
| Property | Value |
|---|---|
| Patient Folders | **113 TCGA patients** |
| MRI + Mask Pairs | ~3,929 image pairs |
| Naming Convention | `TCGA_XX_XXXX_XXXXXXXX` (Patient ID + Date) |

### What Each Patient Folder Contains
Each TCGA folder has matched pairs:
```
TCGA_CS_4941_19960909/
├── TCGA_CS_4941_19960909_1.tif      ← MRI slice (input)
├── TCGA_CS_4941_19960909_1_mask.tif ← Binary mask (ground truth)
├── TCGA_CS_4941_19960909_2.tif
├── TCGA_CS_4941_19960909_2_mask.tif
└── ... (multiple slices per patient)
```

### Mask Properties
| Property | Value |
|---|---|
| Format | TIFF (.tif) |
| Values | Binary — 0 (no tumor) or 255 (tumor pixel) |
| Resolution | Variable (resized to 128×128 during training) |
| Tumor Type | Lower Grade Glioma (LGG) |

---

## 🧠 PRE-TRAINED MODEL WEIGHTS

Both of our models were **not trained from scratch**. We used **Transfer Learning** — starting from models already trained on ImageNet (1.2 million general images). This is crucial for performance with limited medical data.

### 1. EfficientNet-B4 (ImageNet Pre-trained)
| Property | Value |
|---|---|
| Source | `timm` library (PyTorch Image Models) |
| Pre-trained on | ImageNet-1K (1,281,167 images, 1,000 classes) |
| Parameters | ~19 million |
| What we kept | All backbone weights (feature extractors) |
| What we replaced | Final classification head (1000 → 4 classes) |
| Why it helps | The model already knows edges, textures, shapes — we just teach it medical context |

### 2. ResNet-34 (U-Net Encoder, ImageNet Pre-trained)
| Property | Value |
|---|---|
| Source | `segmentation_models_pytorch` library |
| Pre-trained on | ImageNet-1K |
| Parameters | ~21 million (full U-Net) |
| Role | Encoder (feature extraction) for the U-Net |
| What we kept | All ResNet-34 encoder weights |
| What we trained | The U-Net decoder (upsampling path) |

---

## 🔧 TRAINING CONFIGURATION (What Was Actually Used)

### Classification Model Training

| Parameter | Value Used |
|---|---|
| Model | EfficientNet-B4 |
| Input Image Size | 380 × 380 pixels |
| Batch Size | 16 (M5 optimized) |
| Effective Batch Size | 32 (with gradient accumulation ×2) |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Phase A Learning Rate | 3e-5 |
| Phase B Learning Rate | 8e-6 |
| Loss Function | CrossEntropyLoss + Label Smoothing (0.05) |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Gradient Clipping | 1.0 |
| EMA Decay | 0.999 |
| Validation Split | 10% of training set |
| Early Stopping Patience | 10 epochs |
| Hardware | Apple M5 (PyTorch MPS backend) |
| Workers | 2 (efficiency cores) |
| Seed | 42 (reproducibility) |

### Segmentation Model Training

| Parameter | Value Used |
|---|---|
| Model | U-Net with ResNet-34 encoder |
| Input Image Size | 128 × 128 pixels |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 5e-5 |
| Loss Function | 0.5 × BCE Loss + 0.5 × Dice Loss |
| LR Scheduler | CosineAnnealingLR |
| Epochs | 70 |
| Early Stopping Patience | 10 |
| Validation Split | 20% |
| Mask Suffix | `_mask` |
| Activation | Sigmoid |

---

## 🔀 DATA AUGMENTATION PIPELINE

To prevent overfitting and improve generalization, we applied the following augmentations using the **Albumentations** library:

### Classification Augmentations
| Augmentation | Probability | Effect |
|---|---|---|
| `HorizontalFlip` | 0.5 | Mirrors image left-right |
| `Rotate` | 0.3 | Random rotation ±30° |
| `RandomResizedCrop` | 0.2 | Random zoom and crop |
| `RandomBrightnessContrast` | 0.2 | Vary brightness/contrast |
| `ElasticTransform` | 0.2 | Deforms image like tissue |
| `GridDistortion` | 0.2 | Grid-based warping |
| `OpticalDistortion` | 0.2 | Lens-like distortion |
| `GaussNoise` | 0.1 | Adds random pixel noise |
| `CoarseDropout` | 0.1 | Masks random image patches |
| `Normalize` | Always | ImageNet mean/std normalization |

### Why These Specific Augmentations for MRI?
- **ElasticTransform + GridDistortion** — Simulates natural tissue deformation in MRI
- **GaussNoise** — Simulates MRI scanner noise artifacts
- **CoarseDropout** — Forces model to use context, not rely on single region
- **No color jitter** — MRI scans are grayscale-like, color changes would be unrealistic

---

## 📊 TOTAL DATA SUMMARY

| Dataset | Task | Images | Source |
|---|---|---|---|
| Brain Tumor MRI (Kaggle) | Classification | 7,208 | Kaggle (masoudnickparvar) |
| LGG MRI Segmentation (TCGA) | Segmentation | ~3,929 pairs | Kaggle / TCGA |
| ImageNet (pre-trained) | Transfer Learning | 1.28 million | ImageNet-1K |
| **Total used** | | **~11,137 images** | |

---

## 📚 LIBRARIES & FRAMEWORKS

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.x | Core deep learning framework |
| `torchvision` | 0.x | Image utilities |
| `timm` | latest | EfficientNet-B4 pre-trained model |
| `segmentation_models_pytorch` | latest | U-Net with ResNet-34 encoder |
| `albumentations` | latest | Image augmentation pipeline |
| `opencv-python` | latest | Image loading, mask overlay |
| `numpy` | latest | Array operations |
| `scikit-learn` | latest | Stratified train/val split |
| `matplotlib` | latest | Training plots |
| `tqdm` | latest | Training progress bars |
| `flask` | latest | Web API backend |
| `pyyaml` | latest | Config file parsing |

---

## 🔑 KEY DESIGN DECISIONS

### Why Kaggle Datasets?
1. **Publicly available** — No patient privacy concerns for academic research
2. **Pre-labeled** — Labels verified by medical professionals
3. **Balanced** — Roughly equal samples per class
4. **Standard benchmark** — Used by many published research papers, allowing comparison

### Why Transfer Learning?
- Medical datasets are small (thousands vs millions needed for scratch training)
- Pre-trained ImageNet weights already detect low-level features (edges, corners, textures)
- Fine-tuning needs only a fraction of the compute vs training from scratch
- Achieves higher accuracy with less data

### Why Albumentations over torchvision transforms?
- Albumentations is **5-10× faster** on CPU
- Supports medical-specific augmentations (ElasticTransform, GridDistortion)
- Consistent augmentation applied to **both image and mask** simultaneously (critical for segmentation)

---

*Document auto-generated from actual project configuration and verified against disk contents*
*Brain Tumor Detection & Segmentation System — Department of CSE*
