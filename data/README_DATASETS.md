# Brain Tumor MRI Datasets Guide

> **DISCLAIMER**: NeuroVR is an educational/research prototype and is **NOT** intended for clinical diagnosis.
> All results are for academic and research purposes only.

---

## Supported Datasets

### 1. BraTS 2020 / BraTS 2021 (Recommended)

The Brain Tumor Segmentation (BraTS) challenge provides the most widely-used
benchmark for brain tumor segmentation.

**Labels:**
| Value | Region | Abbreviation |
|-------|--------|--------------|
| 0 | Background | BG |
| 1 | Necrotic and Non-Enhancing Tumor Core | NCR/NET |
| 2 | Peritumoral Edema | ED |
| 4 | Enhancing Tumor | ET |

**Derived regions (computed automatically):**
| Region | Labels | Note |
|--------|--------|------|
| Whole Tumor (WT) | {1, 2, 4} | All non-background |
| Tumor Core (TC) | {1, 4} | Core without edema |
| Enhancing Tumor (ET) | {4} | Gadolinium-enhancing |

**Biological containment invariant (enforced automatically):**
```
ET ⊆ TC ⊆ WT
```

### 2. Medical Segmentation Decathlon (MSD) Task 01

Same label scheme as BraTS 2020. Compatible out-of-the-box.

---

## Download Instructions

### Option A: BraTS 2021 via Kaggle

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Download (requires Kaggle account + API key)
kaggle datasets download -d dschettler8845/brats-2021-task1

# 3. Extract
unzip brats-2021-task1.zip -d data/raw/brats

# 4. Set environment variable
export BRATS_DATASET_PATH="$(pwd)/data/raw/brats"
```

### Option B: BraTS 2020 via Kaggle

```bash
kaggle datasets download -d awsaf49/brats2020-training-data
unzip brats2020-training-data.zip -d data/raw/brats
export BRATS_DATASET_PATH="$(pwd)/data/raw/brats"
```

### Option C: Medical Decathlon (MSD) Task01

```bash
# Download from Google Drive (official link):
# http://medicaldecathlon.com/
# Task01_BrainTumour.tar (~6GB)

tar -xf Task01_BrainTumour.tar -C data/raw/
export BRATS_DATASET_PATH="$(pwd)/data/raw/Task01_BrainTumour/imagesTr"
```

### Option D: Use local datasets

```bash
# If you already have BraTS data on disk:
export BRATS_DATASET_PATH="/path/to/your/BraTS/training"

# Or set in config.yaml:
# dataset:
#   brats_path: /path/to/your/BraTS/training
```

---

## Expected Directory Structure

```
data/raw/brats/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_001_flair.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
└── ...
```

**Required files per patient:**
- `*t1*.nii.gz` — T1 (without contrast)
- `*t1ce*.nii.gz` — T1ce (with contrast)
- `*t2*.nii.gz` — T2 FLAIR alternative
- `*flair*.nii.gz` — FLAIR
- `*seg*.nii.gz` — Segmentation mask

---

## Dataset Validation

Run the dataset manager to validate your dataset:

```bash
# Validate structure + print statistics
python -m data.dataset_manager --data_dir data/raw/brats --validate --stats

# Create and save patient splits
python -m data.dataset_manager --data_dir data/raw/brats --split
```

---

## Dataset Statistics (BraTS 2020, 369 patients)

| Metric | Value |
|--------|-------|
| Total patients | 369 |
| Volume shape | ~240 × 240 × 155 |
| Voxel spacing | 1.0 × 1.0 × 1.0 mm |
| Modalities | T1, T1ce, T2, FLAIR |
| WT volume (mean) | ~70,000 voxels |
| TC volume (mean) | ~22,000 voxels |
| ET volume (mean) | ~7,000 voxels |
| Cases without ET | ~15% |

---

## License & Citation

BraTS datasets are provided for research purposes only.
By using BraTS data, you agree to the BraTS challenge terms:
- https://www.med.upenn.edu/cbica/brats/

**Required citation (BraTS 2021):**
```
Baid et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on 
Brain Tumor Segmentation and Radiogenomic Classification." 
arXiv:2107.02314, 2021.
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Dataset not found` | Set `BRATS_DATASET_PATH` env var |
| `Missing modality: t1ce` | Check file naming (must contain `t1ce`, not just `t1`) |
| `Affine mismatch` | Run affine consistency check; data may not be co-registered |
| `Corrupted file` | Re-download the specific patient |
| MPS deadlock on macOS | Set `DISABLE_MPS=1` before training |

---

*NeuroVR — B.Tech CSE Major Project | Research Prototype Only*
