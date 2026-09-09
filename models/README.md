# models/

This directory contains the pretrained AI models used by NeuroVR 3D.

## Active Model — MONAI BraTS MRI Segmentation

```
models/monai/brats_mri_segmentation/
├── configs/
│   ├── inference.json     ← inference configuration
│   └── metadata.json      ← model metadata
└── models/
    ├── model.pt           ← TorchScript weights (~36 MB)
    └── model.ts           ← TorchScript traced model
```

**Source:** [MONAI Model Zoo](https://monai.io/model-zoo.html)  
**Task:** BraTS 3D brain tumor segmentation  
**Input:** 4-channel MRI [T1, T1ce, T2, FLAIR], shape [4, H, W, D]  
**Output:** 3-channel segmentation [Tumor Core, Whole Tumor, Enhancing Tumor]  
**License:** Apache 2.0

## Downloading the Model

The model is bundled in this repository. If it is missing, run:

```bash
python3 scripts/download_bundle.py
```

This will download and extract the MONAI BraTS bundle into `models/monai/`.

## Legacy Models (archived)

The following model weights were used by the old 2D pipeline and are no longer
part of the production system. They have been moved to `archive/legacy_2d/models/`:

- `brain_tumor_classifier_best.pth` — EfficientNet-B4 4-class classifier (68 MB)
- `brain_tumor_segmenter_best.pth` — ResNet34 U-Net binary segmenter (93 MB)
