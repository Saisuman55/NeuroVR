# NeuroVR 3D
### AI-Powered 3D Brain MRI Segmentation · Anatomical Localization · AR/VR Visualization

> **Research prototype — NOT for clinical use.**  
> All measurements and anatomical estimates are for research and demonstration purposes only.

---

## Overview

NeuroVR 3D is a full-stack medical imaging application that:

1. **Accepts** multi-modal NIfTI MRI (T1, T1ce, T2, FLAIR) or DICOM series
2. **Segments** brain tumors in 3D using the pretrained MONAI BraTS model
3. **Reconstructs** volumetric tumor meshes (GLB) via Marching Cubes
4. **Measures** tumor volumes and spatial extents in physical units (cm³ / mm)
5. **Localizes** tumors anatomically (LEFT/RIGHT hemisphere, estimated lobe)
6. **Visualizes** everything in an interactive Three.js 3D viewer
7. **Supports** WebXR AR and VR immersive viewing

---

## Pipeline

```
NIfTI / DICOM input
        ↓
  Volume preprocessing
  (resample → 1 mm³, z-score normalize, crop foreground)
        ↓
  MONAI BraTS 3D segmentation
  (sliding-window inference → TC / WT / ET masks)
        ↓
  Mask post-processing
  (connected-component filter, morphological closing)
        ↓
  3D mesh reconstruction
  (Marching Cubes → Laplacian smoothing → GLB export)
        ↓
  Physical measurements
  (volume cm³, bounding box mm, centroid RAS mm)
        ↓
  Anatomical localization
  (NIfTI affine → RAS centroid → LEFT/RIGHT/MIDLINE + estimated lobe)
        ↓
  Three.js 3D viewer + WebXR AR/VR
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the MONAI model is present

```bash
ls models/monai/brats_mri_segmentation/models/model.pt
# If missing:
python3 scripts/download_bundle.py
```

### 3. Start the server

```bash
DISABLE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 flask_app_3d.py
```

Open **[http://localhost:7861](http://localhost:7861)**

### 4. Demo mode

Click **⚗ Demo Mode** in the UI to run the full pipeline on the bundled real MRI scan (MNI152 standard brain).

---

## Project Structure

```
brain_tumor_project/
│
├── flask_app_3d.py          ← Main application (port 7861)
├── config.yaml              ← Pipeline configuration
├── requirements.txt         ← Python dependencies
├── .env.example             ← Environment variable template
│
├── preprocessing/           ← NIfTI/DICOM loading + volume preprocessing
│   ├── nifti_loader.py
│   ├── dicom_loader.py
│   └── volume_preprocessor.py
│
├── inference/               ← MONAI 3D segmentation + anatomical localization
│   ├── model_loader_3d.py
│   ├── segmentation_3d.py
│   ├── segmentation_numpy.py   ← CPU fallback (no MONAI required for demo)
│   └── localization.py
│
├── reconstruction/          ← Mesh generation + measurements
│   ├── mask_processing.py
│   ├── tumor_mesh.py
│   └── measurements.py
│
├── frontend/                ← Three.js medical imaging workstation UI
│   ├── index.html
│   ├── style.css
│   ├── viewer.js
│   └── ui.js
│
├── scripts/
│   └── download_bundle.py   ← Download MONAI BraTS model bundle
│
├── models/
│   ├── README.md
│   └── monai/brats_mri_segmentation/   ← Pretrained model
│
├── data/
│   └── real_brats/          ← Real MNI152 MRI (demo data, ~3.4 MB)
│
├── demo_data/
│   └── generate_demo.py     ← Synthetic BraTS case generator
│
├── tests/
│   ├── smoke_test.py        ← End-to-end pipeline test (no MONAI required)
│   └── test_pipeline.py
│
├── outputs/                 ← Runtime session data (generated, not committed)
│   └── .gitkeep
│
└── archive/
    └── legacy_2d/           ← Old 2D EfficientNet/U-Net pipeline (reference only)
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload NIfTI files (multipart/form-data) |
| `POST` | `/api/analyze/<sid>` | Start 3D AI segmentation pipeline |
| `GET` | `/api/status/<sid>` | Poll pipeline progress |
| `GET` | `/api/results/<sid>` | Tumor measurements JSON |
| `GET` | `/api/mesh/<sid>/<type>` | Download GLB mesh (brain/tumor_whole/tumor_core/tumor_enhancing) |
| `GET` | `/api/localization/<sid>` | Anatomical localization result |
| `GET` | `/api/slice/<sid>/<plane>/<idx>` | 2D MRI slice PNG (axial/coronal/sagittal) |
| `GET` | `/api/slice_info/<sid>` | Slice dimension info |
| `GET` | `/api/demo` | Run demo with real MNI152 MRI |
| `GET` | `/api/health` | System status |

---

## Model

**MONAI BraTS MRI Segmentation**  
Source: [MONAI Model Zoo](https://monai.io/model-zoo.html)  
Architecture: SegResNet  
Input: 4-channel NIfTI [T1, T1ce, T2, FLAIR]  
Output: 3-class segmentation [Tumor Core, Whole Tumor, Enhancing Tumor]  
License: Apache 2.0

---

## Anatomical Localization

Tumor location is derived from:
1. **NIfTI affine matrix** → converts voxel centroid to RAS world coordinates (mm)
2. **RAS X-coordinate** → LEFT (X < 0) / RIGHT (X > 0) / MIDLINE (|X| < 8 mm)
3. **MNI152 bounding boxes** → estimated lobe (Frontal/Parietal/Temporal/Occipital/Cerebellum/Brainstem)

> **Disclaimer:** Localization is coordinate-estimated — not atlas-registered.  
> It is NOT a clinical diagnosis. Consult a qualified radiologist.

---

## Supported Input Formats

| Format | Modalities |
|---|---|
| `.nii`, `.nii.gz` | T1, T1ce, T2, FLAIR (one file per modality) |
| `.dcm` (DICOM) | Any MRI series |

Expected modality naming: files containing `t1`, `t1ce`/`t1c`, `t2`, `flair` in their filename are auto-detected.

---

## AR / VR

WebXR is supported via Three.js `XRButton`:
- **AR** — Places the 3D tumor model in your physical space (requires ARCore/ARKit device)
- **VR** — Full immersive viewing (requires VR headset or compatible browser)

Both require HTTPS in production. Localhost works without HTTPS in Chrome.

---

## Tests

```bash
# End-to-end pipeline smoke test (no MONAI model required)
python3 tests/smoke_test.py

# Full pipeline test (requires MONAI + model)
python3 tests/test_pipeline.py
```

---

## Privacy

- Uploaded MRI files are stored temporarily in `outputs/<session_id>/uploads/`
- Session data is cleared on server restart
- No patient data is committed to this repository
- Never upload real patient data to a development server

---

## License

Research prototype. See individual component licenses:
- MONAI model: Apache 2.0
- Three.js: MIT
- Application code: MIT
