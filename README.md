# 🧠 BrainTumor AI — Clinical MRI Analysis Suite

> End-to-end PyTorch pipeline for brain MRI classification and pixel-level segmentation, with a modern web dashboard for clinical use.

---

## ✨ Features

- **Multi-class Tumor Classification** — EfficientNet-B4 classifies MRI scans into 4 categories: Glioma, Meningioma, Pituitary, No Tumor
- **Pixel-level Segmentation** — U-Net with ResNet34 encoder produces a precise binary tumor mask
- **Cross-Check Logic** — Segmentation model validates classifier output to reduce false negatives
- **Clinical Web Dashboard** — Upload an MRI scan and view 5 real-time result panels: Original, Binary Mask, Green Overlay, Contour Boundary, and Probability Heatmap
- **PDF Report Generation** — One-click clinical PDF export with findings and visualizations
- **Live Analysis Stats** — Tumor coverage %, severity index, and pixel count computed directly from the model output

---

## 🏗️ Architecture

```
MRI Input
    │
    ▼
EfficientNet-B4  ──── Classification ──── Tumor Class (glioma / meningioma / pituitary / notumor)
                                               │
                              ┌────────────────┘
                              ▼
                     U-Net + ResNet34  ──── Binary Mask ──── Cross-Check
                                               │
                              ┌────────────────┘
                              ▼
              5 Output Images: Original │ Mask │ Overlay │ Contour │ Heatmap
                              │
                              ▼
                     Flask API → Web Dashboard → PDF Report
```

---

## 📁 Project Structure

```
brain_tumor_project/
├── src/
│   ├── model.py              # EfficientNet-B4 classifier + U-Net segmenter definitions
│   ├── data_loader.py        # Dataset loaders, augmentations, config parser
│   ├── train_m5_optimized.py # Full training script (M-series/CUDA optimized)
│   ├── inference.py          # Unified inference: classify → segment → 5 outputs
│   └── evaluate.py           # Evaluation metrics, confusion matrix, IoU/Dice
│
├── stitch_frontend/          # Web dashboard (HTML/CSS/JS)
│   ├── index.html            # Home — pipeline overview & 3D brain animation
│   ├── inference.html        # MRI upload, result gallery, PDF export
│   ├── training_status.html  # Training metrics charts
│   ├── about.html            # Model details & dataset info
│   └── metrics.json          # Live metrics updated during training
│
├── models/
│   ├── classifier/           # Place brain_tumor_classifier_best.pth here
│   └── segmenter/            # Place brain_tumor_segmenter_best.pth here
│
├── data/
│   ├── classification/       # Kaggle MRI classification dataset
│   └── segmentation/         # LGG MRI segmentation dataset
│
├── outputs/
│   ├── predictions/          # Inference output images (5 per run)
│   ├── plots/                # Training curves, confusion matrices
│   └── uploads/              # Temporary upload directory (auto-cleaned)
│
├── app.py                    # Flask API server (port 8080)
├── report_generator.py       # ReportLab PDF clinical report generator
├── monitor.py                # Training monitor (updates metrics.json live)
├── config.yaml               # All hyperparameters & paths
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-ai.git
cd brain-tumor-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

| Task | Dataset | Link | Place at |
|---|---|---|---|
| Classification | Brain Tumor MRI Dataset (Masoud Nickparvar) | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | `data/classification/` |
| Segmentation | LGG MRI Segmentation (Mateusz Buda) | [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | `data/segmentation/` |

Expected structure:
```
data/classification/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/ ...

data/segmentation/
└── kaggle_3m/
    └── <patient_folders>/  (image + _mask pairs)
```

### 4. Download pre-trained weights *(optional — skip if training from scratch)*

> Model weights are not stored in this repo due to file size (~170 MB).  
> Either train from scratch (see below) or download from the release page.

Place downloaded weights in:
```
models/classifier/brain_tumor_classifier_best.pth
models/segmenter/brain_tumor_segmenter_best.pth
```

---

## 🚀 Running the Web Dashboard

```bash
python app.py
```

Open **http://localhost:8080** in your browser.

---

## 🏋️ Training

Train both models end-to-end (M-series Mac / CUDA optimized):

```bash
python src/train_m5_optimized.py
```

Monitor training live (updates `stitch_frontend/metrics.json` for the dashboard):

```bash
python monitor.py
```

Training outputs:
- `models/classifier/brain_tumor_classifier_best.pth`
- `models/segmenter/brain_tumor_segmenter_best.pth`
- `outputs/plots/` — training curves and confusion matrices

---

## 🔬 Inference (CLI)

Run on a single MRI image:

```bash
python src/inference.py --image test_samples/glioma_Te-gl_1.jpg
```

Produces 5 output images in `outputs/predictions/`:

| File | Description |
|---|---|
| `original.png` | Input MRI as loaded |
| `binary_mask.png` | U-Net predicted tumor mask |
| `green_overlay.png` | Tumor region highlighted in green on original |
| `contour.png` | Cyan tumor boundary drawn on original |
| `heatmap.png` | JET-colormap probability heatmap from raw U-Net output |
| `inference_result.png` | Combined 3-panel figure (legacy) |

---

## 📊 Evaluation

```bash
python src/evaluate.py --task both
```

Or for a specific task:
```bash
python src/evaluate.py --task classification
python src/evaluate.py --task segmentation --num_visualize 10
```

---

## 📈 Results

| Model | Task | Metric | Performance |
|---|---|---|---|
| EfficientNet-B4 | 4-class Classification | Accuracy | >98% |
| U-Net + ResNet34 | Binary Segmentation | Dice Coefficient | >0.80 |

---

## 🖼️ Test Samples

12 ready-to-use test images are included in `test_samples/` — 3 per class:

```
test_samples/
├── glioma_Te-gl_1.jpg
├── meningioma_Te-aug-me_1.jpg
├── notumor_Te-no_1.jpg
└── pituitary_Te-pi_1.jpg
... (12 total)
```

---

## 🧪 Reproducibility

- Fixed seeds via `set_seed()` in `src/data_loader.py`
- `torch.backends.cudnn.deterministic = True`
- All hyperparameters in `config.yaml`

---

## 🔧 Configuration

All training/inference parameters in [`config.yaml`](config.yaml):

```yaml
classification:
  model_name: efficientnet_b4
  img_size: 380
  num_classes: 4
  batch_size: 8
  learning_rate: 0.0001

segmentation:
  encoder: resnet34
  img_size: 128
  classes: 1
  activation: sigmoid
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Classification | EfficientNet-B4 (timm via segmentation-models-pytorch) |
| Segmentation | U-Net + ResNet34 (segmentation-models-pytorch) |
| Augmentation | Albumentations |
| Web Server | Flask |
| Frontend | HTML / Tailwind CSS / Vanilla JS |
| PDF Reports | ReportLab |
| Training Acceleration | Apple MPS / CUDA |

---

## 📄 License

This project is for educational and research purposes.  
Datasets are sourced from Kaggle under their respective licenses.
