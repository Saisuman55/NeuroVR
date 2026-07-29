# 🧠 BrainTumor AI — Clinical MRI Analysis Suite

> End-to-end PyTorch pipeline for brain MRI **classification** and pixel-level **segmentation**, served through a modern clinical web dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-class Classification** | EfficientNet-B4 classifies MRI into 4 types: Glioma, Meningioma, Pituitary, No Tumor |
| **Pixel-level Segmentation** | U-Net + ResNet34 produces a binary tumor mask |
| **Cross-Check Logic** | Segmentation result cross-validates the classifier to reduce false negatives |
| **5-Panel Result Gallery** | Original · Binary Mask · Green Overlay · Contour · Probability Heatmap |
| **PDF Clinical Report** | One-click export with findings, confidence scores, and visualizations |
| **Live Training Monitor** | `monitor.py` tails the training log and updates `metrics.json` every 30 s |
| **Live Analysis Stats** | Tumor coverage %, severity index, and pixel count — computed from the mask |

---

## 🏗️ System Architecture

```
MRI Image Upload
      │
      ▼
 EfficientNet-B4 ──► Tumor Class + Confidence + Probabilities
      │
      │ (if tumor detected)
      ▼
 U-Net + ResNet34 ──► Binary Mask ──► Cross-Check Validation
                           │
                           ▼
           ┌─────────────────────────────────┐
           │  5 Output Images saved to disk  │
           │  original.png                   │
           │  binary_mask.png                │
           │  green_overlay.png              │
           │  contour.png                    │
           │  heatmap.png                    │
           └─────────────┬───────────────────┘
                         │
                         ▼
            Flask API (/api/predict)
                         │
                         ▼
             Web Dashboard (port 8080)
                         │
                         ▼
             PDF Report (/api/generate_report)
```

---

## 📁 Project Structure

```
brain_tumor_project/
│
├── src/
│   ├── model.py              # EfficientNet-B4 classifier + U-Net segmenter
│   ├── data_loader.py        # Dataset classes, augmentations, seed utils
│   ├── train_m5_optimized.py # Two-phase training (frozen → fine-tune)
│   ├── inference.py          # Classify → segment → save 5 output images
│   └── evaluate.py           # Confusion matrix, Dice, IoU, CSV reports
│
├── stitch_frontend/          # Web dashboard (pure HTML/CSS/JS)
│   ├── index.html            # Home — pipeline overview + 3D brain animation
│   ├── inference.html        # Upload MRI, view results, export PDF
│   ├── training_status.html  # Live training charts (reads metrics.json)
│   ├── about.html            # Model details, datasets, architecture
│   └── metrics.json          # Written by monitor.py during training
│
├── models/
│   ├── classifier/           # ← place brain_tumor_classifier_best.pth here
│   └── segmenter/            # ← place brain_tumor_segmenter_best.pth here
│
├── data/
│   ├── classification/       # Brain Tumor MRI Dataset (Kaggle)
│   └── segmentation/         # LGG MRI Segmentation Dataset (Kaggle)
│
├── outputs/
│   ├── predictions/          # 5 inference images (generated at runtime)
│   ├── plots/                # Training curves, confusion matrices
│   ├── uploads/              # Temp directory for web uploads (auto-cleaned)
│   └── training_log.txt      # Written by train_m5_optimized.py (read by monitor.py)
│
├── test_samples/             # 12 ready-to-use MRI images (3 per class)
│
├── app.py                    # Flask server — routes + API endpoints
├── report_generator.py       # ReportLab PDF clinical report builder
├── monitor.py                # Parses training_log.txt → updates metrics.json
├── config.yaml               # All hyperparameters and file paths
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### Requirements

- Python **3.9 – 3.12**
- macOS (Apple MPS) **or** Linux/Windows with CUDA GPU
- ~4 GB RAM minimum; 8+ GB recommended for training

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-ai.git
cd brain-tumor-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

| Task | Dataset | Author | Kaggle Link | Place files at |
|---|---|---|---|---|
| Classification | Brain Tumor MRI Dataset | Masoud Nickparvar | [Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | `data/classification/` |
| Segmentation | LGG MRI Segmentation | Mateusz Buda et al. | [Link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | `data/segmentation/` |

**Expected folder layout after extraction:**

```
data/classification/
├── Training/
│   ├── glioma/        (~1300 images)
│   ├── meningioma/    (~1300 images)
│   ├── notumor/       (~1600 images)
│   └── pituitary/     (~1200 images)
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

data/segmentation/kaggle_3m/
└── <patient_TCGA_folder>/
    ├── TCGA_xxx_1.tif        (MRI slice)
    └── TCGA_xxx_1_mask.tif   (binary tumor mask)
```

### 4. Pre-trained weights *(skip if training from scratch)*

> Model weights (~170 MB) are excluded from this repo via `.gitignore`.  
> Download from the GitHub Releases page and place them at:

```
models/classifier/brain_tumor_classifier_best.pth
models/segmenter/brain_tumor_segmenter_best.pth
```

---

## 🚀 Running the Web Dashboard

```bash
python app.py
```

Open **[http://localhost:8080](http://localhost:8080)** in your browser.

The app serves the frontend and exposes three API endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serves `stitch_frontend/index.html` |
| `GET /api/metrics` | GET | Returns current `metrics.json` for training charts |
| `POST /api/predict` | POST | Accepts `multipart/form-data` image → runs inference → returns 5 base64 images + class |
| `POST /api/generate_report` | POST | Accepts JSON with class/confidence/images → returns PDF binary |

---

## 🏋️ Training

Training runs in two phases automatically:
1. **Phase 1 — Frozen backbone**: Only the classification head is trained
2. **Phase 2 — Fine-tuning**: Top `N` EfficientNet layers are unfrozen and trained at a lower learning rate

### Start training

Open **two terminals** side by side:

**Terminal 1** — start training:
```bash
python src/train_m5_optimized.py 2>&1 | tee outputs/training_log.txt
```

**Terminal 2** — monitor live (updates the dashboard):
```bash
python monitor.py
```

`monitor.py` reads `outputs/training_log.txt` every 30 s and updates `stitch_frontend/metrics.json`, which the Training Status page reads to render live charts.

### Training outputs

| File | Description |
|---|---|
| `models/classifier/brain_tumor_classifier_best.pth` | Best classifier checkpoint |
| `models/segmenter/brain_tumor_segmenter_best.pth` | Best segmenter checkpoint |
| `outputs/plots/classification_history.png` | Train/val accuracy + loss curves |
| `outputs/plots/segmentation_history.png` | Dice + loss curves |
| `outputs/training_log.txt` | Full training log (read by `monitor.py`) |

---

## 🔬 Inference (CLI)

Run on a single MRI image:

```bash
python src/inference.py --image test_samples/glioma_Te-gl_1.jpg
```

Saves 5 images to `outputs/predictions/`:

| File | Description |
|---|---|
| `original.png` | Input MRI as loaded |
| `binary_mask.png` | U-Net binary tumor mask |
| `green_overlay.png` | Tumor highlighted in green on original |
| `contour.png` | Cyan tumor boundary on original |
| `heatmap.png` | JET-colormap probability heatmap |

The inference script also prints to stdout (parsed by `app.py`):
```
Predicted class: GLIOMA (confidence: 0.9472)
Probabilities: [0.9472, 0.0312, 0.0189, 0.0027]
Classes: ['glioma', 'meningioma', 'pituitary', 'notumor']
```

---

## 📊 Evaluation

```bash
# Evaluate both models
python src/evaluate.py --task both

# Classifier only — confusion matrix + classification report
python src/evaluate.py --task classification

# Segmenter only — Dice + IoU per sample
python src/evaluate.py --task segmentation --num_visualize 10
```

Outputs saved to `outputs/plots/` and `outputs/predictions/`.

---

## 📈 Results

| Model | Task | Primary Metric | Value |
|---|---|---|---|
| EfficientNet-B4 | 4-class classification | Accuracy | >98% |
| U-Net + ResNet34 | Binary segmentation | Dice Coefficient | >0.80 |

Loss functions:
- **Classifier**: Cross-Entropy with label smoothing (0.1)
- **Segmenter**: BCE + Dice combined loss (50/50 weight)

---

## 🖼️ Test Samples

12 real MRI images are included in `test_samples/` — 3 per class, ready to upload directly in the web dashboard or use with the CLI:

```
test_samples/
├── glioma_Te-gl_1.jpg         glioma_Te-gl_10.jpg        glioma_Te-gl_100.jpg
├── meningioma_Te-aug-me_1.jpg meningioma_Te-aug-me_10.jpg meningioma_Te-aug-me_100.jpg
├── notumor_Te-no_1.jpg        notumor_Te-no_10.jpg        notumor_Te-no_100.jpg
└── pituitary_Te-pi_1.jpg      pituitary_Te-pi_10.jpg      pituitary_Te-pi_100.jpg
```

---

## 🔧 Configuration (`config.yaml`)

All training and inference parameters are centralized:

```yaml
seed: 42

paths:
  data_classification: data/classification
  data_segmentation:   data/segmentation/kaggle_3m
  model_classifier:    models/classifier/brain_tumor_classifier.pth
  model_segmenter:     models/segmenter/brain_tumor_segmenter.pth
  outputs:             outputs

classification:
  model_name:    efficientnet_b4
  img_size:      380
  num_classes:   4
  batch_size:    8
  epochs:        50
  learning_rate: 0.0001
  dropout:       0.2
  class_names:   [glioma, meningioma, notumor, pituitary]
  early_stopping_patience: 15

segmentation:
  encoder:          resnet34
  encoder_weights:  imagenet
  img_size:         128
  classes:          1
  activation:       sigmoid
  batch_size:       16
  epochs:           70
  learning_rate:    0.00005
```

---

## 🧪 Reproducibility

- `set_seed(42)` applied to Python `random`, `numpy`, and `torch`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- All hyperparameters in `config.yaml` — no magic numbers in code

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Classification model | EfficientNet-B4 (`timm`) |
| Segmentation model | U-Net + ResNet34 (`segmentation-models-pytorch`) |
| Augmentation | `albumentations` |
| Web server | `Flask` |
| Frontend | HTML5 · Tailwind CSS · Vanilla JS · Chart.js |
| PDF generation | `reportlab` |
| Hardware acceleration | Apple MPS (`mps`) · NVIDIA CUDA (`cuda`) · CPU fallback |

---

## 📄 License

This project is for **educational and research purposes only** and is not intended for clinical diagnosis.

Datasets are sourced from Kaggle and are subject to their respective licenses:
- [Brain Tumor MRI Dataset License](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [LGG MRI Segmentation License](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## 📚 References

All models, architectures, loss functions, augmentations, and datasets in this project are based on the following papers:

### Model Architectures

**[1] EfficientNet** — Classification backbone  
Tan, M., & Le, Q. V. (2019).  
*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*  
ICML 2019. https://arxiv.org/abs/1905.11946

**[2] U-Net** — Segmentation architecture  
Ronneberger, O., Fischer, P., & Brox, T. (2015).  
*U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
MICCAI 2015. https://arxiv.org/abs/1505.04597

**[3] ResNet** — U-Net encoder backbone  
He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
*Deep Residual Learning for Image Recognition.*  
CVPR 2016. https://arxiv.org/abs/1512.03385

### Loss Functions & Metrics

**[4] Dice Loss** — Combined BCE + Dice segmentation loss  
Milletari, F., Navab, N., & Ahmadi, S. A. (2016).  
*V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*  
3DV 2016. https://arxiv.org/abs/1606.04797

**[5] Label Smoothing** — Used in classifier cross-entropy  
Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).  
*Rethinking the Inception Architecture for Computer Vision.*  
CVPR 2016. https://arxiv.org/abs/1512.00567

### Transfer Learning

**[6] ImageNet Pretraining** — Pretrained weights for EfficientNet & ResNet encoders  
Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009).  
*ImageNet: A Large-Scale Hierarchical Image Database.*  
CVPR 2009. https://ieeexplore.ieee.org/document/5206848

### Data Augmentation

**[7] Albumentations** — Fast image augmentation library used throughout  
Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020).  
*Albumentations: Fast and Flexible Image Augmentations.*  
Information 2020. https://arxiv.org/abs/1809.06839

### Datasets

**[8] LGG MRI Segmentation Dataset** — Segmentation training data  
Buda, M., Saha, A., & Mazurowski, M. A. (2019).  
*Association of Genomic Subtypes of Lower-Grade Gliomas with Shape Features Automatically Extracted by a Deep Learning Algorithm.*  
Computers in Biology and Medicine, 109. https://doi.org/10.1016/j.compbiomed.2019.05.002

**[9] Brain Tumor MRI Dataset** — Classification training data  
Nickparvar, M. (2021).  
*Brain Tumor MRI Dataset.*  
Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
