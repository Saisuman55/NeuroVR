---
title: NeuroVR — BrainTumor AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
license: mit
short_description: Brain MRI tumor classification & segmentation
---

<div align="center">

<img src="https://img.shields.io/badge/🧠-NeuroVR%20BrainTumor%20AI-0ea5e9?style=for-the-badge" alt="NeuroVR"/>

# 🧠 NeuroVR — BrainTumor AI

### Clinical-Grade MRI Tumor Classification & Segmentation

[![🤗 Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/swaggersamantaray55/NeuroVR)
[![GitHub](https://img.shields.io/badge/GitHub-Saisuman55%2FNeuroVR-181717?style=for-the-badge&logo=github)](https://github.com/Saisuman55/NeuroVR)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![ZeroGPU](https://img.shields.io/badge/ZeroGPU-A100%20Free-7c3aed?style=for-the-badge)](https://huggingface.co/docs/hub/spaces-zerogpu)

<br/>

> End-to-end deep learning pipeline for brain MRI **tumor classification** and **pixel-level segmentation** — served as a medical-grade AI interface on HuggingFace Spaces with ZeroGPU.

<br/>

**🔗 Try it live → [huggingface.co/spaces/swaggersamantaray55/NeuroVR](https://huggingface.co/spaces/swaggersamantaray55/NeuroVR)**

</div>

---

## 🎯 What It Does

Upload any brain MRI scan and get **instant AI analysis** in under 10 seconds:

| Step | Model | Output |
|---|---|---|
| **1. Classification** | EfficientNet-B4 | Tumor type + confidence + class probabilities |
| **2. Segmentation** | U-Net + ResNet34 | Binary tumor mask + 5 visualization outputs |
| **3. Cross-Check** | Logic layer | Validates classifier against segmentation mask |
| **4. Report** | ReportLab PDF | Downloadable clinical-style report with all results |

---

## ✨ Features

| Feature | Details |
|---|---|
| **4-Class Classification** | Glioma · Meningioma · Pituitary · No Tumor |
| **Pixel-Level Segmentation** | U-Net + ResNet34 binary tumor mask |
| **Cross-Check Validation** | Segmentation corrects classifier false negatives |
| **5-Panel Output Gallery** | Original · Binary Mask · Green Overlay · Contour · Heatmap |
| **Confidence Bar Chart** | Per-class probability visualization |
| **PDF Clinical Report** | Patient info, diagnosis, all outputs — one-click download |
| **ZeroGPU Powered** | Free A100 GPU inference on HuggingFace |
| **Medical-Grade UI** | Dark radiology theme, risk badges, analysis IDs |

---

## 🏗️ System Architecture

```
MRI Image Upload (Gradio UI)
        │
        ▼
 EfficientNet-B4 ──► Class + Confidence + All Class Probabilities
        │
        ├─── notumor? ──► Segmentation Cross-Check
        │
        ▼
 U-Net + ResNet34 ──► Binary Tumor Mask
        │
        ├── > 50 tumor pixels? ──► Override classifier → tumor confirmed
        │
        ▼
  5 Output Images
  ├── original.png
  ├── binary_mask.png
  ├── green_overlay.png
  ├── contour.png
  └── heatmap.png
        │
        ▼
  Gradio UI Result Card + PDF Report
```

---

## 🚀 Live Demo

**→ [huggingface.co/spaces/swaggersamantaray55/NeuroVR](https://huggingface.co/spaces/swaggersamantaray55/NeuroVR)**

Or embed directly in any website:

```html
<iframe
  src="https://swaggersamantaray55-neurovr.hf.space"
  width="100%"
  height="900"
  frameborder="0"
  allow="camera;microphone"
  style="border-radius:12px;box-shadow:0 4px 32px rgba(0,0,0,.4)"
></iframe>
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
├── stitch_frontend/          # Local web dashboard (HTML/CSS/JS)
│   ├── index.html            # Home — pipeline overview + 3D brain
│   ├── inference.html        # Upload MRI, view results, export PDF
│   ├── training_status.html  # Live training charts
│   ├── about.html            # Model details & architecture
│   └── metrics.json          # Written by monitor.py during training
│
├── models/                   # Auto-downloaded from HF Hub at runtime
│   ├── classifier/           # brain_tumor_classifier_best.pth (~71 MB)
│   └── segmenter/            # brain_tumor_segmenter_best.pth (~98 MB)
│
├── app.py                    # Gradio app — HF Spaces entry point
├── download_models.py        # Auto-downloads weights from HF Hub on startup
├── monitor.py                # Parses training logs → updates metrics.json
├── config.yaml               # All hyperparameters and file paths
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

### Requirements

- Python **3.9 – 3.12**
- macOS (Apple MPS) **or** Linux/Windows with CUDA GPU
- ~4 GB RAM minimum; 8+ GB recommended for training

### 1. Clone

```bash
git clone https://github.com/Saisuman55/NeuroVR.git
cd NeuroVR
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

| Task | Dataset | Kaggle Link | Place at |
|---|---|---|---|
| Classification | Brain Tumor MRI Dataset | [Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | `data/classification/` |
| Segmentation | LGG MRI Segmentation | [Link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | `data/segmentation/` |

### 4. Model Weights

Weights (~170 MB total) are automatically downloaded from HuggingFace Hub when running on Spaces.

For local use, either:
- Set `HF_MODEL_REPO=swaggersamantaray55/brain-tumor-ai-weights` and run `python download_models.py`
- Or place them manually:
  ```
  models/classifier/brain_tumor_classifier_best.pth
  models/segmenter/brain_tumor_segmenter_best.pth
  ```

### 5. Run locally

```bash
python app.py
```

Opens at **[http://localhost:7860](http://localhost:7860)**

---

## 🏋️ Training

Training runs in two phases automatically:

1. **Phase 1 — Frozen backbone**: Only the classification head is trained
2. **Phase 2 — Fine-tuning**: Top EfficientNet layers unfrozen at lower LR

```bash
# Terminal 1 — start training
python src/train_m5_optimized.py 2>&1 | tee outputs/training_log.txt

# Terminal 2 — live monitor (updates metrics.json for dashboard)
python monitor.py
```

---

## 🔬 Inference (CLI)

```bash
python src/inference.py --image test_samples/glioma_Te-gl_1.jpg
```

Saves 5 images to `outputs/predictions/` and prints:

```
Predicted class: GLIOMA (confidence: 0.9472)
Probabilities: [0.9472, 0.0312, 0.0189, 0.0027]
Classes: ['glioma', 'meningioma', 'pituitary', 'notumor']
```

---

## 📊 Evaluation

```bash
python src/evaluate.py --task both
python src/evaluate.py --task classification
python src/evaluate.py --task segmentation --num_visualize 10
```

---

## 📈 Results

| Model | Task | Metric | Value |
|---|---|---|---|
| EfficientNet-B4 | 4-class classification | Accuracy | >98% |
| U-Net + ResNet34 | Binary segmentation | Dice Coefficient | >0.80 |

**Loss functions:**
- Classifier: Cross-Entropy + Label Smoothing (0.1)
- Segmenter: BCE + Dice combined loss (50/50)

---

## 🧪 Test Samples

12 real MRI images included in `test_samples/` — 3 per class:

```
glioma_Te-gl_1.jpg        glioma_Te-gl_10.jpg        glioma_Te-gl_100.jpg
meningioma_Te-aug-me_1.jpg meningioma_Te-aug-me_10.jpg meningioma_Te-aug-me_100.jpg
notumor_Te-no_1.jpg        notumor_Te-no_10.jpg        notumor_Te-no_100.jpg
pituitary_Te-pi_1.jpg      pituitary_Te-pi_10.jpg      pituitary_Te-pi_100.jpg
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Classification** | EfficientNet-B4 (`timm`) |
| **Segmentation** | U-Net + ResNet34 (`segmentation-models-pytorch`) |
| **Augmentation** | `albumentations` |
| **AI Interface** | `Gradio 4.44` on HuggingFace Spaces |
| **GPU** | ZeroGPU (free A100 via `spaces.GPU`) |
| **Model Hub** | HuggingFace Hub (`huggingface_hub`) |
| **PDF Reports** | `reportlab` |
| **Local Frontend** | HTML5 · Tailwind CSS · Vanilla JS · Three.js |
| **Hardware** | Apple MPS · NVIDIA CUDA · CPU fallback |

---

## 🔧 Configuration (`config.yaml`)

```yaml
seed: 42

paths:
  model_classifier:  models/classifier/brain_tumor_classifier.pth
  model_segmenter:   models/segmenter/brain_tumor_segmenter.pth
  outputs:           outputs

classification:
  model_name:    efficientnet_b4
  img_size:      380
  num_classes:   4
  class_names:   [glioma, meningioma, notumor, pituitary]
  dropout:       0.2

segmentation:
  encoder:       resnet34
  img_size:      128
  classes:       1
  activation:    sigmoid
```

---

## 📄 License

MIT License — **educational and research purposes only**.  
Not intended or approved for clinical diagnosis.

Datasets subject to their respective Kaggle licenses.

---

## 📚 References

**[1] EfficientNet** — Tan & Le, ICML 2019. https://arxiv.org/abs/1905.11946  
**[2] U-Net** — Ronneberger et al., MICCAI 2015. https://arxiv.org/abs/1505.04597  
**[3] ResNet** — He et al., CVPR 2016. https://arxiv.org/abs/1512.03385  
**[4] Dice Loss** — Milletari et al., 3DV 2016. https://arxiv.org/abs/1606.04797  
**[5] Albumentations** — Buslaev et al., 2020. https://arxiv.org/abs/1809.06839  
**[6] LGG MRI Dataset** — Buda et al., 2019. https://doi.org/10.1016/j.compbiomed.2019.05.002  
**[7] Brain Tumor MRI Dataset** — Nickparvar, Kaggle 2021.

---

<div align="center">

**Built with ❤️ · Powered by PyTorch + HuggingFace ZeroGPU**

[![🤗 Live Demo](https://img.shields.io/badge/🤗%20Try%20Live%20Demo-NeuroVR-ff9d00?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/swaggersamantaray55/NeuroVR)

</div>
