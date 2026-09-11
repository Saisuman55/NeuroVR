# NeuroVR model training

Research and educational use only. Not intended for clinical diagnosis.

- `train_2d_classifier.py`: EfficientNet-B0 CNN trained on the CC BY 4.0 Cheng CE-MRI dataset. Splits are deterministic and patient-separated. Classes are glioma, meningioma, and pituitary; this dataset does not contain a healthy class.
- `train_3d_segmentation.py`: MONAI SegResNet fine-tuning on CC BY-SA 4.0 Medical Segmentation Decathlon Task01. It starts from the existing MONAI BraTS checkpoint and predicts TC, WT, and ET jointly.

Default runs use 20 epochs. Checkpoints are selected only by validation metrics and saved separately from the production model. Review `metrics.json` before promotion.
