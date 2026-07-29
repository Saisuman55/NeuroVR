"""Unified inference pipeline for classification and segmentation."""
import argparse
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from data_loader import load_config, set_seed
from model import get_classifier, get_device, get_segmenter


def _best_path(path: str) -> str:
    """Return the best-model path by appending '_best' before the extension.

    Args:
        path: Original model checkpoint path.

    Returns:
        Path to the best checkpoint.
    """
    base, ext = os.path.splitext(path)
    return f"{base}_best{ext}"


def preprocess_classification(image_path: str, img_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for the classification model.

    Args:
        image_path: Path to the input image.
        img_size: Target square image size.

    Returns:
        Tuple of (preprocessed image tensor, original RGB image array).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0), original


def preprocess_segmentation(image_path: str, img_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for the segmentation model.

    Args:
        image_path: Path to the input image.
        img_size: Target square image size.

    Returns:
        Tuple of (preprocessed image tensor, original RGB image array).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0), original


def run_inference(image_path: str, config_path: str = "config.yaml") -> None:
    """Run the unified classification + segmentation inference pipeline.

    Classifies the input MRI image; if a tumor is detected, the segmentation
    model is run to produce a tumor mask overlay.

    Args:
        image_path: Path to the input MRI image.
        config_path: Path to the configuration file.
    """
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()

    class_names = config["classification"]["class_names"]
    no_tumor_class = "notumor"

    # Classification
    model_clf = get_classifier(
        num_classes=config["classification"]["num_classes"],
        dropout=config["classification"]["dropout"],
        model_name=config["classification"].get("model_name", "efficientnet_b3"),
    ).to(device)
    clf_checkpoint = torch.load(
        _best_path(config["paths"]["model_classifier"]), map_location=device
    )
    model_clf.load_state_dict(clf_checkpoint["model_state_dict"])
    model_clf.eval()

    input_clf, original_image = preprocess_classification(
        image_path, config["classification"]["img_size"]
    )
    input_clf = input_clf.to(device)

    with torch.no_grad():
        logits = model_clf(input_clf)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs[0, pred_idx].item()

    print(f"Predicted class: {pred_class} (confidence: {confidence:.4f})")
    print(f"Probabilities: {probs[0].tolist()}")
    print(f"Classes: {class_names}")

    output_dir = os.path.join(config["paths"]["outputs"], "predictions")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "inference_result.png")

    if pred_class == no_tumor_class:
        print("Classifier output 'notumor'. Running Segmentation Cross-Check...")

    # Segmentation
    model_seg = get_segmenter(
        encoder=config["segmentation"]["encoder"],
        encoder_weights=None,
        in_channels=config["segmentation"]["in_channels"],
        classes=config["segmentation"]["classes"],
        activation=config["segmentation"]["activation"],
    ).to(device)
    seg_checkpoint = torch.load(
        _best_path(config["paths"]["model_segmenter"]), map_location=device
    )
    model_seg.load_state_dict(seg_checkpoint["model_state_dict"])
    model_seg.eval()

    input_seg, _ = preprocess_segmentation(
        image_path, config["segmentation"]["img_size"]
    )
    input_seg = input_seg.to(device)

    with torch.no_grad():
        seg_output = model_seg(input_seg)
        # Model already has sigmoid activation, so seg_output is in [0, 1]
        seg_mask = seg_output.squeeze().cpu().numpy()
        seg_mask = (seg_mask > 0.5).astype(np.uint8)

    # Cross Check Logic
    tumor_pixels = seg_mask.sum()
    if pred_class == no_tumor_class:
        if tumor_pixels > 50:
            print(f"Cross-Check ALERT: Segmentation model detected {tumor_pixels} tumor pixels.")
            probs[0, pred_idx] = 0.0 # Zero out notumor probability
            new_pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[new_pred_idx]
            confidence = probs[0, new_pred_idx].item()
            print(f"Overridden Predicted class: {pred_class} (confidence: {confidence:.4f})")
            print(f"Probabilities: {probs[0].tolist()}")
            print(f"Classes: {class_names}")
        else:
            print("Cross-Check confirmed: No tumor mask found.")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(original_image)
            ax.set_title(f"No Tumor ({confidence:.3f})")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(result_path)
            return

    # Resize mask to original image dimensions
    h, w = original_image.shape[:2]
    seg_mask_resized = cv2.resize(seg_mask, (w, h))

    # Also resize raw probability map (before thresholding) for heatmap
    seg_prob_resized = cv2.resize(
        seg_output.squeeze().cpu().numpy(), (w, h)
    )

    def _save_img(arr_rgb: np.ndarray, path: str) -> None:
        """Save an RGB numpy array as a PNG."""
        cv2.imwrite(path, cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))

    # 1. Original
    original_path = os.path.join(output_dir, "original.png")
    _save_img(original_image, original_path)

    # 2. Binary mask (white on black)
    mask_vis = (seg_mask_resized * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
    mask_path = os.path.join(output_dir, "binary_mask.png")
    _save_img(mask_rgb, mask_path)

    # 3. Green overlay
    overlay = original_image.copy()
    green_channel = np.zeros_like(overlay)
    green_channel[:, :, 1] = (seg_mask_resized * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.7, green_channel, 0.3, 0)
    overlay_path = os.path.join(output_dir, "green_overlay.png")
    _save_img(overlay, overlay_path)

    # 4. Contour boundary drawn on original
    contour_img = original_image.copy()
    contours, _ = cv2.findContours(
        seg_mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_bgr = cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(contour_bgr, contours, -1, (0, 255, 255), 2)
    contour_img = cv2.cvtColor(contour_bgr, cv2.COLOR_BGR2RGB)
    contour_path = os.path.join(output_dir, "contour.png")
    _save_img(contour_img, contour_path)

    # 5. Heat map — JET colormap on raw probability map (real model output)
    prob_uint8 = (np.clip(seg_prob_resized, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 0.5, heatmap_bgr, 0.5, 0
    )
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    cv2.imwrite(heatmap_path, blended)

    # Legacy combined figure (kept for backwards compat)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(seg_mask_resized, cmap="gray")
    axes[1].set_title(f"Predicted Mask\nClass: {pred_class}")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay: {pred_class} ({confidence:.3f})")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()

    print(f"Tumor type: {pred_class}, segmentation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run unified classification + segmentation inference on a single MRI image."
    )
    parser.add_argument("--image", required=True, help="Path to MRI image.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration YAML."
    )
    args = parser.parse_args()

    run_inference(args.image, args.config)
