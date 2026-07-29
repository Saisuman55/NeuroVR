"""Evaluation scripts for classification and segmentation models."""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import (
    get_classification_loaders,
    get_segmentation_loaders,
    load_config,
    set_seed,
)
from model import (
    dice_coefficient,
    get_classifier,
    get_device,
    get_segmenter,
    iou_score,
)


def _best_path(path: str) -> str:
    """Return the best-model path by appending '_best' before the extension.

    Args:
        path: Original model checkpoint path.

    Returns:
        Path to the best checkpoint.
    """
    base, ext = os.path.splitext(path)
    return f"{base}_best{ext}"


def evaluate_classifier(config_path: str = "config.yaml") -> None:
    """Evaluate the classification model on the test set.

    Produces a confusion matrix plot and a classification report CSV.

    Args:
        config_path: Path to configuration file.
    """
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()

    _, _, test_loader, class_names = get_classification_loaders(config)

    model = get_classifier(
        num_classes=config["classification"]["num_classes"],
        dropout=config["classification"]["dropout"],
        model_name=config["classification"].get("model_name", "efficientnet_b3"),
    ).to(device)

    checkpoint_path = _best_path(config["paths"]["model_classifier"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    output_dir = os.path.join(config["paths"]["outputs"], "plots")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_confusion_matrix.png"))
    plt.close()

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(
        f"Precision: {precision_score(all_labels, all_preds, average='macro'):.4f}"
    )
    print(
        f"Recall:    {recall_score(all_labels, all_preds, average='macro'):.4f}"
    )
    print(f"F1-Score:  {f1_score(all_labels, all_preds, average='macro'):.4f}")


def evaluate_segmenter(
    config_path: str = "config.yaml", num_visualize: int = 10
) -> None:
    """Evaluate the segmentation model on the validation/test set.

    Computes per-sample Dice and IoU and saves overlay visualizations.

    Args:
        config_path: Path to configuration file.
        num_visualize: Number of sample overlays to save.
    """
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()

    _, val_loader = get_segmentation_loaders(config)

    model = get_segmenter(
        encoder=config["segmentation"]["encoder"],
        encoder_weights=None,
        in_channels=config["segmentation"]["in_channels"],
        classes=config["segmentation"]["classes"],
        activation=config["segmentation"]["activation"],
    ).to(device)

    checkpoint_path = _best_path(config["paths"]["model_segmenter"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dice_scores = []
    iou_scores = []
    samples = []

    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                dice = dice_coefficient(outputs[i], masks[i]).item()
                iou = iou_score(outputs[i], masks[i]).item()
                dice_scores.append(dice)
                iou_scores.append(iou)

                if len(samples) < num_visualize:
                    samples.append(
                        (inputs[i].cpu(), masks[i].cpu(), outputs[i].cpu(), dice, iou)
                    )

    dice_scores = np.array(dice_scores)
    iou_scores = np.array(iou_scores)

    output_dir = os.path.join(config["paths"]["outputs"], "predictions")
    os.makedirs(output_dir, exist_ok=True)

    mean_dice = dice_scores.mean()
    mean_iou = iou_scores.mean()
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU:  {mean_iou:.4f}")

    for idx, (img, mask, pred, dice, iou) in enumerate(samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (
            img_np * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        )
        img_np = np.clip(img_np, 0, 1)

        pred_mask = (pred.numpy().squeeze() > 0.5).astype(np.uint8)
        true_mask = mask.numpy().squeeze().astype(np.uint8)

        overlay = img_np.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = pred_mask
        overlay = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title(f"Predicted Mask\nDice: {dice:.3f}")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nIoU: {iou:.3f}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"segmentation_sample_{idx}.png"))
        plt.close()

    results_df = pd.DataFrame({"dice": dice_scores, "iou": iou_scores})
    results_df.to_csv(
        os.path.join(output_dir, "segmentation_scores.csv"), index=False
    )

    with open(
        os.path.join(output_dir, "segmentation_summary.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Mean IoU:  {mean_iou:.4f}\n")
        f.write(f"Samples:   {len(dice_scores)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate brain tumor classification and/or segmentation models."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "segmentation", "both"],
        default="both",
        help="Which task to evaluate.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration YAML."
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=10,
        help="Number of segmentation overlays to generate.",
    )
    args = parser.parse_args()

    if args.task in ["classification", "both"]:
        evaluate_classifier(args.config)
    if args.task in ["segmentation", "both"]:
        evaluate_segmenter(args.config, args.num_visualize)
