"""Model architectures, loss functions, and metric helpers."""
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp


def get_device() -> torch.device:
    """Auto-detect CUDA or MPS device, falling back to CPU.

    Returns:
        torch.device object set to the best available accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_classifier(
    num_classes: int = 4, dropout: float = 0.4, freeze_backbone: bool = True, model_name: str = "efficientnet_b3"
) -> nn.Module:
    """Build an EfficientNet classifier with a custom head.

    Uses ImageNet-pretrained weights and replaces the final classification
    layer with Dropout + Linear(num_classes).

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, freeze all backbone parameters initially.
        model_name: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7').

    Returns:
        PyTorch EfficientNet model.
    """
    weight_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        "efficientnet_b2": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        "efficientnet_b3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "efficientnet_b5": models.EfficientNet_B5_Weights.IMAGENET1K_V1,
        "efficientnet_b6": models.EfficientNet_B6_Weights.IMAGENET1K_V1,
        "efficientnet_b7": models.EfficientNet_B7_Weights.IMAGENET1K_V1,
    }
    weights = weight_map.get(model_name, models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    model = getattr(models, model_name)(
        weights=weights
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_top_layers(model: nn.Module, num_layers: int = 30) -> None:
    """Unfreeze the top N layers of an EfficientNet backbone for fine-tuning.

    Args:
        model: EfficientNet model whose backbone layers will be unfrozen.
        num_layers: Number of top layers to unfreeze.
    """
    children = list(model.features.children())
    for layer in children[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def get_segmenter(
    encoder: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str = "sigmoid",
) -> nn.Module:
    """Build a U-Net with a ResNet34 encoder for binary segmentation.

    Args:
        encoder: Encoder backbone name (e.g., 'resnet34').
        encoder_weights: Pretrained weights to load ('imagenet' or None).
        in_channels: Number of input channels (3 for RGB/grayscale-as-RGB).
        classes: Number of output classes (1 for binary segmentation).
        activation: Output activation ('sigmoid' or 'softmax2d').

    Returns:
        PyTorch U-Net model from segmentation_models_pytorch.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice loss.

    Expects model outputs to already be in probability space (sigmoid applied).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        """Initialize the combined loss.

        Args:
            bce_weight: Weight for BCE term.
            dice_weight: Weight for Dice term.
            smooth: Smoothing constant to avoid division by zero.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE + Dice loss.

        Args:
            probs: Predicted probability tensor from sigmoid output.
            targets: Ground truth binary mask tensor.

        Returns:
            Scalar loss value.
        """
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        bce_loss = self.bce(probs, targets)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        dice_loss = 1.0 - dice
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute the Dice coefficient for a single prediction-target pair.

    Args:
        pred: Predicted tensor (logits or probabilities).
        target: Ground truth binary mask tensor.
        smooth: Smoothing constant.

    Returns:
        Dice coefficient scalar.
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute the Intersection over Union (Jaccard Index).

    Args:
        pred: Predicted tensor (logits or probabilities).
        target: Ground truth binary mask tensor.
        smooth: Smoothing constant.

    Returns:
        IoU scalar.
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)
