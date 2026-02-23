"""Test-Time Augmentation (TTA) strategies for GenAI image detection."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TTA_MODES = ["none", "flip", "multiscale", "full"]

DEFAULT_SCALES = [256, 288, 320]


def _center_crop(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Center-crop a ``(B, C, H, W)`` tensor to ``(B, C, crop_size, crop_size)``."""
    _, _, h, w = tensor.shape
    if h == crop_size and w == crop_size:
        return tensor
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return tensor[:, :, top : top + crop_size, left : left + crop_size]


def _resize_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` tensor to ``(B, C, size, size)`` via bilinear interpolation."""
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def generate_augmented_views(
    images: torch.Tensor,
    tta_mode: str,
    image_size: int = 224,
    scales: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """Generate augmented views of the input batch.

    Each returned tensor has shape ``(B, C, image_size, image_size)``.

    Args:
        images: Original batch ``(B, C, H, W)``, already normalized.
        tta_mode: One of ``"none"``, ``"flip"``, ``"multiscale"``, ``"full"``.
        image_size: Spatial size the model expects.
        scales: Resize targets for multi-scale crops.

    Returns:
        List of ``(B, C, image_size, image_size)`` tensors.
    """
    if scales is None:
        scales = DEFAULT_SCALES

    views: List[torch.Tensor] = [images]

    if tta_mode == "none":
        return views

    if tta_mode in ("flip", "full"):
        views.append(torch.flip(images, dims=[-1]))

    if tta_mode == "full":
        views.append(torch.rot90(images, k=1, dims=[-2, -1]))
        views.append(torch.rot90(images, k=2, dims=[-2, -1]))
        views.append(torch.rot90(images, k=3, dims=[-2, -1]))

    if tta_mode in ("multiscale", "full"):
        for scale in scales:
            resized = _resize_tensor(images, scale)
            cropped = _center_crop(resized, image_size)
            views.append(cropped)

    return views


@torch.no_grad()
def tta_forward(
    model: nn.Module,
    images: torch.Tensor,
    tta_mode: str,
    image_size: int = 224,
    scales: Optional[List[int]] = None,
) -> torch.Tensor:
    """Run TTA-augmented forward pass and return averaged logits.

    Processes views sequentially to avoid OOM.

    Args:
        model: Classifier model.
        images: Input batch ``(B, C, H, W)``.
        tta_mode: TTA strategy name.
        image_size: Expected model input spatial size.
        scales: Multi-scale resize targets.

    Returns:
        Averaged logits of shape ``(B, num_classes)``.
    """
    if tta_mode == "none":
        return model(images)

    views = generate_augmented_views(images, tta_mode, image_size, scales)

    logits_sum = None
    for view in views:
        logits = model(view)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    return logits_sum / len(views)
