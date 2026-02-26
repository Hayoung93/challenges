"""Test-Time Augmentation (TTA) strategies for GenAI image detection.

Supports two regimes:

1. **Legacy** (H == W == image_size): All views are generated from the
   image_size tensor, identical to the original implementation.

2. **Prep-tensor** (H == W > image_size): Generates pixel-preserving
   crop views (center crop, corner crops, flipped center crop) alongside
   resize-based views.  Crop-only views always outnumber resize views
   in ``"full"`` mode.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TTA_MODES = ["none", "flip", "multiscale", "full", "full_legacy"]

DEFAULT_SCALES = [256, 288, 320]


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════


def _center_crop(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Center-crop a ``(B, C, H, W)`` tensor to ``(B, C, crop_size, crop_size)``."""
    _, _, h, w = tensor.shape
    if h == crop_size and w == crop_size:
        return tensor
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"crop_size ({crop_size}) exceeds tensor spatial dims ({h}x{w})"
        )
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return tensor[:, :, top : top + crop_size, left : left + crop_size]


def _corner_crops(tensor: torch.Tensor, crop_size: int) -> List[torch.Tensor]:
    """Extract four corner crops of ``crop_size`` from ``(B, C, H, W)``.

    Returns crops from: top-left, top-right, bottom-left, bottom-right.
    All crops are pure tensor slices — no interpolation.
    """
    _, _, h, w = tensor.shape
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"crop_size ({crop_size}) exceeds tensor spatial dims ({h}x{w})"
        )
    return [
        tensor[:, :, :crop_size, :crop_size],
        tensor[:, :, :crop_size, w - crop_size :],
        tensor[:, :, h - crop_size :, :crop_size],
        tensor[:, :, h - crop_size :, w - crop_size :],
    ]


def _resize_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` tensor to ``(B, C, size, size)`` via bilinear interpolation."""
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def _has_prep_margin(tensor: torch.Tensor, image_size: int) -> bool:
    """Check if the tensor is larger than the model's expected input size."""
    _, _, h, w = tensor.shape
    return h > image_size and w > image_size


# ═══════════════════════════════════════════════════════════════════
#  View generation
# ═══════════════════════════════════════════════════════════════════


def generate_augmented_views(
    images: torch.Tensor,
    tta_mode: str,
    image_size: int = 224,
    scales: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """Generate augmented views of the input batch.

    Automatically selects between the legacy and prep-tensor code paths
    based on whether ``images`` is larger than ``image_size``.

    Each returned tensor has shape ``(B, C, image_size, image_size)``.

    Args:
        images: Original batch ``(B, C, H, W)``, already normalized.
        tta_mode: One of ``"none"``, ``"flip"``, ``"multiscale"``, ``"full"``.
        image_size: Spatial size the model expects.
        scales: Resize targets for multi-scale crops.

    Returns:
        List of ``(B, C, image_size, image_size)`` tensors.
    """
    if tta_mode not in TTA_MODES:
        raise ValueError(
            f"Invalid tta_mode '{tta_mode}'. Must be one of {TTA_MODES}"
        )

    if scales is None:
        scales = DEFAULT_SCALES
        # Auto-adjust default scales for larger image_size so that
        # multiscale/full TTA modes work without user intervention.
        if tta_mode in ("multiscale", "full", "full_legacy"):
            if any(s < image_size for s in scales):
                scales = [image_size + (i + 1) * 32 for i in range(len(DEFAULT_SCALES))]

    # Validate user-provided scales: all must be >= image_size for resize→crop
    if tta_mode in ("multiscale", "full", "full_legacy"):
        for scale in scales:
            if scale < image_size:
                raise ValueError(
                    f"TTA scale {scale} is smaller than image_size {image_size}. "
                    f"All scales must be >= image_size."
                )

    # full_legacy: center-crop to image_size, then legacy rotation-based views
    if tta_mode == "full_legacy":
        if _has_prep_margin(images, image_size):
            images = _center_crop(images, image_size)
        return _generate_legacy_views(images, "full", image_size, scales)

    if _has_prep_margin(images, image_size):
        return _generate_prep_views(images, tta_mode, image_size, scales)
    return _generate_legacy_views(images, tta_mode, image_size, scales)


def _generate_legacy_views(
    images: torch.Tensor,
    tta_mode: str,
    image_size: int,
    scales: List[int],
) -> List[torch.Tensor]:
    """Original view generation for ``(B, C, image_size, image_size)`` tensors.

    Preserves exact backward compatibility with the previous implementation.
    """
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


def _generate_prep_views(
    images: torch.Tensor,
    tta_mode: str,
    image_size: int,
    scales: List[int],
) -> List[torch.Tensor]:
    """View generation for larger prep tensors (H == W > image_size).

    Generates pixel-preserving crops (no interpolation) and optionally
    resize-based views.  In ``"full"`` mode the crop-only to resize view
    ratio is 6:3 = 2:1.

    View counts by mode::

        "none"      : 1 crop-only
        "flip"      : 2 crop-only
        "multiscale": 1 crop-only + len(scales) resize
        "full"      : 6 crop-only + len(scales) resize   (2:1 ratio)
    """
    center = _center_crop(images, image_size)
    views: List[torch.Tensor] = [center]

    if tta_mode == "none":
        return views

    if tta_mode in ("flip", "full"):
        views.append(torch.flip(center, dims=[-1]))

    if tta_mode == "full":
        views.extend(_corner_crops(images, image_size))

    if tta_mode in ("multiscale", "full"):
        for scale in scales:
            resized = _resize_tensor(images, scale)
            cropped = _center_crop(resized, image_size)
            views.append(cropped)

    return views


# ═══════════════════════════════════════════════════════════════════
#  TTA forward
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def tta_forward(
    model: nn.Module,
    images: torch.Tensor,
    tta_mode: str,
    image_size: int = 224,
    scales: Optional[List[int]] = None,
) -> torch.Tensor:
    """Run TTA-augmented forward pass and return averaged logits.

    Processes views sequentially to avoid OOM.  Transparently handles
    both legacy ``(B, C, image_size, image_size)`` inputs and larger
    prep tensors ``(B, C, prep_size, prep_size)``.

    Args:
        model: Classifier model.
        images: Input batch ``(B, C, H, W)``.
        tta_mode: TTA strategy name.
        image_size: Expected model input spatial size.
        scales: Multi-scale resize targets.

    Returns:
        Averaged logits of shape ``(B, num_classes)``.
    """
    if tta_mode == "none" and not _has_prep_margin(images, image_size):
        return model(images)

    if tta_mode == "none" and _has_prep_margin(images, image_size):
        return model(_center_crop(images, image_size))

    views = generate_augmented_views(images, tta_mode, image_size, scales)

    logits_sum = None
    for view in views:
        logits = model(view)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    return logits_sum / len(views)
