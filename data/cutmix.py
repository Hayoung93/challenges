"""Same-label CutMix augmentation for robust training pipelines.

CutMix replaces a rectangular region of one image with a patch from
another image.  The "same-label" constraint ensures mixing only happens
between images of the same class (real+real or fake+fake), so the
output label remains unchanged -- no soft label mixing is needed.

Operates on GPU tensors at the batch level (after collation, before
the forward pass).  Only active for robust augmentation modes.
"""

import random

import torch


def _rand_bbox(H: int, W: int, lam: float):
    """Generate random bounding box for CutMix.

    The box area is approximately ``(1 - lam) * H * W``.

    Returns:
        ``(y1, x1, y2, x2)`` coordinates, or ``None`` if the box is
        degenerate (height or width < 2).
    """
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = random.randint(0, H - 1)
    cx = random.randint(0, W - 1)

    y1 = max(0, cy - cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    y2 = min(H, y1 + cut_h)
    x2 = min(W, x1 + cut_w)

    if (y2 - y1) < 2 or (x2 - x1) < 2:
        return None
    return y1, x1, y2, x2


def _rand_bbox_mask(B: int, H: int, W: int, lam: float,
                    device: torch.device) -> torch.Tensor | None:
    """Generate per-sample random bounding box masks for CutMix.

    Each sample gets an independent random box position, but all boxes
    share the same size (determined by *lam*) to keep the mixing ratio
    consistent across the batch.

    Returns:
        ``(B, 1, H, W)`` boolean mask tensor, or ``None`` if the box
        size is degenerate (height or width < 2).
    """
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    if cut_h < 2 or cut_w < 2:
        return None

    # Per-sample random centers
    cy = torch.randint(0, H, (B,), device=device)
    cx = torch.randint(0, W, (B,), device=device)

    y1 = (cy - cut_h // 2).clamp(min=0)
    x1 = (cx - cut_w // 2).clamp(min=0)
    y2 = (y1 + cut_h).clamp(max=H)
    x2 = (x1 + cut_w).clamp(max=W)

    # Build mask: (B, 1, H, W)
    rows = torch.arange(H, device=device).view(1, 1, H, 1)
    cols = torch.arange(W, device=device).view(1, 1, 1, W)

    mask = ((rows >= y1.view(B, 1, 1, 1)) & (rows < y2.view(B, 1, 1, 1))
            & (cols >= x1.view(B, 1, 1, 1)) & (cols < x2.view(B, 1, 1, 1)))
    return mask


def _build_perm_index(labels):
    """Build a batch-wide permutation index for same-label pairing.

    For each label group with >= 2 samples, creates a random permutation
    within the group.  Samples that are the sole representative of their
    label map to themselves (no-op).

    Returns:
        ``(B,)`` index tensor on the same device as *labels*.
    """
    B = labels.size(0)
    perm_idx = torch.arange(B, device=labels.device)
    for lab in labels.unique():
        mask = (labels == lab).nonzero(as_tuple=True)[0]
        if mask.size(0) < 2:
            continue
        perm_idx[mask] = mask[torch.randperm(mask.size(0), device=mask.device)]
    return perm_idx


def same_label_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    p: float = 0.15,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Apply CutMix only between images with the same label.

    Args:
        images: ``(B, C, H, W)`` batch of images (GPU tensor).
        labels: ``(B,)`` integer labels.
        p: Probability of applying CutMix to the batch.
        alpha: Beta distribution parameter controlling cut size.

    Returns:
        Modified images tensor (clone of input).
    """
    if random.random() > p or images.size(0) < 2:
        return images

    B, C, H, W = images.shape
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    mask = _rand_bbox_mask(B, H, W, lam, images.device)
    if mask is None:
        return images

    perm_idx = _build_perm_index(labels)
    return torch.where(mask, images[perm_idx], images)


def same_label_cutmix_multi_view(
    views1: torch.Tensor,
    views2: torch.Tensor,
    labels: torch.Tensor,
    p: float = 0.15,
    alpha: float = 0.4,
) -> tuple:
    """Apply same-label CutMix to both views in multi-view training.

    Uses the same cut box and the same pairing for both views to
    maintain consistency.

    Returns:
        ``(mixed_views1, mixed_views2)``.
    """
    if random.random() > p or views1.size(0) < 2:
        return views1, views2

    if views1.shape != views2.shape:
        raise ValueError(
            f"views1 and views2 must have the same shape, "
            f"got {views1.shape} and {views2.shape}"
        )

    B, C, H, W = views1.shape
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    mask = _rand_bbox_mask(B, H, W, lam, views1.device)
    if mask is None:
        return views1, views2

    perm_idx = _build_perm_index(labels)
    result1 = torch.where(mask, views1[perm_idx], views1)
    result2 = torch.where(mask, views2[perm_idx], views2)
    return result1, result2
