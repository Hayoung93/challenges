from typing import Callable

import torchvision.transforms as T

from .genai_transforms import (
    CurricularWrapper,
    RandomDownscaleUpscale,
    RandomGaussianNoise,
    RandomJPEGCompression,
    RandomPNGReencode,
)

# ImageNet normalization (matches MambaVision backbone)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _strong_geometric(image_size: int) -> list:
    """Shared geometric + color augmentations for strong / genai pipelines."""
    return [
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]


def _to_tensor_normalize() -> list:
    """Shared final steps: PIL → Tensor → Normalize."""
    return [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def get_train_transform(
    image_size: int = 224,
    augmentation: str = "default",
    total_epochs: int = 30,
    epoch_state=None,
) -> Callable:
    """Build training transform pipeline.

    Args:
        image_size: Target crop size.
        augmentation: One of ``"none"``, ``"default"``, ``"strong"``,
            ``"genai"``, ``"genai_curriculum"``.
        total_epochs: Total training epochs (used by ``"genai_curriculum"``).
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop (used by ``"genai_curriculum"``).
    """
    if augmentation == "none":
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif augmentation == "default":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif augmentation == "strong":
        return T.Compose(
            _strong_geometric(image_size) + _to_tensor_normalize()
        )
    elif augmentation == "genai":
        return T.Compose(
            _strong_geometric(image_size)
            + [
                RandomJPEGCompression(quality_range=(30, 95), p=0.5),
                RandomDownscaleUpscale(scale_range=(0.5, 0.9), p=0.3),
                RandomGaussianNoise(std_range=(1.0, 10.0), p=0.3),
                RandomPNGReencode(p=0.2),
            ]
            + _to_tensor_normalize()
        )
    elif augmentation == "genai_curriculum":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        curricular = CurricularWrapper(
            transforms=[
                RandomJPEGCompression(quality_range=(30, 95), p=0.5),
                RandomDownscaleUpscale(scale_range=(0.5, 0.9), p=0.3),
                RandomGaussianNoise(std_range=(1.0, 10.0), p=0.3),
                RandomPNGReencode(p=0.2),
            ],
            epoch_state=epoch_state,
            total_epochs=total_epochs,
            min_scale=0.1,
        )
        return T.Compose(
            _strong_geometric(image_size) + [curricular] + _to_tensor_normalize()
        )
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def get_val_transform(
    image_size: int = 224,
    resize_size: int = 256,
) -> Callable:
    """Build validation/test transform pipeline."""
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
