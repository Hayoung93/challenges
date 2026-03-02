from typing import Callable

import torchvision.transforms as T

from .genai_transforms import (
    CurricularGroupedNOfCompose,
    CurricularNOfCompose,
    CurricularWrapper,
    GroupedNOfCompose,
    RandomBoxBlur,
    RandomBrightnessCurve,
    RandomColorQuantization,
    RandomContrastCurve,
    RandomDownscaleUpscale,
    RandomGammaCorrection,
    RandomGaussianNoise,
    RandomImpulseNoise,
    RandomJPEGCompression,
    RandomLensBlur,
    RandomMedianBlur,
    RandomMotionBlur,
    RandomNOfCompose,
    RandomPixelization,
    RandomPNGReencode,
    RandomResizeOrCrop,
    RandomSaltPepperNoise,
    RandomSharpen,
    RandomSpatialJitter,
    RandomWebPCompression,
)

# ImageNet normalization (matches MambaVision backbone)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeIfSmaller:
    """Resize the image only if its shortest edge is smaller than *min_size*.

    When the shortest edge is already >= *min_size* the image is returned
    untouched, preserving original pixel-level artifacts.
    """

    def __init__(self, min_size: int):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) >= self.min_size:
            return img
        return T.functional.resize(img, self.min_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(min_size={self.min_size})"


class ReflectPadIfSmaller:
    """Reflect-pad the image if its shortest edge is smaller than *min_size*.

    When the shortest edge is already >= *min_size* the image is returned
    untouched at its original (possibly larger) resolution.  Unlike
    :class:`ResizeIfSmaller`, this never performs interpolation — padding
    uses mirror reflection of edge pixels, preserving pixel-level artifacts.
    """

    def __init__(self, min_size: int):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) >= self.min_size:
            return img
        pad_w = max(0, self.min_size - w)
        pad_h = max(0, self.min_size - h)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        return T.functional.pad(
            img, (pad_left, pad_top, pad_right, pad_bottom),
            padding_mode="reflect",
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(min_size={self.min_size})"


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


def _genai_geometric(image_size: int, crop_p: float = 0.5) -> list:
    """Geometric + color augmentations for genai pipelines.

    Uses ``RandomResizeOrCrop`` instead of ``RandomResizedCrop`` to
    preserve pixel-level artifacts that are critical for GenAI detection.
    """
    return [
        RandomResizeOrCrop(image_size, crop_p=crop_p, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]


def _augly_artifact_pool() -> list:
    """Build a pool of artifact transforms for AugLy-hybrid N-of-K composition.

    Each transform targets a distinct degradation type observed in
    test-set augmentations.  Parameters are deliberately wider than the
    ``genai`` pipeline to cover aggressive test-time perturbations.
    Individual ``p`` values are set to 1.0 because selection probability
    is controlled by :class:`RandomNOfCompose`.
    """
    return [
        RandomJPEGCompression(quality_range=(20, 95), p=1.0),
        RandomDownscaleUpscale(scale_range=(0.3, 0.9), p=1.0),
        RandomGaussianNoise(std_range=(1.0, 15.0), p=1.0),
        RandomSaltPepperNoise(amount=0.05, p=1.0),
        RandomImpulseNoise(amount=0.05, p=1.0),
        RandomMedianBlur(kernel_sizes=(3, 5, 7), p=1.0),
        RandomBoxBlur(radius_range=(1, 3), p=1.0),
        RandomSharpen(factor_range=(1.0, 3.0), p=1.0),
        RandomPixelization(ratio_range=(0.2, 0.8), p=1.0),
        RandomPNGReencode(p=1.0),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
    ]


def _robust_artifact_groups() -> dict:
    """Build grouped artifact transforms for robust N-of-K composition.

    Each group represents a distinct degradation category.  The grouped
    composition operator selects N groups and picks one transform per
    group, ensuring diverse degradation coverage without same-category
    duplicates.

    All transforms use ``p=1.0`` because selection probability is
    controlled by the composition operator.
    """
    return {
        "blur": [
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
            RandomLensBlur(radius_range=(1, 6), p=1.0),
            RandomMotionBlur(kernel_size_range=(3, 15), p=1.0),
            RandomMedianBlur(kernel_sizes=(3, 5, 7), p=1.0),
            RandomBoxBlur(radius_range=(1, 3), p=1.0),
        ],
        "compression": [
            RandomJPEGCompression(quality_range=(20, 95), p=1.0),
            RandomWebPCompression(quality_range=(20, 95), p=1.0),
            RandomPNGReencode(p=1.0),
        ],
        "noise": [
            RandomGaussianNoise(std_range=(1.0, 15.0), p=1.0),
            RandomSaltPepperNoise(amount=0.05, p=1.0),
            RandomImpulseNoise(amount=0.05, p=1.0),
        ],
        "resize": [
            RandomDownscaleUpscale(scale_range=(0.3, 0.9), p=1.0),
            RandomPixelization(ratio_range=(0.2, 0.8), p=1.0),
        ],
        "color": [
            RandomColorQuantization(levels_range=(7, 20), p=1.0),
            RandomGammaCorrection(gamma_range=(0.5, 2.0), p=1.0),
            T.RandomGrayscale(p=1.0),
        ],
        "spatial": [
            RandomSpatialJitter(amount_range=(0.05, 0.5), p=1.0),
            T.RandomPerspective(distortion_scale=0.3, p=1.0),
            T.RandomRotation(degrees=15),
        ],
        "sharpness_brightness": [
            RandomSharpen(factor_range=(1.0, 3.0), p=1.0),
            RandomContrastCurve(amount_range=(-0.4, 0.3), p=1.0),
            RandomBrightnessCurve(amount_range=(-0.4, 0.5), p=1.0),
        ],
    }


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
            ``"genai"``, ``"genai_curriculum"``, ``"augly"``,
            ``"augly_curriculum"``, ``"robust"``, ``"robust_curriculum"``.
        total_epochs: Total training epochs (used by curricular variants).
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop (used by curricular variants).
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
            _genai_geometric(image_size)
            + [
                RandomJPEGCompression(quality_range=(30, 95), p=0.5),
                RandomDownscaleUpscale(scale_range=(0.5, 0.9), p=0.3),
                RandomGaussianNoise(std_range=(1.0, 10.0), p=0.3),
                RandomSaltPepperNoise(amount=0.05, p=0.05),
                RandomImpulseNoise(amount=0.05, p=0.03),
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
        sp_curricular = CurricularWrapper(
            transforms=[
                RandomSaltPepperNoise(amount=0.05, p=0.05),
                RandomImpulseNoise(amount=0.05, p=0.03),
            ],
            epoch_state=epoch_state,
            total_epochs=total_epochs,
            min_scale=0.02,  # 0.05 * 0.02 = 0.001 = 0.1%
        )
        return T.Compose(
            _genai_geometric(image_size)
            + [curricular, sp_curricular]
            + _to_tensor_normalize()
        )
    elif augmentation == "augly":
        return T.Compose(
            _genai_geometric(image_size)
            + [RandomNOfCompose(_augly_artifact_pool(), n=5)]
            + _to_tensor_normalize()
        )
    elif augmentation == "augly_curriculum":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        return T.Compose(
            _genai_geometric(image_size)
            + [
                CurricularNOfCompose(
                    _augly_artifact_pool(),
                    epoch_state=epoch_state,
                    total_epochs=total_epochs,
                    n_max=5,
                    n_min=1,
                ),
            ]
            + _to_tensor_normalize()
        )
    elif augmentation == "robust":
        return T.Compose(
            _genai_geometric(image_size)
            + [GroupedNOfCompose(_robust_artifact_groups(), n=4)]
            + _to_tensor_normalize()
        )
    elif augmentation == "robust_curriculum":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        return T.Compose(
            _genai_geometric(image_size)
            + [
                CurricularGroupedNOfCompose(
                    _robust_artifact_groups(),
                    epoch_state=epoch_state,
                    total_epochs=total_epochs,
                    n_max=5,
                    n_min=2,
                ),
            ]
            + _to_tensor_normalize()
        )
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def get_val_transform(
    image_size: int = 224,
    resize_size: int = 256,
) -> Callable:
    """Build validation/test transform pipeline."""
    # Ensure resize_size >= image_size to prevent zero-padded CenterCrop.
    resize_size = max(resize_size, image_size)
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_prep_transform(
    min_prep_size: int = 512,
) -> Callable:
    """Build a TTA preparation transform that preserves pixel artifacts.

    Images smaller than *min_prep_size* are reflect-padded (no
    interpolation).  Images already >= *min_prep_size* are kept at
    their native resolution — no CenterCrop is applied, so edge
    content is preserved for corner-crop TTA views.

    Output tensor size varies by image (``max(original, min_prep_size)``).
    Use :func:`_tta_collate_fn` in :mod:`data` for batching.

    Args:
        min_prep_size: Minimum spatial size.  Smaller images are
            reflect-padded to this size.
    """
    return T.Compose([
        ReflectPadIfSmaller(min_prep_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
