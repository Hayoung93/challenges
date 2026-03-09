from typing import Callable

import torchvision.transforms as T

from .genai_transforms import (
    CurricularGroupedNOfCompose,
    CurricularNOfCompose,
    CurricularWrapper,
    GroupedNOfCompose,
    IntensityGaussianBlur,
    IntensityRandomPerspective,
    SkipIfClean,
    RandomBoxBlur,
    RandomDCTBasisOverlay,
    RandomMoire,
    RandomBrightnessCurve,
    RandomChromaNoise,
    RandomColorQuantization,
    RandomContrastCurve,
    RandomDownscaleUpscale,
    RandomGammaCorrection,
    RandomGaussianNoise,
    RandomImpulseNoise,
    RandomJPEGCompression,
    RandomLensBlur,
    RandomLuminanceNoise,
    RandomMedianBlur,
    RandomMotionBlur,
    RandomNOfCompose,
    RandomPixelization,
    RandomPNGReencode,
    RandomPosterize,
    RandomResizeOrCrop,
    ResizeOrCropWithSmallPad,
    RandomAVIFCompression,
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


class MultiscaleTransformWrapper:
    """Wrap a transform factory to read target size from shared state.

    Each call reads ``scale_state.value`` to determine the current
    target resolution, then delegates to the appropriate transform.
    Transform instances are cached per resolution to avoid repeated
    construction.

    Args:
        build_fn: Callable ``(image_size) -> transform`` that builds a
            complete transform pipeline for a given resolution.
        scale_state: ``multiprocessing.Value('i', ...)`` holding the
            current target resolution.
        default_size: Fallback size when ``scale_state`` is not set.
    """

    def __init__(self, build_fn, scale_state, default_size: int = 224):
        self.build_fn = build_fn
        self.scale_state = scale_state
        self.default_size = default_size
        self._cache = {}

    def __call__(self, img):
        size = self.scale_state.value if self.scale_state is not None else self.default_size
        if size not in self._cache:
            self._cache[size] = self.build_fn(size)
        return self._cache[size](img)

    def __repr__(self):
        cached = sorted(self._cache.keys())
        return (
            f"{self.__class__.__name__}("
            f"default_size={self.default_size}, "
            f"cached_sizes={cached})"
        )


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


def _genai_geometric(image_size: int, crop_p: float = 0.5,
                     small_pad_p: float = 0.0,
                     small_crop_range: tuple = (48, 192)) -> list:
    """Geometric + color augmentations for genai pipelines.

    Uses ``RandomResizeOrCrop`` instead of ``RandomResizedCrop`` to
    preserve pixel-level artifacts that are critical for GenAI detection.

    When ``small_pad_p > 0``, uses :class:`ResizeOrCropWithSmallPad`
    to randomly simulate very small test images that are reflect-padded
    to ``image_size``.
    """
    if small_pad_p > 0:
        first_transform = ResizeOrCropWithSmallPad(
            image_size, crop_p=crop_p, scale=(0.5, 1.0),
            small_pad_p=small_pad_p, small_crop_range=small_crop_range,
        )
    else:
        first_transform = RandomResizeOrCrop(
            image_size, crop_p=crop_p, scale=(0.5, 1.0),
        )
    return [
        first_transform,
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
        RandomSaltPepperNoise(amount_range=(0.01, 0.08), p=1.0),
        RandomImpulseNoise(amount_range=(0.01, 0.08), p=1.0),
        RandomMedianBlur(kernel_sizes=(3, 5, 7), p=1.0),
        RandomBoxBlur(radius_range=(1, 3), p=1.0),
        RandomSharpen(factor_range=(1.0, 3.0), p=1.0),
        RandomPixelization(ratio_range=(0.2, 0.8), p=1.0),
        RandomPNGReencode(p=1.0),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
    ]


def _robust_geometric(image_size: int, crop_p: float = 0.5,
                      small_pad_p: float = 0.0,
                      small_crop_range: tuple = (48, 192)) -> list:
    """Geometric augmentations for robust pipelines.

    Only spatial transforms (resize/crop, flip) that must always run to
    produce a correctly-sized output.  Colour perturbation, grayscale,
    and blur are intentionally omitted — they are covered by the
    artifact group composition, which honours the ``clean_p``
    pass-through gate and intensity curriculum scheduling.

    When ``small_pad_p > 0``, uses :class:`ResizeOrCropWithSmallPad`
    to randomly simulate very small test images that are reflect-padded
    to ``image_size``.

    Args:
        image_size: Target square output size.
        crop_p: Probability of pixel-preserving crop path.
        small_pad_p: Probability of small-crop+reflect-pad path
            (0.0 = disabled).
        small_crop_range: ``(min, max)`` pixel range for small crops.
    """
    if small_pad_p > 0:
        first_transform = ResizeOrCropWithSmallPad(
            image_size, crop_p=crop_p, scale=(0.5, 1.0),
            small_pad_p=small_pad_p, small_crop_range=small_crop_range,
        )
    else:
        first_transform = RandomResizeOrCrop(
            image_size, crop_p=crop_p, scale=(0.5, 1.0),
        )
    return [
        first_transform,
        T.RandomHorizontalFlip(p=0.5),
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
            IntensityGaussianBlur(sigma=(0.1, 3.0)),
            RandomLensBlur(radius_range=(1, 6), p=1.0),
            RandomMotionBlur(kernel_size_range=(3, 15), p=1.0),
            RandomMedianBlur(kernel_sizes=(3, 5, 7), p=1.0),
            RandomBoxBlur(radius_range=(1, 3), p=1.0),
        ],
        "compression": [
            RandomJPEGCompression(quality_range=(20, 95), p=1.0),
            RandomWebPCompression(quality_range=(20, 95), p=1.0),
            RandomPNGReencode(p=1.0),
            RandomAVIFCompression(quality_range=(20, 95), p=1.0),
        ],
        "noise": [
            RandomGaussianNoise(std_range=(1.0, 15.0), p=1.0),
            RandomSaltPepperNoise(amount_range=(0.01, 0.08), p=1.0),
            RandomImpulseNoise(amount_range=(0.01, 0.08), p=1.0),
            RandomChromaNoise(std_range=(3.0, 20.0), p=1.0),
            RandomLuminanceNoise(std_range=(2.0, 15.0), p=1.0),
        ],
        "resize": [
            RandomDownscaleUpscale(scale_range=(0.3, 0.9), p=1.0),
            RandomPixelization(ratio_range=(0.2, 0.8), p=1.0),
        ],
        "color": [
            RandomColorQuantization(levels_range=(7, 20), p=1.0),
            RandomGammaCorrection(gamma_range=(0.5, 2.0), p=1.0),
            RandomPosterize(bits_range=(2, 6), p=1.0),
        ],
        "spatial": [
            RandomSpatialJitter(amount_range=(0.05, 0.5), p=1.0),
            IntensityRandomPerspective(distortion_scale=0.3),
        ],
        "sharpness_brightness": [
            RandomSharpen(factor_range=(1.0, 3.0), p=1.0),
            RandomContrastCurve(amount_range=(-0.4, 0.3), p=1.0),
            RandomBrightnessCurve(amount_range=(-0.4, 0.5), p=1.0),
        ],
    }


def _robust_artifact_groups_extended() -> dict:
    """Build grouped artifact transforms with extended intensity ranges.

    Compared to :func:`_robust_artifact_groups`, every group's maximum
    strength is raised to cover more extreme real-world degradations
    (aggressive social-media re-compression, heavy sensor noise,
    severe motion blur, etc.).  Intended for use with
    ``robust_curriculum_range`` where curriculum scheduling prevents
    the model from seeing only extreme augmentations early on.
    """
    return {
        "blur": [
            IntensityGaussianBlur(sigma=(0.1, 10.0)),
            RandomLensBlur(radius_range=(1, 9), p=1.0),
            RandomMotionBlur(kernel_size_range=(3, 21), p=1.0),
            RandomMedianBlur(kernel_sizes=(3, 5, 7, 9), p=1.0),
            RandomBoxBlur(radius_range=(1, 5), p=1.0),
        ],
        "compression": [
            RandomJPEGCompression(quality_range=(10, 95), p=1.0),
            RandomWebPCompression(quality_range=(10, 95), p=1.0),
            RandomPNGReencode(p=1.0),
            RandomAVIFCompression(quality_range=(10, 95), p=1.0),
        ],
        "noise": [
            RandomGaussianNoise(std_range=(1.0, 25.0), p=1.0),
            RandomSaltPepperNoise(amount_range=(0.01, 0.08), p=1.0),
            RandomImpulseNoise(amount_range=(0.01, 0.08), p=1.0),
            RandomChromaNoise(std_range=(3.0, 30.0), p=1.0),
            RandomLuminanceNoise(std_range=(2.0, 22.0), p=1.0),
        ],
        "resize": [
            RandomDownscaleUpscale(scale_range=(0.15, 0.9), p=1.0),
            RandomPixelization(ratio_range=(0.2, 0.8), p=1.0),
        ],
        "color": [
            RandomColorQuantization(levels_range=(7, 20), p=1.0),
            RandomGammaCorrection(gamma_range=(0.5, 2.0), p=1.0),
            RandomPosterize(bits_range=(2, 6), p=1.0),
        ],
        "spatial": [
            RandomSpatialJitter(amount_range=(0.05, 0.5), p=1.0),
            IntensityRandomPerspective(distortion_scale=0.2),
        ],
        "sharpness_brightness": [
            RandomSharpen(factor_range=(1.0, 5.0), p=1.0),
            RandomContrastCurve(amount_range=(-0.4, 0.3), p=1.0),
            RandomBrightnessCurve(amount_range=(-0.4, 0.5), p=1.0),
        ],
    }


# Group sampling weights for robust pipelines.
# Groups not listed default to 1.0 (uniform).
_ROBUST_GROUP_WEIGHTS = {
    "color": 0.25,
    "spatial": 0.5,
    "sharpness_brightness": 0.5,
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
    curriculum_ratio: float = 0.5,
    curriculum_n_min: int = 2,
    curriculum_n_max_start: int = 3,
    curriculum_n_max_end: int = 7,
    scale_state=None,
    clean_view: bool = False,
    small_pad_p: float = 0.0,
    small_crop_range: tuple = (48, 192),
) -> Callable:
    """Build training transform pipeline.

    Args:
        image_size: Target crop size.
        augmentation: One of ``"none"``, ``"default"``, ``"strong"``,
            ``"genai"``, ``"genai_curriculum"``, ``"augly"``,
            ``"augly_curriculum"``, ``"robust"``, ``"robust_curriculum"``,
            ``"robust_curriculum_range"``.
        total_epochs: Total training epochs (used by curricular variants).
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop (used by curricular variants).
        curriculum_ratio: Fraction of total epochs for curriculum to
            reach max strength (0.5 = halfway, 1.0 = original).
        curriculum_n_min: Fixed lower bound for group count sampling.
        curriculum_n_max_start: Upper bound of group count at epoch 0.
        curriculum_n_max_end: Upper bound of group count at curriculum
            completion.
        scale_state: ``multiprocessing.Value('i', ...)`` holding the
            current target resolution for multi-scale training.
            When provided, returns a :class:`MultiscaleTransformWrapper`.
        clean_view: When ``True``, returns the geometric-only variant
            of the requested augmentation (no artifact transforms).
            Used by multi-view training to provide a clean anchor view.
        small_pad_p: Probability of small-crop+reflect-pad augmentation
            (0.0 = disabled).  Simulates very small test images.
        small_crop_range: ``(min, max)`` pixel range for small crops.
    """
    # Multi-scale: wrap with dynamic resolution dispatch
    if scale_state is not None:
        def _build_for_size(sz):
            return get_train_transform(
                image_size=sz,
                augmentation=augmentation,
                total_epochs=total_epochs,
                epoch_state=epoch_state,
                curriculum_ratio=curriculum_ratio,
                curriculum_n_min=curriculum_n_min,
                curriculum_n_max_start=curriculum_n_max_start,
                curriculum_n_max_end=curriculum_n_max_end,
                scale_state=None,  # prevent recursion
                clean_view=clean_view,
                small_pad_p=small_pad_p,
                small_crop_range=small_crop_range,
            )
        return MultiscaleTransformWrapper(_build_for_size, scale_state, image_size)

    # Clean view: return geometric-only variant (no artifact transforms,
    # no color jitter).  Used by multi-view training to provide a stable
    # anchor view.  Color perturbation is learned through the augmented
    # view's CE loss; the anchor must stay colour-neutral so that the
    # multi-view consistency loss (symmetrised KL) produces clean gradients.
    if clean_view:
        _GEOMETRIC_MAP = {
            "robust": _robust_geometric,
            "robust_curriculum": _robust_geometric,
            "robust_curriculum_range": _robust_geometric,
            "genai": _genai_geometric,
            "genai_curriculum": _genai_geometric,
            "augly": _genai_geometric,
            "augly_curriculum": _genai_geometric,
        }
        geo_fn = _GEOMETRIC_MAP.get(augmentation)
        if geo_fn is not None:
            # Clean view: no small-pad simulation (anchor must be stable)
            geo_list = [
                t for t in geo_fn(image_size, small_pad_p=0.0)
                if not isinstance(t, T.ColorJitter)
            ]
            return T.Compose(geo_list + _to_tensor_normalize())
        # For other types (none, default, strong) fall through — they
        # are already artifact-free.

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
            _genai_geometric(image_size, small_pad_p=small_pad_p,
                             small_crop_range=small_crop_range)
            + [
                RandomJPEGCompression(quality_range=(30, 95), p=0.5),
                RandomDownscaleUpscale(scale_range=(0.5, 0.9), p=0.3),
                RandomGaussianNoise(std_range=(1.0, 10.0), p=0.3),
                RandomSaltPepperNoise(amount_range=(0.01, 0.08), p=0.05),
                RandomImpulseNoise(amount_range=(0.01, 0.08), p=0.03),
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
            curriculum_ratio=curriculum_ratio,
        )
        sp_curricular = CurricularWrapper(
            transforms=[
                RandomSaltPepperNoise(amount_range=(0.01, 0.08), p=0.05),
                RandomImpulseNoise(amount_range=(0.01, 0.08), p=0.03),
            ],
            epoch_state=epoch_state,
            total_epochs=total_epochs,
            min_scale=0.02,  # 0.05 * 0.02 = 0.001 = 0.1%
            curriculum_ratio=curriculum_ratio,
        )
        return T.Compose(
            _genai_geometric(image_size, small_pad_p=small_pad_p,
                             small_crop_range=small_crop_range)
            + [curricular, sp_curricular]
            + _to_tensor_normalize()
        )
    elif augmentation == "augly":
        return T.Compose(
            _genai_geometric(image_size, small_pad_p=small_pad_p,
                             small_crop_range=small_crop_range)
            + [RandomNOfCompose(_augly_artifact_pool(), n=5)]
            + _to_tensor_normalize()
        )
    elif augmentation == "augly_curriculum":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        return T.Compose(
            _genai_geometric(image_size, small_pad_p=small_pad_p,
                             small_crop_range=small_crop_range)
            + [
                CurricularNOfCompose(
                    _augly_artifact_pool(),
                    epoch_state=epoch_state,
                    total_epochs=total_epochs,
                    n_max=5,
                    n_min=1,
                    curriculum_ratio=curriculum_ratio,
                ),
            ]
            + _to_tensor_normalize()
        )
    elif augmentation == "robust":
        artifact_compose = GroupedNOfCompose(
            _robust_artifact_groups(), n=4,
            weights=_ROBUST_GROUP_WEIGHTS, clean_p=0.1,
        )
        return T.Compose(
            _robust_geometric(image_size, small_pad_p=small_pad_p,
                              small_crop_range=small_crop_range)
            + [artifact_compose]
            + [SkipIfClean(artifact_compose, RandomDCTBasisOverlay(p=0.05))]
            + [SkipIfClean(artifact_compose, RandomMoire(p=0.05))]
            + _to_tensor_normalize()
        )
    elif augmentation == "robust_curriculum":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        artifact_compose = CurricularGroupedNOfCompose(
            _robust_artifact_groups(),
            epoch_state=epoch_state,
            total_epochs=total_epochs,
            n_min=2,
            n_max_start=2,
            n_max_end=5,
            curriculum_ratio=curriculum_ratio,
            weights=_ROBUST_GROUP_WEIGHTS,
            clean_p_start=0.5,
            clean_p_end=0.1,
        )
        return T.Compose(
            _robust_geometric(image_size, small_pad_p=small_pad_p,
                              small_crop_range=small_crop_range)
            + [artifact_compose]
            + [SkipIfClean(artifact_compose, RandomDCTBasisOverlay(p=0.05))]
            + [SkipIfClean(artifact_compose, RandomMoire(p=0.05))]
            + _to_tensor_normalize()
        )
    elif augmentation == "robust_curriculum_range":
        if epoch_state is None:
            import multiprocessing
            epoch_state = multiprocessing.Value("i", 0)
        artifact_compose = CurricularGroupedNOfCompose(
            _robust_artifact_groups_extended(),
            epoch_state=epoch_state,
            total_epochs=total_epochs,
            n_min=curriculum_n_min,
            n_max_start=curriculum_n_max_start,
            n_max_end=curriculum_n_max_end,
            curriculum_ratio=curriculum_ratio,
            weights=_ROBUST_GROUP_WEIGHTS,
            clean_p_start=0.5,
            clean_p_end=0.15,
            intensity_curriculum=True,
        )
        return T.Compose(
            _robust_geometric(image_size,
                              small_pad_p=small_pad_p,
                              small_crop_range=small_crop_range)
            + [artifact_compose]
            + [SkipIfClean(artifact_compose, RandomDCTBasisOverlay(p=0.08))]
            + [SkipIfClean(artifact_compose, RandomMoire(p=0.08))]
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
