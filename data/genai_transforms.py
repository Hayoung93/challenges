"""GenAI-specific augmentation transforms for deepfake/AI-image detection.

These transforms target pixel-level artifacts (JPEG compression, resize
interpolation, sensor noise) that are critical signals for distinguishing
real images from AI-generated ones.  All transforms operate on PIL Images
and are compatible with ``torchvision.transforms.Compose``.
"""

import io
import math
import random

import cv2
import numpy as np
import scipy.ndimage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from scipy.interpolate import PchipInterpolator

try:
    import pillow_avif  # noqa: F401
    _AVIF_AVAILABLE = True
except ImportError:
    _AVIF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Intensity-scaling helpers for curriculum-aware transforms
# ---------------------------------------------------------------------------


def _iscale_upper(rng, intensity):
    """(a, b) where higher b = stronger effect.

    At intensity=0 → (a, a), at intensity=1 → (a, b).
    Example: std_range=(1.0, 25.0), intensity=0.5 → (1.0, 13.0)
    """
    a, b = rng
    return (a, a + (b - a) * intensity)


def _iscale_lower(rng, intensity):
    """(a, b) where lower a = stronger effect.

    At intensity=0 → (b, b), at intensity=1 → (a, b).
    Example: quality_range=(10, 95), intensity=0.5 → (52, 95)
    """
    a, b = rng
    return (b - (b - a) * intensity, b)


def _iscale_neutral(rng, intensity, neutral):
    """(a, b) symmetric around neutral point.

    At intensity=0 → (neutral, neutral), at intensity=1 → (a, b).
    Example: gamma_range=(0.3, 3.0), neutral=1.0, intensity=0.5
             → (0.65, 2.0)
    """
    a, b = rng
    return (neutral - (neutral - a) * intensity,
            neutral + (b - neutral) * intensity)


class RandomJPEGCompression:
    """Randomly compress image via JPEG at a random quality level.

    Simulates real-world image sharing where JPEG re-compression introduces
    blocking and ringing artifacts.

    Args:
        quality_range: ``(min_quality, max_quality)``, integers in 1–100.
        p: Probability of applying this transform.
    """

    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_lower(self.quality_range, self._intensity)
        quality = random.randint(int(round(lo)), int(round(hi)))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p})"
        )


_INTERPOLATION_METHODS = [
    Image.NEAREST,
    Image.BILINEAR,
    Image.BICUBIC,
    Image.LANCZOS,
]


class RandomDownscaleUpscale:
    """Downscale then upscale to introduce resize / interpolation artifacts.

    Simulates resolution changes from web sharing, screenshots, or
    thumbnail generation.  The round-trip destroys high-frequency detail
    and introduces characteristic aliasing patterns.  Interpolation
    methods are randomly selected per call to cover the diversity of
    resampling algorithms used across platforms.

    Args:
        scale_range: ``(min_scale, max_scale)`` relative to original size.
        p: Probability of applying this transform.
        interpolation_methods: List of PIL resampling filters to randomly
            choose from.  Defaults to NEAREST, BILINEAR, BICUBIC, LANCZOS.
    """

    def __init__(self, scale_range=(0.5, 0.9), p=0.3,
                 interpolation_methods=None):
        self.scale_range = scale_range
        self.p = p
        self._intensity = 1.0
        self.interpolation_methods = (
            interpolation_methods or _INTERPOLATION_METHODS
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        lo, hi = _iscale_lower(self.scale_range, self._intensity)
        scale = random.uniform(lo, hi)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        method_down = random.choice(self.interpolation_methods)
        method_up = random.choice(self.interpolation_methods)
        down = img.resize((new_w, new_h), method_down)
        up = down.resize((w, h), method_up)
        return up

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"scale_range={self.scale_range}, p={self.p})"
        )


class RandomGaussianNoise:
    """Add Gaussian noise to a PIL image.

    Forces the model to focus on structural artifacts rather than
    overfitting to noise-level differences between real and generated
    images.

    Args:
        std_range: ``(min_std, max_std)`` in pixel values [0, 255].
        p: Probability of applying this transform.
    """

    def __init__(self, std_range=(1.0, 10.0), p=0.3):
        self.std_range = std_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        lo, hi = _iscale_upper(self.std_range, self._intensity)
        std = random.uniform(lo, hi)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"std_range={self.std_range}, p={self.p})"
        )


class RandomSaltPepperNoise:
    """Add salt-and-pepper (impulse) noise to a PIL image.

    Randomly replaces a fraction of pixels with pure white (salt, 255)
    or pure black (pepper, 0).  The corruption amount is sampled
    uniformly from *amount_range* and scaled by ``_intensity``.

    Args:
        amount_range: ``(min_amount, max_amount)`` fraction of pixels
            to corrupt (0.0--1.0).
        p: Probability of applying this transform.
    """

    def __init__(self, amount_range=(0.01, 0.08), p=0.05):
        self.amount_range = amount_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        lo, hi = _iscale_upper(self.amount_range, self._intensity)
        amount = random.uniform(lo, hi)
        mask = np.random.random((h, w))
        arr[mask < amount / 2] = 255       # salt
        arr[mask > 1 - amount / 2] = 0     # pepper
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount_range={self.amount_range}, p={self.p})"
        )


class RandomImpulseNoise:
    """Add channel-independent impulse noise to a PIL image.

    Unlike salt-and-pepper noise which sets entire pixels to black or
    white, impulse noise corrupts each RGB channel independently,
    producing colourful speckles.  The corruption amount is sampled
    uniformly from *amount_range* and scaled by ``_intensity``.

    Args:
        amount_range: ``(min_amount, max_amount)`` fraction of channel
            values to corrupt (0.0--1.0).
        p: Probability of applying this transform.
    """

    def __init__(self, amount_range=(0.01, 0.08), p=0.03):
        self.amount_range = amount_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        lo, hi = _iscale_upper(self.amount_range, self._intensity)
        amount = random.uniform(lo, hi)
        mask = np.random.random(arr.shape)
        arr[mask < amount / 2] = 255
        arr[mask > 1 - amount / 2] = 0
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount_range={self.amount_range}, p={self.p})"
        )


class RandomPNGReencode:
    """Re-encode image through PNG format.

    While PNG is lossless, the encode/decode round-trip exercises a
    different code path and can expose subtle color-space handling
    differences.  This is a lightweight augmentation that mimics real
    image save/load cycles.

    Args:
        p: Probability of applying this transform.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomResizeOrCrop:
    """Randomly choose between resize-crop and direct crop (pixel-preserving).

    Path A (resize, probability ``1 - crop_p``):
        Delegates to ``torchvision.transforms.RandomResizedCrop``.  Crops a
        random region then resizes to ``(size, size)`` via interpolation.

    Path B (crop + pad, probability ``crop_p``):
        Takes a random ``(size, size)`` crop directly, preserving original
        pixel-level artifacts.  If the image is smaller than *size* in either
        dimension it is first padded with *padding_mode*.

    Args:
        size: Target square output size.
        crop_p: Probability of choosing the crop+pad path.
        scale: ``(min, max)`` area fraction for the resize path.
        padding_mode: Padding mode when the image is smaller than *size*.
    """

    def __init__(self, size, crop_p=0.5, scale=(0.5, 1.0),
                 padding_mode="reflect"):
        self.size = size
        self.crop_p = crop_p
        self.scale = scale
        self.padding_mode = padding_mode
        self._rrc = T.RandomResizedCrop(size, scale=scale)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.crop_p:
            return self._crop_pad(img)
        return self._rrc(img)

    def _crop_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.size or h < self.size:
            pad_w = max(self.size - w, 0)
            pad_h = max(self.size - h, 0)
            pad_left = pad_w // 2
            pad_top = pad_h // 2
            img = TF.pad(
                img,
                [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top],
                padding_mode=self.padding_mode,
            )
            w, h = img.size
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        return TF.crop(img, top, left, self.size, self.size)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, crop_p={self.crop_p}, "
            f"scale={self.scale}, padding_mode={self.padding_mode!r})"
        )


class RandomSmallCropReflectPad:
    """Simulate small test images by cropping a tiny region and reflect-padding.

    With probability ``p``, crops a random square region of a random small
    size from the input image, then symmetrically reflect-pads it to
    ``target_size``.  This simulates what happens at test time when very
    small images (48-192 px) are reflect-padded to the model's input size.

    The crop preserves original pixels (no interpolation), and the
    reflect padding mirrors edge pixels — matching the inference
    pipeline's :class:`ReflectPadIfSmaller` behaviour.

    Args:
        target_size: Output spatial size (typically ``image_size``).
        crop_range: ``(min_crop, max_crop)`` pixel range for the random
            small crop size.
        p: Probability of applying this transform.
    """

    def __init__(self, target_size: int = 224,
                 crop_range: tuple = (48, 192), p: float = 0.1):
        self.target_size = target_size
        self.crop_range = crop_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        min_dim = min(w, h)
        # Clamp crop range to image dimensions
        lo = min(self.crop_range[0], min_dim)
        hi = min(self.crop_range[1], min_dim)
        crop_size = random.randint(lo, max(lo, hi))

        # Random crop position
        left = random.randint(0, max(0, w - crop_size))
        top = random.randint(0, max(0, h - crop_size))
        cropped = TF.crop(img, top, left, crop_size, crop_size)

        # Reflect-pad to target_size.  PIL reflect padding requires
        # pad < image_dim, so we pad iteratively when the crop is very
        # small relative to target_size.
        return self._iterative_reflect_pad(cropped, self.target_size)

    @staticmethod
    def _iterative_reflect_pad(img: Image.Image,
                               target_size: int) -> Image.Image:
        """Reflect-pad *img* to ``(target_size, target_size)``.

        PIL's ``reflect`` padding mode requires the pad amount to be
        strictly less than the image dimension.  For very small crops
        (e.g. 48 px → 224 px) this is violated, so we apply padding in
        multiple rounds, each time padding by at most ``dim - 1`` pixels.
        """
        w, h = img.size
        # 1px dimension: reflect-pad needs pad < dim, so pad=0 → infinite
        # loop.  Upscale to 2px with NEAREST (preserves pixel value) first.
        if w < 2 or h < 2:
            img = img.resize((max(w, 2), max(h, 2)), Image.NEAREST)
            w, h = img.size
        while w < target_size or h < target_size:
            pad_w = min(w - 1, max(0, target_size - w))
            pad_h = min(h - 1, max(0, target_size - h))
            pad_left = pad_w // 2
            pad_top = pad_h // 2
            img = TF.pad(
                img,
                [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top],
                padding_mode="reflect",
            )
            w, h = img.size

        # Final center-crop to exact target_size if slightly oversized
        # from rounding during iterative padding.
        if w > target_size or h > target_size:
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            img = TF.crop(img, top, left, target_size, target_size)

        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"target_size={self.target_size}, "
            f"crop_range={self.crop_range}, p={self.p})"
        )


class ResizeOrCropWithSmallPad:
    """Randomly choose between normal resize/crop and small-crop + reflect-pad.

    With probability ``small_pad_p``, delegates to
    :class:`RandomSmallCropReflectPad` which simulates very small test
    images.  Otherwise delegates to :class:`RandomResizeOrCrop`.

    When the small-pad path fires, the output is already at
    ``target_size``, so downstream transforms (flip, colour jitter, etc.)
    apply normally.

    Args:
        target_size: Output spatial size.
        crop_p: Probability of crop+pad path in ``RandomResizeOrCrop``.
        scale: Area fraction range for the resize path.
        small_pad_p: Probability of using the small-crop+reflect-pad path.
        small_crop_range: ``(min, max)`` pixel range for small crops.
        padding_mode: Padding mode for the normal crop path.
    """

    def __init__(self, target_size: int, crop_p: float = 0.5,
                 scale: tuple = (0.5, 1.0), small_pad_p: float = 0.1,
                 small_crop_range: tuple = (48, 192),
                 padding_mode: str = "reflect"):
        self.resize_or_crop = RandomResizeOrCrop(
            target_size, crop_p=crop_p, scale=scale,
            padding_mode=padding_mode,
        )
        self.small_pad = RandomSmallCropReflectPad(
            target_size=target_size,
            crop_range=small_crop_range,
            p=1.0,  # probability managed by this wrapper
        )
        self.target_size = target_size
        self.small_pad_p = small_pad_p
        self.small_crop_range = small_crop_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.small_pad_p:
            return self.small_pad(img)
        return self.resize_or_crop(img)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"target_size={self.target_size}, "
            f"small_pad_p={self.small_pad_p}, "
            f"small_crop_range={self.small_crop_range}, "
            f"inner={self.resize_or_crop})"
        )


class CurricularWrapper:
    """Wrap transforms and scale their probability by training progress.

    At epoch 0, all child-transform probabilities are scaled down by
    ``min_scale``; at ``total_epochs * curriculum_ratio`` they reach
    their full configured probability.  This implements a linear
    curriculum that starts with mild augmentation and gradually
    increases difficulty.

    The ``epoch_state`` must be a ``multiprocessing.Value('i', 0)``
    (shared-memory integer) so that persistent DataLoader workers can
    observe epoch updates made by the training loop.

    Args:
        transforms: Transform objects, each with a ``p`` attribute.
        epoch_state: ``multiprocessing.Value`` holding the current epoch.
        total_epochs: Total number of training epochs.
        min_scale: Probability scale factor at epoch 0.
        curriculum_ratio: Fraction of total epochs over which the
            curriculum ramps from ``min_scale`` to 1.0.  After that
            point the scale stays at 1.0.  Default ``0.5`` means the
            curriculum completes at the halfway point.
    """

    def __init__(self, transforms, epoch_state, total_epochs, min_scale=0.1,
                 curriculum_ratio=0.5):
        self.transforms = transforms
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.min_scale = min_scale
        self.curriculum_ratio = curriculum_ratio

    def _get_scale(self):
        if self.total_epochs <= 1:
            return 1.0
        curriculum_epochs = max(self.total_epochs * self.curriculum_ratio, 1)
        if curriculum_epochs <= 1:
            return 1.0
        progress = self.epoch_state.value / (curriculum_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_scale + (1.0 - self.min_scale) * progress

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = self._get_scale()
        for t in self.transforms:
            if random.random() < t.p * scale:
                original_p = t.p
                t.p = 1.0
                img = t(img)
                t.p = original_p
        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_transforms={len(self.transforms)}, "
            f"total_epochs={self.total_epochs}, "
            f"min_scale={self.min_scale}, "
            f"curriculum_ratio={self.curriculum_ratio})"
        )


# ---------------------------------------------------------------------------
# AugLy-compatible transforms & composition operators
# ---------------------------------------------------------------------------

class AugLyTransform:
    """Wrap an AugLy ``BaseTransform`` for ``torchvision.transforms.Compose``.

    AugLy transforms accept extra keyword arguments (``metadata``,
    ``bboxes``, etc.) that torchvision's ``Compose`` does not pass.
    This thin wrapper bridges the two APIs.

    Args:
        augly_cls: An AugLy transform **class** (not instance).
        p: Probability of applying the transform.
        **kwargs: Forwarded to ``augly_cls(p=1.0, **kwargs)``.
    """

    def __init__(self, augly_cls, p=1.0, **kwargs):
        # Instantiate with p=1.0; probability is handled by this wrapper.
        self.transform = augly_cls(p=1.0, **kwargs)
        self.p = p
        self._cls_name = augly_cls.__name__
        self._kwargs = kwargs

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return self.transform(img)

    def __repr__(self):
        kw = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"AugLyTransform({self._cls_name}, p={self.p}, {kw})"


class RandomMedianBlur:
    """Apply a median filter with a random kernel size.

    Median filtering is a non-linear smoothing technique that preserves
    edges better than Gaussian blur.  It is commonly used in test-set
    augmentation pipelines (e.g. AugLy, albumentations).

    Args:
        kernel_sizes: Odd-valued kernel sizes to sample from.
        p: Probability of applying this transform.
    """

    def __init__(self, kernel_sizes=(3, 5, 7), p=0.3):
        self.kernel_sizes = kernel_sizes
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        n = max(1, round(len(self.kernel_sizes) * self._intensity))
        effective_sizes = self.kernel_sizes[:n]
        k = random.choice(effective_sizes)
        arr = np.array(img)
        arr = cv2.medianBlur(arr, k)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"kernel_sizes={self.kernel_sizes}, p={self.p})"
        )


class RandomBoxBlur:
    """Apply a box (average) blur with a random radius.

    Box blur replaces each pixel with the unweighted average of its
    neighbours, producing a uniform smoothing effect distinct from
    Gaussian blur.

    Args:
        radius_range: ``(min_radius, max_radius)`` in pixels.
        p: Probability of applying this transform.
    """

    def __init__(self, radius_range=(1, 3), p=0.3):
        self.radius_range = radius_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.radius_range, self._intensity)
        r = random.randint(int(round(lo)), int(round(hi)))
        return img.filter(ImageFilter.BoxBlur(radius=r))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"radius_range={self.radius_range}, p={self.p})"
        )


class RandomSharpen:
    """Randomly adjust image sharpness.

    A factor of 1.0 leaves the image unchanged; values > 1.0 sharpen,
    values < 1.0 blur.  Sharpening can be applied after compression or
    blur to simulate post-processing commonly seen on social media.

    Args:
        factor_range: ``(min_factor, max_factor)``.
        p: Probability of applying this transform.
    """

    def __init__(self, factor_range=(1.0, 3.0), p=0.3):
        self.factor_range = factor_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.factor_range, self._intensity)
        factor = random.uniform(lo, hi)
        return ImageEnhance.Sharpness(img).enhance(factor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"factor_range={self.factor_range}, p={self.p})"
        )


class RandomPixelization:
    """Pixelate an image by down-scaling and up-scaling with nearest-neighbour.

    Unlike ``RandomDownscaleUpscale`` (which uses bilinear interpolation),
    pixelization uses nearest-neighbour, producing blocky artifacts typical
    of intentionally pixelated or low-resolution content.

    Args:
        ratio_range: ``(min_ratio, max_ratio)``; smaller values = more
            pixelated.  1.0 means no change.
        p: Probability of applying this transform.
    """

    def __init__(self, ratio_range=(0.2, 0.8), p=0.2):
        self.ratio_range = ratio_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        lo, hi = _iscale_lower(self.ratio_range, self._intensity)
        ratio = random.uniform(lo, hi)
        small_w, small_h = max(1, int(w * ratio)), max(1, int(h * ratio))
        small = img.resize((small_w, small_h), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ratio_range={self.ratio_range}, p={self.p})"
        )


class RandomNOfCompose:
    """Randomly select *n* transforms from a pool and apply them in random order.

    Designed to mimic test-set augmentation pipelines that apply a fixed
    number of randomly chosen transforms per image (e.g. "5 of K").

    When used inside ``torchvision.transforms.Compose``, each call
    samples *n* transforms (without replacement), shuffles their order,
    and applies them sequentially.  Individual transform ``p`` values are
    **ignored** (all selected transforms are force-applied).

    Args:
        transforms: Pool of candidate transforms.
        n: Number of transforms to select per call.
    """

    def __init__(self, transforms, n=5):
        self.transforms = transforms
        self.n = n

    def __call__(self, img: Image.Image) -> Image.Image:
        k = min(self.n, len(self.transforms))
        selected = random.sample(self.transforms, k)
        for t in selected:
            original_p = getattr(t, "p", None)
            if original_p is not None:
                t.p = 1.0
            img = t(img)
            if original_p is not None:
                t.p = original_p
        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n={self.n}, pool_size={len(self.transforms)})"
        )


class CurricularNOfCompose:
    """Curriculum-aware variant of :class:`RandomNOfCompose`.

    The number of transforms applied per image increases linearly from
    ``n_min`` at epoch 0 to ``n_max`` at
    ``total_epochs * curriculum_ratio``.

    Args:
        transforms: Pool of candidate transforms.
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop.
        total_epochs: Total number of training epochs.
        n_max: Maximum number of transforms at full curriculum.
        n_min: Starting number of transforms at epoch 0.
        curriculum_ratio: Fraction of total epochs over which the
            curriculum ramps from ``n_min`` to ``n_max``.
    """

    def __init__(self, transforms, epoch_state, total_epochs,
                 n_max=5, n_min=1, curriculum_ratio=0.5):
        self.transforms = transforms
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.n_max = n_max
        self.n_min = n_min
        self.curriculum_ratio = curriculum_ratio

    def _get_n(self):
        if self.total_epochs <= 1:
            return self.n_max
        curriculum_epochs = max(self.total_epochs * self.curriculum_ratio, 1)
        if curriculum_epochs <= 1:
            return self.n_max
        progress = self.epoch_state.value / (curriculum_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return max(self.n_min, round(self.n_min + (self.n_max - self.n_min) * progress))

    def __call__(self, img: Image.Image) -> Image.Image:
        n = self._get_n()
        k = min(n, len(self.transforms))
        selected = random.sample(self.transforms, k)
        for t in selected:
            original_p = getattr(t, "p", None)
            if original_p is not None:
                t.p = 1.0
            img = t(img)
            if original_p is not None:
                t.p = original_p
        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_min={self.n_min}, n_max={self.n_max}, "
            f"pool_size={len(self.transforms)}, "
            f"total_epochs={self.total_epochs}, "
            f"curriculum_ratio={self.curriculum_ratio})"
        )


# ---------------------------------------------------------------------------
# Robust augmentation transforms (reimplementations from aug_utils_train
# + novel transforms for generalization against unknown distortions)
# ---------------------------------------------------------------------------


class RandomLensBlur:
    """Apply circular (disk) blur simulating camera defocus.

    Unlike Gaussian blur, a disk kernel has a flat frequency response
    within a radius, producing qualitatively different bokeh-like
    artifacts.  Reimplements ``lens_blur`` from ``aug_utils_train``.

    Args:
        radius_range: ``(min_radius, max_radius)`` in pixels.
        p: Probability of applying this transform.
    """

    def __init__(self, radius_range=(1, 6), p=0.3):
        self.radius_range = radius_range
        self.p = p
        self._intensity = 1.0

    def _make_disk_kernel(self, radius):
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = (x * x + y * y) <= radius * radius
        kernel = mask.astype(np.float64)
        kernel /= kernel.sum()
        return kernel

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.radius_range, self._intensity)
        radius = random.randint(int(round(lo)), int(round(hi)))
        kernel = self._make_disk_kernel(radius).astype(np.float32)
        arr = np.array(img)
        arr = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"radius_range={self.radius_range}, p={self.p})"
        )


class RandomColorQuantization:
    """Reduce the number of distinct intensity levels (posterize).

    Simulates GIF conversion, poster effects, or heavy color-space
    reduction seen in low-bandwidth image sharing.  Reimplements
    ``quantization`` from ``aug_utils_train``.

    Args:
        levels_range: ``(min_levels, max_levels)`` of output intensity
            levels.  Fewer levels = more banding.
        p: Probability of applying this transform.
    """

    def __init__(self, levels_range=(7, 20), p=0.3):
        self.levels_range = levels_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_lower(self.levels_range, self._intensity)
        levels = random.randint(int(round(lo)), int(round(hi)))
        arr = np.array(img, dtype=np.float64)
        bins = np.linspace(0, 255, levels + 1)
        indices = np.digitize(arr, bins[1:-1])  # 0..levels-1
        arr = (indices * (255.0 / (levels - 1))).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"levels_range={self.levels_range}, p={self.p})"
        )


class RandomSpatialJitter:
    """Randomly displace pixels to simulate vibration or scatter.

    Each pixel is shifted by a random sub-pixel offset and resampled
    via bilinear interpolation.  Reimplements ``jitter`` (``imscatter``)
    from ``aug_utils_train``.

    Args:
        amount_range: ``(min_amount, max_amount)`` standard deviation of
            the random displacement (in pixels).
        iterations: Number of scatter passes.
        p: Probability of applying this transform.
    """

    def __init__(self, amount_range=(0.05, 0.5), iterations=1, p=0.2):
        self.amount_range = amount_range
        self.iterations = iterations
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.amount_range, self._intensity)
        amount = random.uniform(lo, hi)
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape[:2]
        for _ in range(self.iterations):
            dy = np.random.randn(h, w) * amount
            dx = np.random.randn(h, w) * amount
            yy, xx = np.mgrid[0:h, 0:w]
            new_y = (yy + dy).astype(np.float64)
            new_x = (xx + dx).astype(np.float64)
            for c in range(arr.shape[2]):
                arr[:, :, c] = scipy.ndimage.map_coordinates(
                    arr[:, :, c], [new_y, new_x],
                    order=1, mode="reflect",
                )
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount_range={self.amount_range}, "
            f"iterations={self.iterations}, p={self.p})"
        )


class RandomContrastCurve:
    """Adjust contrast via a spline-based tone curve.

    The curve passes through ``(0, 0)``, three control points, and
    ``(1, 1)``.  Positive *amount* increases contrast; negative
    decreases it.  Reimplements ``linear_contrast_change`` from
    ``aug_utils_train``.

    Args:
        amount_range: ``(min_amount, max_amount)``.
        p: Probability of applying this transform.
    """

    def __init__(self, amount_range=(-0.4, 0.3), p=0.3):
        self.amount_range = amount_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_neutral(self.amount_range, self._intensity, neutral=0.0)
        amount = random.uniform(lo, hi)
        x_pts = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        y_pts = np.array([
            0.0,
            0.25 - amount / 4,
            0.5,
            0.75 + amount / 4,
            1.0,
        ])
        spline = PchipInterpolator(x_pts, y_pts)
        xs = np.linspace(0, 1, 256)
        lut = np.clip(spline(xs) * 255, 0, 255).astype(np.uint8).tolist()
        return img.point(lut * 3)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount_range={self.amount_range}, p={self.p})"
        )


class RandomBrightnessCurve:
    """Adjust brightness via spline curves in RGB + LAB space.

    Positive *amount* brightens; negative darkens.  Combines an RGB
    tone-curve with a LAB lightness adjustment, blending at a 2:1
    ratio to match the behaviour of ``brighten``/``darken`` in
    ``aug_utils_train``.

    Args:
        amount_range: ``(min_amount, max_amount)``.
        p: Probability of applying this transform.
    """

    def __init__(self, amount_range=(-0.4, 0.5), p=0.3):
        self.amount_range = amount_range
        self.p = p
        self._intensity = 1.0

    @staticmethod
    def _build_curve_lut(coef):
        """Build a 256-entry LUT from a single-midpoint spline."""
        x_pts = np.array([0.0, 0.5, 1.0])
        y_pts = np.array([0.0, coef, 1.0])
        spline = PchipInterpolator(x_pts, y_pts)
        xs = np.linspace(0, 1, 256)
        return np.clip(spline(xs) * 255, 0, 255).astype(np.uint8)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_neutral(self.amount_range, self._intensity, neutral=0.0)
        amount = random.uniform(lo, hi)
        if amount >= 0:
            coef = 0.5 + amount / 2
        else:
            coef = 0.5 - abs(amount) / 2

        # RGB curve component
        lut = self._build_curve_lut(coef).tolist()
        rgb_curved = img.point(lut * 3)

        # LAB lightness component
        arr = np.array(img, dtype=np.float64) / 255.0
        lut_arr = np.array(lut, dtype=np.float64) / 255.0
        xs = np.linspace(0, 1, 256)
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        gray_idx = np.clip((gray * 255).astype(int), 0, 255)
        l_adjusted = lut_arr[gray_idx]
        safe_gray = np.where(gray > 1e-6, gray, 1.0)
        scale = np.where(gray > 1e-6, l_adjusted / safe_gray, 1.0)
        lab_arr = np.clip(arr * scale[:, :, np.newaxis] * 255, 0, 255).astype(np.uint8)
        lab_adjusted = Image.fromarray(lab_arr)

        # Blend (2 * rgb_curve + 1 * lab_adjusted) / 3
        return Image.blend(rgb_curved, lab_adjusted, alpha=1.0 / 3.0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount_range={self.amount_range}, p={self.p})"
        )


class RandomWebPCompression:
    """Randomly compress image via WebP at a random quality level.

    WebP uses 4x4 transform blocks (vs JPEG's 8x8 DCT blocks),
    producing different compression artifacts.  Training on both JPEG
    and WebP helps the model generalize to unseen codecs.

    Args:
        quality_range: ``(min_quality, max_quality)``, integers in 1--100.
        p: Probability of applying this transform.
    """

    def __init__(self, quality_range=(20, 95), p=0.3):
        self.quality_range = quality_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_lower(self.quality_range, self._intensity)
        quality = random.randint(int(round(lo)), int(round(hi)))
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p})"
        )


class RandomAVIFCompression:
    """Randomly compress image via AVIF at a random quality level.

    AVIF uses AV1 intra-frame coding, producing different compression
    artifacts from JPEG (8x8 DCT) and WebP (4x4 transform).  Training
    on AVIF helps the model generalize as modern browsers and operating
    systems increasingly adopt AVIF as a default format.

    Requires ``pillow-avif-plugin``.  When the plugin is unavailable
    the transform passes through without modification.

    Args:
        quality_range: ``(min_quality, max_quality)``, integers in 1--100.
        p: Probability of applying this transform.
    """

    def __init__(self, quality_range=(10, 95), p=0.3):
        self.quality_range = quality_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p or not _AVIF_AVAILABLE:
            return img
        lo, hi = _iscale_lower(self.quality_range, self._intensity)
        quality = random.randint(int(round(lo)), int(round(hi)))
        buffer = io.BytesIO()
        img.save(buffer, format="AVIF", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p}, "
            f"avif_available={_AVIF_AVAILABLE})"
        )


class RandomMotionBlur:
    """Apply directional motion blur at a random angle.

    Simulates camera or subject motion, producing directional smearing
    that is qualitatively different from isotropic blur types (Gaussian,
    lens, box, median).

    Args:
        kernel_size_range: ``(min_size, max_size)`` odd kernel sizes.
        p: Probability of applying this transform.
    """

    def __init__(self, kernel_size_range=(3, 15), p=0.3):
        self.kernel_size_range = kernel_size_range
        self.p = p
        self._intensity = 1.0

    def _make_motion_kernel(self, size, angle):
        kernel = np.zeros((size, size), dtype=np.float64)
        kernel[size // 2, :] = 1.0
        kernel = scipy.ndimage.rotate(
            kernel, angle, reshape=False, order=1, mode="constant",
        )
        kernel = np.clip(kernel, 0, None)
        total = kernel.sum()
        if total > 0:
            kernel /= total
        return kernel

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.kernel_size_range, self._intensity)
        min_s, max_s = int(round(lo)), int(round(hi))
        # Ensure odd kernel size
        sizes = list(range(min_s | 1, max_s + 1, 2))
        if not sizes:
            sizes = [min_s | 1]
        size = random.choice(sizes)
        angle = random.uniform(0, 360)
        kernel = self._make_motion_kernel(size, angle).astype(np.float32)
        arr = np.array(img)
        arr = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"kernel_size_range={self.kernel_size_range}, p={self.p})"
        )


class RandomGammaCorrection:
    """Apply non-linear gamma correction ``I' = I^gamma``.

    Gamma < 1 brightens (compresses highlights, expands shadows);
    gamma > 1 darkens (expands highlights, compresses shadows).
    Covers a different region of tone-curve space than spline-based
    contrast/brightness curves.

    Args:
        gamma_range: ``(min_gamma, max_gamma)``.
        p: Probability of applying this transform.
    """

    def __init__(self, gamma_range=(0.5, 2.0), p=0.3):
        self.gamma_range = gamma_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        lo, hi = _iscale_neutral(self.gamma_range, self._intensity, neutral=1.0)
        gamma = random.uniform(lo, hi)
        lut = [int(((i / 255.0) ** gamma) * 255) for i in range(256)]
        return img.point(lut * 3)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"gamma_range={self.gamma_range}, p={self.p})"
        )


class RandomPosterize:
    """Reduce bit-depth per channel via ``PIL.ImageOps.posterize``.

    Unlike :class:`RandomColorQuantization` which bins continuous values
    into evenly-spaced levels, this zeroes out the least-significant bits
    of each channel, producing hard-edge banding patterns typical of
    social-media re-encoding and format conversion.

    Args:
        bits_range: ``(min_bits, max_bits)`` to keep, 1–7.
            Lower = more aggressive banding.
        p: Probability of applying this transform.
    """

    def __init__(self, bits_range=(2, 6), p=0.3):
        self.bits_range = bits_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        from PIL import ImageOps
        lo, hi = _iscale_lower(self.bits_range, self._intensity)
        bits = random.randint(int(round(lo)), int(round(hi)))
        return ImageOps.posterize(img, bits)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"bits_range={self.bits_range}, p={self.p})"
        )


class RandomChromaNoise:
    """Add Gaussian noise only to chrominance (Cb, Cr) channels.

    Camera sensors and lossy codecs (JPEG, WebP) introduce significantly
    more noise in chroma than in luma.  This transform reproduces that
    pattern by converting to YCbCr, perturbing Cb/Cr, and converting
    back, leaving luminance untouched.

    Args:
        std_range: ``(min_std, max_std)`` of Gaussian noise in [0, 255]
            scale applied to Cb and Cr channels.
        p: Probability of applying this transform.
    """

    def __init__(self, std_range=(3.0, 20.0), p=0.3):
        self.std_range = std_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ycbcr = img.convert("YCbCr")
        arr = np.array(ycbcr, dtype=np.float32)
        lo, hi = _iscale_upper(self.std_range, self._intensity)
        std = random.uniform(lo, hi)
        noise = np.random.normal(0, std, arr[:, :, 1:].shape).astype(np.float32)
        arr[:, :, 1:] += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "YCbCr").convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"std_range={self.std_range}, p={self.p})"
        )


class RandomLuminanceNoise:
    """Add Gaussian noise only to the luminance (Y) channel.

    Simulates film grain and sensor read noise that primarily affects
    brightness while preserving color fidelity.  Operates in YCbCr
    space, perturbing only Y.

    Args:
        std_range: ``(min_std, max_std)`` of Gaussian noise in [0, 255]
            scale applied to the Y channel.
        p: Probability of applying this transform.
    """

    def __init__(self, std_range=(2.0, 15.0), p=0.3):
        self.std_range = std_range
        self.p = p
        self._intensity = 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ycbcr = img.convert("YCbCr")
        arr = np.array(ycbcr, dtype=np.float32)
        lo, hi = _iscale_upper(self.std_range, self._intensity)
        std = random.uniform(lo, hi)
        noise = np.random.normal(0, std, arr[:, :, 0].shape).astype(np.float32)
        arr[:, :, 0] += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "YCbCr").convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"std_range={self.std_range}, p={self.p})"
        )


class RandomBlockDistortion:
    """Apply grid-pattern block distortion observed in challenge test sets.

    Divides the image into a grid of cells and randomly replaces a
    subset of cells with one of: solid fill (average colour), stripe
    pattern, or heavy noise.  Sharp cell boundaries produce the
    characteristic grid-artifact signature.

    This transform is designed for low-frequency application (p ~ 0.05)
    to match the rare occurrence in real test data.  It should be placed
    **outside** ``GroupedNOfCompose`` as an independent pipeline step.

    Args:
        cell_size_range: ``(min, max)`` cell size in pixels.
        affected_ratio_range: ``(min, max)`` fraction of cells to distort.
        prob_solid: Fraction of affected cells filled with average colour.
        prob_striped: Fraction of affected cells filled with stripe pattern.
            Remaining affected cells receive heavy Gaussian noise.
        noise_std: Standard deviation for noise cells.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        cell_size_range=(16, 48),
        affected_ratio_range=(0.1, 0.5),
        prob_solid=0.4,
        prob_striped=0.4,
        noise_std=40.0,
        p=0.05,
    ):
        self.cell_size_range = cell_size_range
        self.affected_ratio_range = affected_ratio_range
        self.prob_solid = prob_solid
        self.prob_striped = prob_striped
        self.noise_std = noise_std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        cell_size = random.randint(*self.cell_size_range)
        affected_ratio = random.uniform(*self.affected_ratio_range)

        arr = np.array(img, dtype=np.float32)
        n_rows = math.ceil(h / cell_size)
        n_cols = math.ceil(w / cell_size)
        n_cells = n_rows * n_cols
        n_affected = max(1, round(n_cells * affected_ratio))

        affected_indices = set(random.sample(range(n_cells), min(n_affected, n_cells)))

        solid_thresh = self.prob_solid
        stripe_thresh = self.prob_solid + self.prob_striped

        for idx in affected_indices:
            r, c = divmod(idx, n_cols)
            y0 = r * cell_size
            x0 = c * cell_size
            y1 = min(y0 + cell_size, h)
            x1 = min(x0 + cell_size, w)
            patch = arr[y0:y1, x0:x1]

            roll = random.random()
            if roll < solid_thresh:
                arr[y0:y1, x0:x1] = patch.mean(axis=(0, 1))
            elif roll < stripe_thresh:
                ph, pw = y1 - y0, x1 - x0
                freq = random.randint(2, 6)
                if random.random() < 0.5:
                    pattern = (np.sin(np.linspace(0, freq * np.pi, ph)) > 0
                               ).astype(np.float32).reshape(-1, 1, 1)
                else:
                    pattern = (np.sin(np.linspace(0, freq * np.pi, pw)) > 0
                               ).astype(np.float32).reshape(1, -1, 1)
                base = patch.mean(axis=(0, 1))
                stripe = np.random.uniform(0, 256, 3).astype(np.float32)
                arr[y0:y1, x0:x1] = base * pattern + stripe * (1 - pattern)
            else:
                noise = np.random.normal(0, self.noise_std, patch.shape
                                         ).astype(np.float32)
                arr[y0:y1, x0:x1] = patch + noise

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"cell_size_range={self.cell_size_range}, "
            f"affected_ratio_range={self.affected_ratio_range}, "
            f"p={self.p})"
        )


class RandomDCTBasisOverlay:
    """Overlay JPEG-like DCT basis patterns onto 8x8 image blocks.

    Simulates the characteristic DCT basis ringing artifacts observed in
    test-set images that have undergone heavy JPEG compression.  Unlike
    :class:`RandomBlockDistortion` (which uses synthetic solid/stripe/noise
    fills), this transform adds **actual DCT-II basis functions** to a
    random subset of 8×8 blocks, producing visually authentic blocking
    artifacts with the correct frequency structure.

    Each affected block receives 1–``n_basis_max`` randomly chosen non-DC
    basis functions (from the 63 possible AC components), each scaled by
    a random coefficient drawn from ``[-strength, +strength]``.

    The 63 basis tiles are precomputed at ``__init__`` time via
    ``scipy.fft.idctn`` for efficiency.

    This transform supports the ``_intensity`` protocol used by
    :class:`CurricularGroupedNOfCompose` for curriculum scheduling.

    Args:
        strength_range: ``(min, max)`` peak additive amplitude per basis.
        n_basis_range: ``(min, max)`` number of basis functions overlaid
            per affected block.
        block_coverage: Fraction of 8×8 blocks that receive the overlay.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        strength_range=(5.0, 40.0),
        n_basis_range=(1, 3),
        block_coverage=0.3,
        p=0.05,
    ):
        self.strength_range = strength_range
        self.n_basis_range = n_basis_range
        self.block_coverage = block_coverage
        self.p = p
        self._intensity = 1.0

        # Precompute all 63 non-DC 8×8 DCT-II basis tiles.
        from scipy.fft import idctn

        self._basis_tiles = []
        self._basis_indices = []
        for u in range(8):
            for v in range(8):
                if u == 0 and v == 0:
                    continue  # skip DC component
                coeff = np.zeros((8, 8), dtype=np.float64)
                coeff[u, v] = 1.0
                tile = idctn(coeff, type=2, norm="ortho").astype(np.float32)
                self._basis_tiles.append(tile)
                self._basis_indices.append((u, v))
        # Stack into (63, 8, 8) array for efficient indexing.
        self._basis_tiles = np.stack(self._basis_tiles)  # (63, 8, 8)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        n_bx = w // 8
        n_by = h // 8
        if n_bx == 0 or n_by == 0:
            return img

        arr = np.array(img, dtype=np.float32)

        # Intensity-scaled strength
        lo, hi = _iscale_upper(self.strength_range, self._intensity)
        max_strength = random.uniform(lo, hi)

        # Select which blocks to affect
        total_blocks = n_bx * n_by
        n_affected = max(1, round(total_blocks * self.block_coverage))
        n_affected = min(n_affected, total_blocks)
        affected = random.sample(range(total_blocks), n_affected)

        n_basis_lo, n_basis_hi = self.n_basis_range
        n_tiles = len(self._basis_tiles)

        for idx in affected:
            by, bx = divmod(idx, n_bx)
            y0, x0 = by * 8, bx * 8

            n = random.randint(n_basis_lo, n_basis_hi)
            chosen = random.sample(range(n_tiles), min(n, n_tiles))

            for ti in chosen:
                tile = self._basis_tiles[ti]  # (8, 8)
                s = random.uniform(-max_strength, max_strength)
                # Add to all channels
                arr[y0:y0 + 8, x0:x0 + 8, :] += (tile * s)[:, :, np.newaxis]

        np.clip(arr, 0, 255, out=arr)
        return Image.fromarray(arr.astype(np.uint8))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"strength_range={self.strength_range}, "
            f"n_basis_range={self.n_basis_range}, "
            f"block_coverage={self.block_coverage}, "
            f"p={self.p})"
        )


class RandomMoire:
    """Apply moire-pattern augmentation via sine-wave synthesis or real pattern blending.

    When triggered (probability *p*), the transform selects **one** of two
    strategies controlled by ``sine_ratio``:

    * **Sine-wave mode** (default 70%): generates 2–4 overlapping sine waves
      at random frequencies and angles, producing a procedural interference
      pattern that is alpha-blended onto the image.
    * **Bank-blend mode** (default 30%): loads a random moire pattern from a
      pre-built image bank and composites it using one of three blend modes
      (multiply / alpha / overlay), each with equal probability.

    The two modes are mutually exclusive per call, preventing double-moire.

    This transform should be placed **outside** ``GroupedNOfCompose`` as an
    independent pipeline step, similar to ``RandomBlockDistortion``.

    Args:
        bank_dir: Path to the moire pattern bank directory containing
            WebP images.  If ``None`` or non-existent, falls back to
            100 % sine-wave mode.
        sine_ratio: Fraction of activations that use sine-wave mode
            (remainder uses bank-blend).
        num_waves_range: ``(min, max)`` number of sine waves to overlay.
        frequency_range: ``(min, max)`` spatial frequency in pixels per cycle.
        angle_range: ``(min, max)`` wave angle in degrees.
        sine_opacity_range: ``(min, max)`` alpha for sine-wave blending.
        bank_opacity_range: ``(min, max)`` alpha / strength for bank blending.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        bank_dir: str | None = "/data/data/uniMoire/moire_bank/",
        sine_ratio: float = 0.7,
        num_waves_range: tuple[int, int] = (2, 4),
        frequency_range: tuple[float, float] = (20.0, 120.0),
        angle_range: tuple[float, float] = (0, 180),
        sine_opacity_range: tuple[float, float] = (0.03, 0.15),
        bank_opacity_range: tuple[float, float] = (0.05, 0.25),
        p: float = 0.05,
    ):
        self.sine_ratio = sine_ratio
        self.num_waves_range = num_waves_range
        self.frequency_range = frequency_range
        self.angle_range = angle_range
        self.sine_opacity_range = sine_opacity_range
        self.bank_opacity_range = bank_opacity_range
        self.p = p
        self._intensity = 1.0

        # Load bank images as compressed bytes for memory efficiency.
        self._bank_bytes: list[bytes] = []
        self._bank_available = False
        if bank_dir is not None:
            import os
            import warnings

            bank_path = os.path.expanduser(bank_dir)
            if os.path.isdir(bank_path):
                files = sorted(
                    f for f in os.listdir(bank_path)
                    if f.lower().endswith((".webp", ".png", ".jpg", ".jpeg"))
                )
                for fname in files:
                    fpath = os.path.join(bank_path, fname)
                    with open(fpath, "rb") as fh:
                        self._bank_bytes.append(fh.read())
                if self._bank_bytes:
                    self._bank_available = True
                else:
                    warnings.warn(
                        f"RandomMoire: bank_dir '{bank_dir}' contains no "
                        f"images. Falling back to 100% sine-wave mode.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"RandomMoire: bank_dir '{bank_dir}' not found. "
                    f"Falling back to 100% sine-wave mode.",
                    stacklevel=2,
                )

    # -- Sine-wave moire --------------------------------------------------

    def _apply_sine(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        n_waves = random.randint(*self.num_waves_range)
        pattern = np.zeros((h, w), dtype=np.float32)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        for _ in range(n_waves):
            freq = random.uniform(*self.frequency_range)
            angle_deg = random.uniform(*self.angle_range)
            phase = random.uniform(0, 2 * np.pi)
            angle_rad = np.radians(angle_deg)

            # Project pixel coordinates onto wave direction.
            proj = xs * np.cos(angle_rad) + ys * np.sin(angle_rad)
            wave = np.sin(2 * np.pi * proj / freq + phase)
            pattern += wave

        # Normalize to [0, 1].
        pmin, pmax = pattern.min(), pattern.max()
        if pmax - pmin > 1e-6:
            pattern = (pattern - pmin) / (pmax - pmin)
        else:
            pattern = np.full_like(pattern, 0.5)

        # Optionally make it coloured (50 % chance).
        if random.random() < 0.5:
            color = np.array(
                [random.uniform(0.5, 1.0) for _ in range(3)],
                dtype=np.float32,
            )
            moire_rgb = pattern[:, :, None] * color[None, None, :] * 255.0
        else:
            moire_rgb = pattern[:, :, None] * 255.0

        lo, hi = _iscale_upper(self.sine_opacity_range, self._intensity)
        opacity = random.uniform(lo, hi)
        blended = arr * (1 - opacity) + moire_rgb * opacity

        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # -- Bank-blend moire -------------------------------------------------

    def _load_random_pattern(self, target_w: int, target_h: int) -> np.ndarray:
        raw = random.choice(self._bank_bytes)
        pat = Image.open(io.BytesIO(raw)).convert("RGB")

        # Random flip / rotation for diversity.
        if random.random() < 0.5:
            pat = pat.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            pat = pat.transpose(Image.FLIP_TOP_BOTTOM)
        rot = random.choice([0, 90, 180, 270])
        if rot:
            pat = pat.rotate(rot, expand=False)

        pw, ph = pat.size
        if pw >= target_w and ph >= target_h:
            # Random crop.
            x0 = random.randint(0, pw - target_w)
            y0 = random.randint(0, ph - target_h)
            pat = pat.crop((x0, y0, x0 + target_w, y0 + target_h))
        else:
            # Resize if pattern is smaller than target.
            pat = pat.resize((target_w, target_h), Image.LANCZOS)

        return np.array(pat, dtype=np.float32)

    def _apply_bank(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        pat = self._load_random_pattern(w, h)
        lo, hi = _iscale_upper(self.bank_opacity_range, self._intensity)
        opacity = random.uniform(lo, hi)

        mode = random.choice(["multiply", "alpha", "overlay"])

        if mode == "multiply":
            blended = arr * (pat / 255.0)
            # Blend with original to control strength.
            blended = arr * (1 - opacity) + blended * opacity

        elif mode == "alpha":
            blended = arr * (1 - opacity) + pat * opacity

        else:  # overlay
            base = arr / 255.0
            blend = pat / 255.0
            low = 2 * base * blend
            high = 1 - 2 * (1 - base) * (1 - blend)
            overlay = np.where(base < 0.5, low, high) * 255.0
            blended = arr * (1 - opacity) + overlay * opacity

        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # -- Main entry -------------------------------------------------------

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        use_sine = (
            random.random() < self.sine_ratio
            or not self._bank_available
        )

        if use_sine:
            return self._apply_sine(img)
        return self._apply_bank(img)

    def __repr__(self):
        bank_info = len(self._bank_bytes) if self._bank_available else "N/A"
        return (
            f"{self.__class__.__name__}("
            f"sine_ratio={self.sine_ratio}, "
            f"bank_images={bank_info}, "
            f"p={self.p})"
        )


# ---------------------------------------------------------------------------
# Grouped composition operators for robust augmentation
# ---------------------------------------------------------------------------


def _weighted_sample(names, weights, k):
    """Sample *k* items from *names* without replacement using *weights*.

    Uses sequential weighted draws: at each step one name is chosen
    proportionally to its weight, then removed from the pool.
    """
    pool = list(range(len(names)))
    w = list(weights)
    selected = []
    for _ in range(k):
        total = sum(w[i] for i in pool)
        r = random.random() * total
        cum = 0.0
        for j, idx in enumerate(pool):
            cum += w[idx]
            if cum >= r:
                selected.append(names[idx])
                pool.pop(j)
                break
    return selected


class GroupedNOfCompose:
    """Select 1–N groups, pick 1 transform per group, apply in random order.

    Ensures at most one transform per semantic category (e.g. no
    blur + blur), providing diverse degradation coverage.  Inspired by
    the group-based sampling in ``aug_utils_train/utils_data.py``.

    The actual number of groups selected each call is sampled uniformly
    from ``[1, n]``, adding intensity variance across images.

    Each group value can be either:

    - A plain list of transforms → uniform intra-group sampling.
    - A list of ``(transform, weight)`` tuples → weighted intra-group
      sampling.  Both forms can be mixed across groups.

    Args:
        groups: Dict mapping group names to lists of transform instances
            or ``(transform, weight)`` tuples.
        n: Maximum number of groups to select per call.
        weights: Optional dict mapping group names to sampling weights.
            Groups not listed default to 1.0.  Higher weight = more
            likely to be selected.  When ``None``, uniform sampling.
        clean_p: Probability of skipping all artifact transforms and
            returning the image unchanged (clean pass-through).
            Default ``0.0`` preserves the original behaviour.
    """

    def __init__(self, groups: dict, n: int = 4,
                 weights: dict | None = None,
                 clean_p: float = 0.0):
        self.groups = {}
        self._intra_weights = {}
        for name, items in groups.items():
            if items and isinstance(items[0], tuple):
                self.groups[name] = [t for t, _w in items]
                self._intra_weights[name] = [w for _t, w in items]
            else:
                self.groups[name] = items
                self._intra_weights[name] = None
        self.group_names = list(self.groups.keys())
        self.n = n
        self.clean_p = clean_p
        self._weights = (
            [weights.get(name, 1.0) for name in self.group_names]
            if weights is not None else None
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.clean_p > 0.0 and random.random() < self.clean_p:
            self._last_clean = True
            self._last_groups = frozenset({"clean"})
            return img
        self._last_clean = False
        n_upper = min(self.n, len(self.group_names))
        if n_upper <= 0:
            self._last_groups = frozenset({"clean"})
            return img
        k = random.randint(1, n_upper)
        if self._weights is not None:
            selected_groups = _weighted_sample(
                self.group_names, self._weights, k,
            )
        else:
            selected_groups = random.sample(self.group_names, k)
        self._last_groups = frozenset(selected_groups)
        transforms = []
        for group_name in selected_groups:
            iw = self._intra_weights[group_name]
            if iw is not None:
                t = random.choices(self.groups[group_name], weights=iw, k=1)[0]
            else:
                t = random.choice(self.groups[group_name])
            transforms.append(t)
        random.shuffle(transforms)
        for t in transforms:
            original_p = getattr(t, "p", None)
            if original_p is not None:
                t.p = 1.0
            img = t(img)
            if original_p is not None:
                t.p = original_p
        return img

    def __repr__(self):
        group_info = {name: len(ts) for name, ts in self.groups.items()}
        return (
            f"{self.__class__.__name__}("
            f"n={self.n}, groups={group_info}, "
            f"clean_p={self.clean_p}, "
            f"weights={self._weights})"
        )


class CurricularGroupedNOfCompose:
    """Curriculum-aware variant of :class:`GroupedNOfCompose`.

    The number of groups to apply each call is sampled uniformly from
    ``[n_min, upper]``, where *upper* ramps linearly from ``n_max_start``
    to ``n_max_end`` over the curriculum schedule.

    A *clean pass-through* probability can also follow the curriculum:
    ``clean_p_start`` at epoch 0 decreases linearly to ``clean_p_end``
    at curriculum completion.  When triggered the artifact transforms
    are skipped entirely, letting the model see unperturbed images.

    ==============  ===========================================
    Parameter       Role
    ==============  ===========================================
    ``n_min``       Fixed lower bound of the sampling range.
    ``n_max_start`` Upper bound at epoch 0.
    ``n_max_end``   Upper bound at curriculum completion.
    ``clean_p_start`` Clean pass-through probability at epoch 0.
    ``clean_p_end``   Clean pass-through probability at curriculum
                      completion (and beyond).
    ==============  ===========================================

    For backward compatibility, the legacy ``(n_max, n_min)`` interface
    is still accepted when ``n_max_start`` / ``n_max_end`` are omitted.

    Args:
        groups: Dict mapping group names to lists of transform instances.
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop.
        total_epochs: Total number of training epochs.
        n_min: Fixed lower bound for group count sampling.
        n_max_start: Upper bound at epoch 0.
        n_max_end: Upper bound when curriculum completes.
        curriculum_ratio: Fraction of total epochs over which the
            upper bound ramps from ``n_max_start`` to ``n_max_end``.
        clean_p_start: Clean pass-through probability at epoch 0.
            Default ``0.0`` preserves the original behaviour.
        clean_p_end: Clean pass-through probability at curriculum
            completion.  Default ``0.0``.
        n_max: **Deprecated** — legacy alias.  When ``n_max_start`` and
            ``n_max_end`` are both ``None``, ``n_min`` and ``n_max``
            fall back to the old behaviour (lower=1, upper=n_min→n_max).
    """

    def __init__(self, groups, epoch_state, total_epochs,
                 n_min=2, n_max_start=None, n_max_end=None,
                 curriculum_ratio=0.5,
                 weights=None,
                 clean_p_start: float = 0.0,
                 clean_p_end: float = 0.0,
                 intensity_curriculum: bool = False,
                 # legacy compat
                 n_max=None, n_start=None):
        # Parse groups: separate transforms and optional intra-weights
        self.groups = {}
        self._intra_weights = {}
        for name, items in groups.items():
            if items and isinstance(items[0], tuple):
                self.groups[name] = [t for t, _w in items]
                self._intra_weights[name] = [w for _t, w in items]
            else:
                self.groups[name] = items
                self._intra_weights[name] = None
        self.group_names = list(self.groups.keys())
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.curriculum_ratio = curriculum_ratio
        self.clean_p_start = clean_p_start
        self.clean_p_end = clean_p_end
        self.intensity_curriculum = intensity_curriculum
        self._weights = (
            [weights.get(name, 1.0) for name in self.group_names]
            if weights is not None else None
        )

        # --- resolve legacy / new params ---
        if n_max_start is not None and n_max_end is not None:
            # New-style params: use directly.
            self.n_min = n_min
            self.n_max_start = n_max_start
            self.n_max_end = n_max_end
            self._legacy = False
        elif n_start is not None and n_max is not None:
            # Transitional style (n_start / n_max).
            self.n_min = n_min
            self.n_max_start = n_start
            self.n_max_end = n_max
            self._legacy = False
        elif n_max is not None:
            # Pure legacy style (n_min / n_max, lower bound = 1).
            self.n_min = n_min
            self.n_max_start = n_min
            self.n_max_end = n_max
            self._legacy = True
        else:
            # Fallback defaults.
            self.n_min = n_min
            self.n_max_start = 3
            self.n_max_end = 7
            self._legacy = False

    def _get_progress(self):
        """Return curriculum progress in [0, 1]."""
        if self.total_epochs <= 1:
            return 1.0
        curriculum_epochs = max(self.total_epochs * self.curriculum_ratio, 1)
        if curriculum_epochs <= 1:
            return 1.0
        progress = self.epoch_state.value / (curriculum_epochs - 1)
        return min(max(progress, 0.0), 1.0)

    def _get_n_upper(self):
        progress = self._get_progress()
        return max(
            self.n_max_start,
            round(self.n_max_start + (self.n_max_end - self.n_max_start) * progress),
        )

    def _get_clean_p(self):
        """Return current clean pass-through probability."""
        if self.clean_p_start == 0.0 and self.clean_p_end == 0.0:
            return 0.0
        progress = self._get_progress()
        return self.clean_p_start + (self.clean_p_end - self.clean_p_start) * progress

    def __call__(self, img: Image.Image) -> Image.Image:
        clean_p = self._get_clean_p()
        if clean_p > 0.0 and random.random() < clean_p:
            self._last_clean = True
            self._last_groups = frozenset({"clean"})
            return img
        self._last_clean = False
        n_upper = min(self._get_n_upper(), len(self.group_names))
        if n_upper <= 0:
            self._last_groups = frozenset({"clean"})
            return img
        n_lower = 1 if self._legacy else self.n_min
        n_lower = min(n_lower, n_upper)
        k = random.randint(n_lower, n_upper)
        if self._weights is not None:
            selected_groups = _weighted_sample(
                self.group_names, self._weights, k,
            )
        else:
            selected_groups = random.sample(self.group_names, k)
        self._last_groups = frozenset(selected_groups)
        transforms = []
        for group_name in selected_groups:
            iw = self._intra_weights[group_name]
            if iw is not None:
                t = random.choices(self.groups[group_name], weights=iw, k=1)[0]
            else:
                t = random.choice(self.groups[group_name])
            transforms.append(t)
        random.shuffle(transforms)

        progress = (self._get_progress()
                    if self.intensity_curriculum else None)

        for t in transforms:
            original_p = getattr(t, "p", None)
            if original_p is not None:
                t.p = 1.0
            if progress is not None and hasattr(t, "_intensity"):
                t._intensity = progress
            img = t(img)
            if original_p is not None:
                t.p = original_p
            if progress is not None and hasattr(t, "_intensity"):
                t._intensity = 1.0
        return img

    def __repr__(self):
        group_info = {name: len(ts) for name, ts in self.groups.items()}
        return (
            f"{self.__class__.__name__}("
            f"n_min={self.n_min}, "
            f"n_max_start={self.n_max_start}, "
            f"n_max_end={self.n_max_end}, "
            f"groups={group_info}, "
            f"total_epochs={self.total_epochs}, "
            f"curriculum_ratio={self.curriculum_ratio}, "
            f"clean_p_start={self.clean_p_start}, "
            f"clean_p_end={self.clean_p_end}, "
            f"intensity_curriculum={self.intensity_curriculum}, "
            f"weights={self._weights})"
        )


class IntensityGaussianBlur:
    """Intensity-aware Gaussian blur. Scales sigma range with _intensity.

    When ``kernel_size=0`` (default), OpenCV auto-computes the kernel
    size from sigma so that the kernel always represents the Gaussian
    faithfully regardless of sigma magnitude.
    """

    def __init__(self, kernel_size=0, sigma=(0.1, 3.0), p=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self._intensity = 1.0

    def __call__(self, img):
        if random.random() > self.p:
            return img
        lo, hi = _iscale_upper(self.sigma, self._intensity)
        chosen_sigma = random.uniform(lo, max(lo, hi))
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (self.kernel_size, self.kernel_size), chosen_sigma)
        return Image.fromarray(arr)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"kernel_size={self.kernel_size}, sigma={self.sigma}, "
                f"p={self.p})")


class IntensityRandomPerspective:
    """Intensity-aware perspective distortion. Scales distortion_scale with _intensity."""

    def __init__(self, distortion_scale=0.3, p=1.0):
        self.distortion_scale = distortion_scale
        self.p = p
        self._intensity = 1.0

    def __call__(self, img):
        if random.random() > self.p:
            return img
        effective_scale = self.distortion_scale * self._intensity
        if effective_scale < 1e-6:
            return img
        orig_w, orig_h = img.size
        startpoints, endpoints = self._get_params(img.size, effective_scale)
        img = TF.perspective(
            img, startpoints, endpoints,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        # Crop to the largest inscribed rectangle to avoid black borders
        tl, tr, br, bl = endpoints
        crop_left = max(tl[0], bl[0])
        crop_right = min(tr[0], br[0])
        crop_top = max(tl[1], tr[1])
        crop_bottom = min(bl[1], br[1])
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        if crop_w >= orig_w * 0.5 and crop_h >= orig_h * 0.5:
            img = TF.crop(img, crop_top, crop_left, crop_h, crop_w)
            img = img.resize((orig_w, orig_h), Image.NEAREST)
        return img

    @staticmethod
    def _get_params(img_size, distortion_scale):
        """T.RandomPerspective.get_params() equivalent."""
        w, h = img_size
        half_h, half_w = h / 2, w / 2
        tl = [
            int(random.uniform(0, distortion_scale * half_w)),
            int(random.uniform(0, distortion_scale * half_h)),
        ]
        tr = [
            int(w - random.uniform(0, distortion_scale * half_w)),
            int(random.uniform(0, distortion_scale * half_h)),
        ]
        br = [
            int(w - random.uniform(0, distortion_scale * half_w)),
            int(h - random.uniform(0, distortion_scale * half_h)),
        ]
        bl = [
            int(random.uniform(0, distortion_scale * half_w)),
            int(h - random.uniform(0, distortion_scale * half_h)),
        ]
        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [tl, tr, br, bl]
        return startpoints, endpoints

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"distortion_scale={self.distortion_scale}, p={self.p})")


class CurricularColorJitter:
    """Curriculum-aware ColorJitter that scales parameters by training progress.

    At epoch 0 the jitter parameters are scaled by ``min_scale`` (gentle
    colour perturbation).  At ``total_epochs * curriculum_ratio`` they
    reach their full configured values, preventing aggressive colour
    distortion early in training when the model is still learning basic
    features.

    Unlike ``IntensityGaussianBlur`` which relies on an externally set
    ``_intensity`` attribute, this class reads ``epoch_state`` directly
    and computes its own scale — matching the self-contained design of
    :class:`CurricularWrapper`.

    Args:
        brightness: Maximum brightness jitter (same semantics as
            ``torchvision.transforms.ColorJitter``).
        contrast: Maximum contrast jitter.
        saturation: Maximum saturation jitter.
        hue: Maximum hue jitter.
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop.
        total_epochs: Total number of training epochs.
        curriculum_ratio: Fraction of total epochs over which the
            curriculum ramps from ``min_scale`` to 1.0.
        min_scale: Scale factor at epoch 0.  Default ``0.2`` gives
            gentle jitter (e.g. brightness ±8 %, hue ±3.6° when the
            full parameters are brightness=0.4, hue=0.1).
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3,
                 hue=0.1, epoch_state=None, total_epochs=30,
                 curriculum_ratio=0.5, min_scale=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.curriculum_ratio = curriculum_ratio
        self.min_scale = min_scale

    def _get_scale(self):
        if self.epoch_state is None or self.total_epochs <= 1:
            return 1.0
        curriculum_epochs = max(self.total_epochs * self.curriculum_ratio, 1)
        if curriculum_epochs <= 1:
            return 1.0
        progress = self.epoch_state.value / (curriculum_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_scale + (1.0 - self.min_scale) * progress

    def __call__(self, img: Image.Image) -> Image.Image:
        s = self._get_scale()
        jitter = T.ColorJitter(
            brightness=self.brightness * s,
            contrast=self.contrast * s,
            saturation=self.saturation * s,
            hue=self.hue * s,
        )
        return jitter(img)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue}, "
            f"min_scale={self.min_scale}, "
            f"curriculum_ratio={self.curriculum_ratio})"
        )


class SkipIfClean:
    """Skip *transform* when the upstream compose triggered clean pass-through.

    Reads ``source._last_clean`` set by :class:`GroupedNOfCompose` or
    :class:`CurricularGroupedNOfCompose`.  When the flag is ``True``
    the wrapped transform is skipped, keeping the image artifact-free.

    Args:
        source: The upstream compose instance whose ``_last_clean``
            attribute is checked each call.
        transform: Transform to apply when not in clean mode.
    """

    def __init__(self, source, transform):
        self.source = source
        self.transform = transform

    def __call__(self, img):
        if getattr(self.source, "_last_clean", False):
            return img
        # Propagate intensity from curriculum source.
        progress = None
        if (hasattr(self.transform, "_intensity")
                and getattr(self.source, "intensity_curriculum", False)):
            progress = self.source._get_progress()
            self.transform._intensity = progress
        result = self.transform(img)
        if progress is not None:
            self.transform._intensity = 1.0  # reset
        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"source={self.source.__class__.__name__}, "
            f"transform={self.transform})"
        )
