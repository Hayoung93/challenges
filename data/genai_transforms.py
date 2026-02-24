"""GenAI-specific augmentation transforms for deepfake/AI-image detection.

These transforms target pixel-level artifacts (JPEG compression, resize
interpolation, sensor noise) that are critical signals for distinguishing
real images from AI-generated ones.  All transforms operate on PIL Images
and are compatible with ``torchvision.transforms.Compose``.
"""

import io
import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter


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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p})"
        )


class RandomDownscaleUpscale:
    """Downscale then upscale to introduce resize / interpolation artifacts.

    Simulates resolution changes from web sharing, screenshots, or
    thumbnail generation.  The round-trip destroys high-frequency detail
    and introduces characteristic aliasing patterns.

    Args:
        scale_range: ``(min_scale, max_scale)`` relative to original size.
        p: Probability of applying this transform.
    """

    def __init__(self, scale_range=(0.5, 0.9), p=0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        down = img.resize((new_w, new_h), Image.BILINEAR)
        up = down.resize((w, h), Image.BILINEAR)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        std = random.uniform(*self.std_range)
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
    or pure black (pepper, 0).

    Args:
        amount: Fraction of pixels to corrupt (0.0--1.0).
        p: Probability of applying this transform.
    """

    def __init__(self, amount=0.05, p=0.05):
        self.amount = amount
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        mask = np.random.random((h, w))
        arr[mask < self.amount / 2] = 255       # salt
        arr[mask > 1 - self.amount / 2] = 0     # pepper
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount={self.amount}, p={self.p})"
        )


class RandomImpulseNoise:
    """Add channel-independent impulse noise to a PIL image.

    Unlike salt-and-pepper noise which sets entire pixels to black or
    white, impulse noise corrupts each RGB channel independently,
    producing colourful speckles.

    Args:
        amount: Fraction of channel values to corrupt (0.0--1.0).
        p: Probability of applying this transform.
    """

    def __init__(self, amount=0.05, p=0.03):
        self.amount = amount
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        mask = np.random.random(arr.shape)
        arr[mask < self.amount / 2] = 255
        arr[mask > 1 - self.amount / 2] = 0
        return Image.fromarray(arr)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"amount={self.amount}, p={self.p})"
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


class CurricularWrapper:
    """Wrap transforms and scale their probability by training progress.

    At epoch 0, all child-transform probabilities are scaled down by
    ``min_scale``; at the final epoch they reach their full configured
    probability.  This implements a linear curriculum that starts with
    mild augmentation and gradually increases difficulty.

    The ``epoch_state`` must be a ``multiprocessing.Value('i', 0)``
    (shared-memory integer) so that persistent DataLoader workers can
    observe epoch updates made by the training loop.

    Args:
        transforms: Transform objects, each with a ``p`` attribute.
        epoch_state: ``multiprocessing.Value`` holding the current epoch.
        total_epochs: Total number of training epochs.
        min_scale: Probability scale factor at epoch 0.
    """

    def __init__(self, transforms, epoch_state, total_epochs, min_scale=0.1):
        self.transforms = transforms
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.min_scale = min_scale

    def _get_scale(self):
        if self.total_epochs <= 1:
            return 1.0
        progress = self.epoch_state.value / (self.total_epochs - 1)
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
            f"min_scale={self.min_scale})"
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice(self.kernel_sizes)
        return img.filter(ImageFilter.MedianFilter(size=k))

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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        r = random.randint(*self.radius_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        factor = random.uniform(*self.factor_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        ratio = random.uniform(*self.ratio_range)
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
    ``n_min`` at epoch 0 to ``n_max`` at the final epoch.

    Args:
        transforms: Pool of candidate transforms.
        epoch_state: ``multiprocessing.Value('i', 0)`` shared with the
            training loop.
        total_epochs: Total number of training epochs.
        n_max: Maximum number of transforms at full curriculum.
        n_min: Starting number of transforms at epoch 0.
    """

    def __init__(self, transforms, epoch_state, total_epochs,
                 n_max=5, n_min=1):
        self.transforms = transforms
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.n_max = n_max
        self.n_min = n_min

    def _get_n(self):
        if self.total_epochs <= 1:
            return self.n_max
        progress = self.epoch_state.value / (self.total_epochs - 1)
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
            f"total_epochs={self.total_epochs})"
        )
