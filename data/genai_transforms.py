"""GenAI-specific augmentation transforms for deepfake/AI-image detection.

These transforms target pixel-level artifacts (JPEG compression, resize
interpolation, sensor noise) that are critical signals for distinguishing
real images from AI-generated ones.  All transforms operate on PIL Images
and are compatible with ``torchvision.transforms.Compose``.
"""

import io
import math
import random

import numpy as np
import scipy.ndimage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from scipy.interpolate import PchipInterpolator


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
        radius = random.randint(*self.radius_range)
        kernel = self._make_disk_kernel(radius)
        arr = np.array(img, dtype=np.float64)
        for c in range(arr.shape[2]):
            arr[:, :, c] = scipy.ndimage.convolve(
                arr[:, :, c], kernel, mode="reflect",
            )
        arr = np.clip(arr, 0, 255).astype(np.uint8)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        levels = random.randint(*self.levels_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        amount = random.uniform(*self.amount_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        amount = random.uniform(*self.amount_range)
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
        amount = random.uniform(*self.amount_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p})"
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
        min_s, max_s = self.kernel_size_range
        # Ensure odd kernel size
        sizes = list(range(min_s | 1, max_s + 1, 2))
        if not sizes:
            sizes = [min_s | 1]
        size = random.choice(sizes)
        angle = random.uniform(0, 360)
        kernel = self._make_motion_kernel(size, angle)
        arr = np.array(img, dtype=np.float64)
        for c in range(arr.shape[2]):
            arr[:, :, c] = scipy.ndimage.convolve(
                arr[:, :, c], kernel, mode="reflect",
            )
        arr = np.clip(arr, 0, 255).astype(np.uint8)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        from PIL import ImageOps
        bits = random.randint(*self.bits_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ycbcr = img.convert("YCbCr")
        arr = np.array(ycbcr, dtype=np.float32)
        std = random.uniform(*self.std_range)
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

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ycbcr = img.convert("YCbCr")
        arr = np.array(ycbcr, dtype=np.float32)
        std = random.uniform(*self.std_range)
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


class RandomMoire:
    """Apply moire-pattern augmentation via sine-wave synthesis or real pattern blending.

    When triggered (probability *p*), the transform selects **one** of two
    strategies controlled by ``sine_ratio``:

    * **Sine-wave mode** (default 70%): generates 2–3 overlapping sine waves
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

        opacity = random.uniform(*self.sine_opacity_range)
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
        opacity = random.uniform(*self.bank_opacity_range)

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


class GroupedNOfCompose:
    """Select 1–N groups, pick 1 transform per group, apply in random order.

    Ensures at most one transform per semantic category (e.g. no
    blur + blur), providing diverse degradation coverage.  Inspired by
    the group-based sampling in ``aug_utils_train/utils_data.py``.

    The actual number of groups selected each call is sampled uniformly
    from ``[1, n]``, adding intensity variance across images.

    Args:
        groups: Dict mapping group names to lists of transform instances.
        n: Maximum number of groups to select per call.
    """

    def __init__(self, groups: dict, n: int = 4):
        self.groups = groups
        self.group_names = list(groups.keys())
        self.n = n

    def __call__(self, img: Image.Image) -> Image.Image:
        n_upper = min(self.n, len(self.group_names))
        if n_upper <= 0:
            return img
        k = random.randint(1, n_upper)
        selected_groups = random.sample(self.group_names, k)
        transforms = []
        for group_name in selected_groups:
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
            f"n={self.n}, groups={group_info})"
        )


class CurricularGroupedNOfCompose:
    """Curriculum-aware variant of :class:`GroupedNOfCompose`.

    The number of groups to apply each call is sampled uniformly from
    ``[n_min, upper]``, where *upper* ramps linearly from ``n_max_start``
    to ``n_max_end`` over the curriculum schedule.

    ==============  ===========================================
    Parameter       Role
    ==============  ===========================================
    ``n_min``       Fixed lower bound of the sampling range.
    ``n_max_start`` Upper bound at epoch 0.
    ``n_max_end``   Upper bound at curriculum completion.
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
        n_max: **Deprecated** — legacy alias.  When ``n_max_start`` and
            ``n_max_end`` are both ``None``, ``n_min`` and ``n_max``
            fall back to the old behaviour (lower=1, upper=n_min→n_max).
    """

    def __init__(self, groups, epoch_state, total_epochs,
                 n_min=2, n_max_start=None, n_max_end=None,
                 curriculum_ratio=0.5,
                 # legacy compat
                 n_max=None, n_start=None):
        self.groups = groups
        self.group_names = list(groups.keys())
        self.epoch_state = epoch_state
        self.total_epochs = total_epochs
        self.curriculum_ratio = curriculum_ratio

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

    def _get_n_upper(self):
        if self.total_epochs <= 1:
            return self.n_max_end
        curriculum_epochs = max(self.total_epochs * self.curriculum_ratio, 1)
        progress = self.epoch_state.value / (curriculum_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return max(
            self.n_max_start,
            round(self.n_max_start + (self.n_max_end - self.n_max_start) * progress),
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        n_upper = min(self._get_n_upper(), len(self.group_names))
        if n_upper <= 0:
            return img
        n_lower = 1 if self._legacy else self.n_min
        k = random.randint(n_lower, n_upper)
        selected_groups = random.sample(self.group_names, k)
        transforms = []
        for group_name in selected_groups:
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
            f"n_min={self.n_min}, "
            f"n_max_start={self.n_max_start}, "
            f"n_max_end={self.n_max_end}, "
            f"groups={group_info}, "
            f"total_epochs={self.total_epochs}, "
            f"curriculum_ratio={self.curriculum_ratio})"
        )
