from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class BaseGenAIDataset(Dataset, ABC):
    """Abstract base for GenAI image detection datasets.

    Every __getitem__ returns:
        image: torch.Tensor (C, H, W) after transform, or PIL.Image if no transform
        label: int  (0=real, 1=fake)
        metadata: dict  dataset-specific metadata
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        self.transform = transform
        self.split = split

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:
        ...

    def _apply_transform(self, image: Image.Image) -> Any:
        if self.transform is not None:
            return self.transform(image)
        return image

    def _load_image_and_label(
        self, index: int,
    ) -> Tuple[Image.Image, int, Dict[str, Any]]:
        """Load a raw PIL image, label, and metadata *without* applying transforms.

        Subclasses should override this to support :class:`MultiViewDataset`.
        The default implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _load_image_and_label"
        )


class MultiViewDataset(Dataset):
    """Wraps a :class:`BaseGenAIDataset` to return two views with shared spatial content.

    Uses :meth:`BaseGenAIDataset._load_image_and_label` to obtain the raw
    PIL image, applies shared spatial transforms **once** (so both views
    see the same crop / flip / rotation), then applies appearance and
    artifact transforms only to view 1.

    Each ``__getitem__`` returns a 4-tuple:
        ``(view1, view2, label, metadata)``

    Args:
        dataset: The underlying dataset (must implement ``_load_image_and_label``).
        shared_spatial: Spatial transforms applied once (PIL -> PIL).
            May be a :class:`_MultiscaleMultiViewWrapper` for multi-scale
            training, in which case *augment_only* and *to_tensor_norm*
            are ignored (the wrapper resolves all three at runtime).
        augment_only: Appearance + artifact transforms for view 1 only
            (PIL -> PIL).
        to_tensor_norm: ``ToTensor`` + ``Normalize`` (PIL -> Tensor).
    """

    def __init__(
        self,
        dataset: BaseGenAIDataset,
        shared_spatial: Callable,
        augment_only: Callable,
        to_tensor_norm: Callable,
    ):
        self.dataset = dataset
        self.shared_spatial = shared_spatial
        self.augment_only = augment_only
        self.to_tensor_norm = to_tensor_norm
        # For multiscale support: wrapper resolves transforms per-call
        self._is_multiscale = hasattr(shared_spatial, "get_transforms")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, Dict[str, Any]]:
        image, label, metadata = self.dataset._load_image_and_label(index)

        # Resolve transforms (multiscale may change per-call)
        if self._is_multiscale:
            spatial, augment, ttn = self.shared_spatial.get_transforms()
        else:
            spatial = self.shared_spatial
            augment = self.augment_only
            ttn = self.to_tensor_norm

        # Shared spatial: both views see the same crop / flip / rotation
        cropped = spatial(image)

        # view1: augmented (appearance + artifacts)
        view1 = ttn(augment(cropped))
        # view2: clean
        view2 = ttn(cropped)

        # Propagate aug_groups metadata (for MoE tracking)
        if hasattr(augment, "last_groups"):
            metadata["aug_groups"] = augment.last_groups
        elif hasattr(augment, "transforms"):
            for t in augment.transforms:
                if hasattr(t, "_last_groups"):
                    metadata["aug_groups"] = t._last_groups
                    break

        return view1, view2, label, metadata
