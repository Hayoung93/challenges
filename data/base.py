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
    """Wraps a :class:`BaseGenAIDataset` to return two augmented views.

    Uses :meth:`BaseGenAIDataset._load_image_and_label` to obtain the raw
    PIL image, then applies two independent transforms.

    Each ``__getitem__`` returns a 4-tuple:
        ``(view1, view2, label, metadata)``

    Args:
        dataset: The underlying dataset (must implement ``_load_image_and_label``).
        transform2: The transform to apply for the second view.
            The first view uses ``dataset.transform``.
    """

    def __init__(self, dataset: BaseGenAIDataset, transform2: Callable):
        self.dataset = dataset
        self.transform1 = dataset.transform
        self.transform2 = transform2

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, Dict[str, Any]]:
        image, label, metadata = self.dataset._load_image_and_label(index)
        view1 = self.transform1(image)
        view2 = self.transform2(image)
        if hasattr(self.transform1, "last_groups"):
            metadata["aug_groups"] = self.transform1.last_groups
        return view1, view2, label, metadata
