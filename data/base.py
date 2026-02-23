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
