import csv
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .base import BaseGenAIDataset


class NTIREDataset(BaseGenAIDataset):
    """Dataset for the NTIRE2026 GenAI detection challenge.

    Each shard directory contains an ``images/`` folder of JPEG files and a
    ``labels.csv`` with columns ``image_name`` and ``label`` (0=real, 1=fake).

    Args:
        root: Path to the top-level directory containing ``shard_N/`` subdirs.
        shards: Shard indices to include.  ``None`` discovers all shards.
        transform: Optional image transform.
        split: ``"train"`` or ``"val"``.
    """

    def __init__(
        self,
        root: str,
        shards: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        super().__init__(transform=transform, split=split)
        self.root = root

        if shards is None:
            shards = sorted(
                int(d.split("_")[1])
                for d in os.listdir(root)
                if d.startswith("shard_") and os.path.isdir(os.path.join(root, d))
            )

        self.samples: List[Tuple[str, int]] = []
        self._load_shards(shards)

    def _load_shards(self, shards: List[int]) -> None:
        for shard_idx in shards:
            shard_dir = os.path.join(self.root, f"shard_{shard_idx}")
            csv_path = os.path.join(shard_dir, "labels.csv")
            images_dir = os.path.join(shard_dir, "images")

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_path = os.path.join(images_dir, row["image_name"])
                    label = int(row["label"])
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_and_label(self, index: int) -> Tuple[Image, int, Dict[str, Any]]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.samples)}")
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        metadata: Dict[str, Any] = {
            "dataset": "ntire",
            "source_id": os.path.basename(image_path),
        }

        return image, label, metadata

    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:
        image, label, metadata = self._load_image_and_label(index)
        image = self._apply_transform(image)
        if hasattr(self.transform, "last_groups"):
            metadata["aug_groups"] = self.transform.last_groups
        return image, label, metadata


class DistortedValDataset(BaseGenAIDataset):
    """Dataset for pre-generated distorted validation images.

    Reads images from a flat ``images/`` directory and labels from a
    ``labels.csv`` file produced by ``scripts/generate_distorted_val.py``.

    The CSV must contain at least ``image_name`` and ``label`` columns.
    Optional columns (``source_id``, ``copy_idx``, ``intensity``,
    ``n_groups``) are stored in metadata.

    Args:
        root: Path to the distorted val directory (contains ``images/``
            and ``labels.csv``).
        transform: Image transform (typically ``get_val_transform``).
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform, split="val")
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.samples: List[Tuple[str, int, Dict[str, Any]]] = []
        self._load_labels()

    def _load_labels(self) -> None:
        csv_path = os.path.join(self.root, "labels.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Distorted val labels not found: {csv_path}. "
                f"Run scripts/generate_distorted_val.py first."
            )
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = os.path.join(self.images_dir, row["image_name"])
                label = int(row["label"])
                extra = {
                    k: row[k] for k in row
                    if k not in ("image_name", "label")
                }
                self.samples.append((image_path, label, extra))
        missing = [p for p, _, _ in self.samples if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images referenced in labels.csv not found. "
                f"First: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_and_label(self, index: int) -> Tuple[Image, int, Dict[str, Any]]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for dataset of size "
                f"{len(self.samples)}"
            )
        image_path, label, extra = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        metadata: Dict[str, Any] = {
            "dataset": "distorted_val",
            "source_id": os.path.basename(image_path),
            **extra,
        }
        return image, label, metadata

    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:
        image, label, metadata = self._load_image_and_label(index)
        image = self._apply_transform(image)
        return image, label, metadata


class NTIRETestDataset(BaseGenAIDataset):
    """Dataset for unlabeled NTIRE2026 test images.

    Reads JPEG/PNG files from one or more flat directories (no labels).
    Each sample gets ``label=-1`` (no ground truth).  Metadata includes a
    ``"subset"`` field identifying the source directory.

    Args:
        root: Path to the test directory (e.g., ``{ntire_root}/test/``).
        subsets: List of subdirectory names to include
            (e.g., ``["val_images"]``, ``["val_images_hard"]``, or both).
        transform: Optional image transform.
    """

    _EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str,
        subsets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform, split="test")
        self.root = root

        if subsets is None:
            subsets = ["val_images"]

        self.samples: List[Tuple[str, int, str]] = []
        self._discover_images(subsets)

    def _discover_images(self, subsets: List[str]) -> None:
        for subset_name in subsets:
            subset_dir = os.path.join(self.root, subset_name)
            if not os.path.isdir(subset_dir):
                raise FileNotFoundError(
                    f"Test subset directory not found: {subset_dir}"
                )
            filenames = sorted(
                f for f in os.listdir(subset_dir)
                if os.path.splitext(f)[1].lower() in self._EXTENSIONS
            )
            for fname in filenames:
                image_path = os.path.join(subset_dir, fname)
                self.samples.append((image_path, -1, subset_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self.samples)}"
            )
        image_path, label, subset = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        metadata: Dict[str, Any] = {
            "dataset": "ntire_test",
            "source_id": os.path.basename(image_path),
            "subset": subset,
        }

        image = self._apply_transform(image)
        return image, label, metadata
