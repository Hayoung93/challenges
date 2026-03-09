import bisect
import io
import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow.ipc as ipc
from PIL import Image

from .base import BaseGenAIDataset


class DragonArrowDataset(BaseGenAIDataset):
    """Dataset for the Dragon GenAI dataset stored as Arrow IPC streaming files.

    All images are AI-generated (label=1). Images are stored as inline PNG bytes
    inside arrow struct columns. Uses an LRU cache to avoid holding all 62 files
    (~29 GB) in memory simultaneously.

    Args:
        root: Directory containing ``data-XXXXX-of-YYYYY.arrow`` files.
        transform: Optional image transform.
        split: ``"train"`` or ``"val"``.
        lru_capacity: Max number of arrow Tables held in memory at once.
        index_cache_path: Path to save/load the per-file row-count index (JSON).
            Skips the full file scan on subsequent runs when provided.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        lru_capacity: int = 4,
        index_cache_path: Optional[str] = None,
    ):
        super().__init__(transform=transform, split=split)
        self.root = root
        self.lru_capacity = lru_capacity

        # Discover and sort arrow files
        self._arrow_files: List[str] = sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".arrow")
        )
        if not self._arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {root}")

        # Build cumulative row index
        self._file_row_counts: List[int] = []
        self._cum_rows: List[int] = []
        self._build_index(index_cache_path)
        self._total_rows = self._cum_rows[-1] if self._cum_rows else 0

        # LRU cache: file_index -> pyarrow.Table
        self._table_cache: OrderedDict[int, Any] = OrderedDict()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, cache_path: Optional[str]) -> None:
        cache_valid = False
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if len(cached) == len(self._arrow_files):
                self._file_row_counts = cached
                cache_valid = True

        if not cache_valid:
            self._file_row_counts = []
            for arrow_path in self._arrow_files:
                reader = ipc.open_stream(arrow_path)
                count = sum(batch.num_rows for batch in reader)
                self._file_row_counts.append(count)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(self._file_row_counts, f)

        cumulative = 0
        for count in self._file_row_counts:
            cumulative += count
            self._cum_rows.append(cumulative)

    # ------------------------------------------------------------------
    # Index resolution & table caching
    # ------------------------------------------------------------------

    def _resolve_index(self, global_idx: int) -> Tuple[int, int]:
        """Map a global row index to (file_index, local_row_index)."""
        file_idx = bisect.bisect_right(self._cum_rows, global_idx)
        offset = self._cum_rows[file_idx - 1] if file_idx > 0 else 0
        return file_idx, global_idx - offset

    def _get_table(self, file_idx: int):
        """Return the pyarrow Table for *file_idx*, using an LRU cache."""
        if file_idx in self._table_cache:
            self._table_cache.move_to_end(file_idx)
            return self._table_cache[file_idx]

        # Cache miss — load from disk
        reader = ipc.open_stream(self._arrow_files[file_idx])
        table = reader.read_all()

        # Evict LRU entries if at capacity
        while len(self._table_cache) >= self.lru_capacity:
            self._table_cache.popitem(last=False)

        self._table_cache[file_idx] = table
        return table

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_rows

    def _load_image_and_label(self, index: int) -> Tuple[Image.Image, int, Dict[str, Any]]:
        if index < 0:
            index += self._total_rows
        if index < 0 or index >= self._total_rows:
            raise IndexError(f"Index {index} out of range for dataset of size {self._total_rows}")

        file_idx, local_idx = self._resolve_index(index)
        table = self._get_table(file_idx)

        # Decode image from inline PNG bytes
        png_struct = table.column("png")[local_idx].as_py()
        image = Image.open(io.BytesIO(png_struct["bytes"])).convert("RGB")

        label = 1  # all Dragon images are AI-generated

        metadata: Dict[str, Any] = {
            "dataset": "dragon",
            "source_id": table.column("__key__")[local_idx].as_py(),
            "model": table.column("model.txt")[local_idx].as_py(),
            "prompt_cls": table.column("prompt.cls")[local_idx].as_py(),
        }

        return image, label, metadata

    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:
        image, label, metadata = self._load_image_and_label(index)
        image = self._apply_transform(image)
        if hasattr(self.transform, "last_groups"):
            metadata["aug_groups"] = self.transform.last_groups
        return image, label, metadata
