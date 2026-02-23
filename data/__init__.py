import multiprocessing
import os
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .base import BaseGenAIDataset
from .dragon import DragonArrowDataset
from .ntire import NTIREDataset, NTIRETestDataset
from .transforms import get_train_transform, get_val_transform

def build_dataset(name: str, args, split: str = "train", epoch_state=None) -> BaseGenAIDataset:
    """Build a single dataset by name.

    Args:
        name: ``"dragon"`` or ``"ntire"``.
        args: Namespace with config attributes (see ``config.py``).
        split: ``"train"`` or ``"val"``.
        epoch_state: ``multiprocessing.Value`` for curricular augmentation.
    """
    if split == "train":
        transform = get_train_transform(
            image_size=getattr(args, "image_size", 224),
            augmentation=getattr(args, "augmentation", "default"),
            total_epochs=getattr(args, "epochs", 30),
            epoch_state=epoch_state,
        )
    else:
        transform = get_val_transform(
            image_size=getattr(args, "image_size", 224),
            resize_size=getattr(args, "resize_size", 256),
        )

    if name == "dragon":
        return DragonArrowDataset(
            root=getattr(args, "dragon_root", "/data/data/dragon_dataset_regular"),
            transform=transform,
            split=split,
            lru_capacity=getattr(args, "dragon_lru_capacity", 4),
            index_cache_path=getattr(
                args,
                "dragon_index_cache",
                "/workspace/challenge_genai/.cache/dragon_index.json",
            ),
        )
    elif name == "ntire":
        ntire_root = getattr(args, "ntire_root", "/data/data/NTIRE2026_GenAI")
        return NTIREDataset(
            root=os.path.join(ntire_root, "train"),
            shards=getattr(args, "ntire_shards", None),
            transform=transform,
            split=split,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Available: ['dragon', 'ntire']"
        )


def _collate_fn(batch):
    """Collate 3-tuples ``(image, label, metadata)`` into batched tensors."""
    images, labels, metadata = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, list(metadata)


def _make_loader_kwargs(args, is_train: bool) -> dict:
    """Build shared DataLoader keyword arguments."""
    num_workers = getattr(args, "num_workers", 8)
    batch_size = getattr(args, "batch_size", 32)
    if getattr(args, "distributed", False):
        import torch.distributed as dist
        batch_size = batch_size // dist.get_world_size()
    kw = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": getattr(args, "pin_memory", True),
        "drop_last": getattr(args, "drop_last", True) if is_train else False,
        "collate_fn": _collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kw["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
    return kw


def build_dataloader(
    args,
    split: str = "train",
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """Build DataLoader(s) for the given split.

    Returns a single ``DataLoader`` when ``args.dataset_mode == "concat"``
    (default), or a ``dict[str, DataLoader]`` when ``"separate"``.
    """
    dataset_names: List[str] = getattr(
        args, "train_datasets" if split == "train" else "val_datasets", []
    )
    dataset_mode: str = getattr(args, "dataset_mode", "concat")

    if not dataset_names:
        raise ValueError(f"No datasets specified for split='{split}'")

    # Build individual datasets
    datasets: Dict[str, BaseGenAIDataset] = {}
    for name in dataset_names:
        datasets[name] = build_dataset(name, args, split=split)
        print(f"  [{split}] {name}: {len(datasets[name]):,} samples")

    # Shared DataLoader kwargs
    is_train = split == "train"
    loader_kwargs = _make_loader_kwargs(args, is_train)

    if dataset_mode == "concat":
        combined = ConcatDataset(list(datasets.values()))
        print(f"  [{split}] total: {len(combined):,} samples")

        if getattr(args, "distributed", False):
            sampler = torch.utils.data.distributed.DistributedSampler(
                combined, shuffle=is_train
            )
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = is_train

        return DataLoader(combined, **loader_kwargs)

    elif dataset_mode == "separate":
        loaders: Dict[str, DataLoader] = {}
        for name, ds in datasets.items():
            kw = dict(loader_kwargs)
            if getattr(args, "distributed", False):
                sampler = torch.utils.data.distributed.DistributedSampler(
                    ds, shuffle=is_train
                )
                kw["sampler"] = sampler
            else:
                kw["shuffle"] = is_train
            loaders[name] = DataLoader(ds, **kw)
        return loaders

    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")


def build_train_val_loaders(
    args,
) -> Tuple[
    Union[DataLoader, Dict[str, DataLoader]],
    Union[DataLoader, Dict[str, DataLoader]],
]:
    """Build train and validation DataLoaders with a random split.

    Uses ``args.val_split_ratio`` to hold out a fraction of training data
    for validation.  The split is seeded by ``args.seed`` for reproducibility.

    Returns:
        ``(train_loader, val_loader)`` — each follows the same
        ``dataset_mode`` semantics as :func:`build_dataloader`.
    """
    val_ratio: float = getattr(args, "val_split_ratio", 0.1)
    seed: int = getattr(args, "seed", 42)
    dataset_mode: str = getattr(args, "dataset_mode", "concat")
    dataset_names: List[str] = getattr(args, "train_datasets", [])

    if not dataset_names:
        raise ValueError("No datasets specified for training")

    # Create shared epoch counter for curricular augmentation
    augmentation = getattr(args, "augmentation", "default")
    epoch_state = None
    if augmentation == "genai_curriculum":
        epoch_state = multiprocessing.Value("i", 0)

    # Build datasets with both train and val transforms
    train_transform_datasets: Dict[str, BaseGenAIDataset] = {}
    val_transform_datasets: Dict[str, BaseGenAIDataset] = {}
    for name in dataset_names:
        train_transform_datasets[name] = build_dataset(
            name, args, split="train", epoch_state=epoch_state,
        )
        val_transform_datasets[name] = build_dataset(name, args, split="val")
        print(f"  [full] {name}: {len(train_transform_datasets[name]):,} samples")

    if dataset_mode == "concat":
        combined_train = ConcatDataset(list(train_transform_datasets.values()))
        combined_val = ConcatDataset(list(val_transform_datasets.values()))
        total = len(combined_train)
        val_size = int(total * val_ratio)
        train_size = total - val_size
        print(f"  [split] total={total:,}  train={train_size:,}  val={val_size:,}")

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_subset = torch.utils.data.Subset(combined_train, train_indices)
        val_subset = torch.utils.data.Subset(combined_val, val_indices)

        train_kw = _make_loader_kwargs(args, is_train=True)
        if getattr(args, "distributed", False):
            train_kw["sampler"] = torch.utils.data.distributed.DistributedSampler(
                train_subset, shuffle=True
            )
        else:
            train_kw["shuffle"] = True
        train_loader = DataLoader(train_subset, **train_kw)
        train_loader._epoch_state = epoch_state

        val_kw = _make_loader_kwargs(args, is_train=False)
        if getattr(args, "distributed", False):
            val_kw["sampler"] = torch.utils.data.distributed.DistributedSampler(
                val_subset, shuffle=False
            )
        else:
            val_kw["shuffle"] = False
        val_loader = DataLoader(val_subset, **val_kw)

        return train_loader, val_loader

    elif dataset_mode == "separate":
        train_loaders: Dict[str, DataLoader] = {}
        val_loaders: Dict[str, DataLoader] = {}

        for name in train_transform_datasets:
            ds_train = train_transform_datasets[name]
            ds_val = val_transform_datasets[name]
            total = len(ds_train)
            val_size = int(total * val_ratio)
            train_size = total - val_size
            print(f"  [split] {name}: total={total:,}  train={train_size:,}  val={val_size:,}")

            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total, generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_sub = torch.utils.data.Subset(ds_train, train_indices)
            val_sub = torch.utils.data.Subset(ds_val, val_indices)

            train_kw = _make_loader_kwargs(args, is_train=True)
            if getattr(args, "distributed", False):
                train_kw["sampler"] = torch.utils.data.distributed.DistributedSampler(
                    train_sub, shuffle=True
                )
            else:
                train_kw["shuffle"] = True
            loader = DataLoader(train_sub, **train_kw)
            loader._epoch_state = epoch_state
            train_loaders[name] = loader

            val_kw = _make_loader_kwargs(args, is_train=False)
            if getattr(args, "distributed", False):
                val_kw["sampler"] = torch.utils.data.distributed.DistributedSampler(
                    val_sub, shuffle=False
                )
            else:
                val_kw["shuffle"] = False
            val_loaders[name] = DataLoader(val_sub, **val_kw)

        return train_loaders, val_loaders

    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")


def build_test_dataloader(
    args,
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """Build DataLoader(s) for NTIRE test (unlabeled) images.

    Behaviour depends on ``args.ntire_test_mode``:
        1 → single DataLoader for ``val_images``
        2 → single DataLoader for ``val_images_hard``
        3 → ``dict[str, DataLoader]`` keyed by subset name

    Returns:
        A single DataLoader (modes 1 & 2) or dict of DataLoaders (mode 3).
    """
    ntire_root = getattr(args, "ntire_root", "/data/data/NTIRE2026_GenAI")
    test_root = os.path.join(ntire_root, "test")
    test_mode: int = getattr(args, "ntire_test_mode", 1)

    transform = get_val_transform(
        image_size=getattr(args, "image_size", 224),
        resize_size=getattr(args, "resize_size", 256),
    )

    _MODE_SUBSETS = {
        1: ["val_images"],
        2: ["val_images_hard"],
        3: ["val_images", "val_images_hard"],
    }
    if test_mode not in _MODE_SUBSETS:
        raise ValueError(
            f"Invalid ntire_test_mode={test_mode}. Must be 1, 2, or 3."
        )

    loader_kwargs = _make_loader_kwargs(args, is_train=False)
    loader_kwargs["shuffle"] = False

    if test_mode in (1, 2):
        subsets = _MODE_SUBSETS[test_mode]
        ds = NTIRETestDataset(root=test_root, subsets=subsets, transform=transform)
        print(f"  [test] {subsets[0]}: {len(ds):,} samples")
        return DataLoader(ds, **loader_kwargs)

    else:  # mode 3
        loaders: Dict[str, DataLoader] = {}
        for subset_name in _MODE_SUBSETS[3]:
            ds = NTIRETestDataset(
                root=test_root, subsets=[subset_name], transform=transform
            )
            print(f"  [test] {subset_name}: {len(ds):,} samples")
            loaders[subset_name] = DataLoader(ds, **dict(loader_kwargs))
        return loaders
