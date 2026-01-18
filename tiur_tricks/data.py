from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

# NOTE:
# This harness uses torchvision datasets/transforms. Colab typically comes with a
# compatible torch+torchvision pair preinstalled. Unfortunately, it's easy to
# accidentally break that by `pip install torchvision` (or by installing a
# torch/torchvision pair that doesn't match the CUDA/ABI in the runtime), which
# can lead to confusing import-time errors such as:
#   RuntimeError: operator torchvision::nms does not exist
#
# To make the package more robust, we treat torchvision as an optional dependency
# at import time and raise a clear error only when datasets are requested.
try:  # pragma: no cover
    from torchvision import datasets, transforms

    _TORCHVISION_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    datasets = None  # type: ignore
    transforms = None  # type: ignore
    _TORCHVISION_IMPORT_ERROR = e


def _require_torchvision() -> None:
    if datasets is None or transforms is None:
        raise ImportError(
            "torchvision failed to import. This harness needs torchvision for datasets/transforms. "
            "If you're in Colab, avoid reinstalling torchvision unless you also install a matching torch build. "
            f"Original error: {_TORCHVISION_IMPORT_ERROR!r}"
        )


class IndexedSubset(Dataset):
    """A subset wrapper that returns (x, y, idx) where idx is *local* in [0, len-1]."""

    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        return x, y, i


@dataclass
class DataBundle:
    train: Dataset
    eval: Dataset


def _cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_tf, eval_tf


def _mnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return tf, tf


def get_datasets(dataset: str, data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    dataset = dataset.lower()

    # Fail fast with a clear message if torchvision is broken/missing.
    _require_torchvision()

    if dataset == "cifar10":
        train_tf, eval_tf = _cifar10_transforms()
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)
        return train, test

    if dataset == "mnist":
        train_tf, eval_tf = _mnist_transforms()
        train = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_tf)
        return train, test

    if dataset == "fashionmnist" or dataset == "fashion-mnist":
        train_tf, eval_tf = _mnist_transforms()
        train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_tf)
        test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=eval_tf)
        return train, test

    raise ValueError(f"Unknown dataset: {dataset}")


def choose_subset_indices(n: int, subset_size: Optional[int], seed: int) -> List[int]:
    if subset_size is None or subset_size >= n:
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    idx = perm[: subset_size].tolist()
    idx.sort()
    return idx


def build_data(cfg, seed: int, data_dir: str = "./data") -> DataBundle:
    """Build train/eval datasets (possibly subsetted) with local indexing."""
    base_train, base_eval = get_datasets(cfg.dataset, data_dir=data_dir)

    train_idx = choose_subset_indices(len(base_train), cfg.subset_train, seed=seed + 11_111)
    eval_idx = choose_subset_indices(len(base_eval), cfg.subset_eval, seed=seed + 22_222)

    train = IndexedSubset(base_train, train_idx)
    eval_ds = IndexedSubset(base_eval, eval_idx)

    return DataBundle(train=train, eval=eval_ds)


def make_eval_loader(eval_ds: Dataset, batch_size: int, num_workers: int = 2, pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_train_loader(
    train_ds: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    *,
    indices: Optional[Sequence[int]] = None,
    weights: Optional[torch.Tensor] = None,
) -> DataLoader:
    """Create a train DataLoader.

    - If weights is provided, uses WeightedRandomSampler (with replacement).
    - Else if indices is provided, uses SubsetRandomSampler (without replacement each epoch).
    - Else uses shuffle=True.
    """

    if weights is not None:
        # weights should be 1D len == len(train_ds) or len(indices) if indices is set.
        if indices is not None:
            raise ValueError("Pass either indices or weights, not both (for simplicity in this harness).")
        sampler = WeightedRandomSampler(weights=weights.double().cpu(), num_samples=len(train_ds), replacement=True)
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if indices is not None:
        sampler = SubsetRandomSampler(list(map(int, indices)))
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )


def num_train_steps(cfg, n_train: int) -> int:
    if cfg.max_steps is not None:
        return int(cfg.max_steps)
    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    return int(cfg.epochs * steps_per_epoch)
