"""
CIFAR-10 dataset loading.

Provides:
- get_cifar10_datasets: returns (trainset, testset) with standard transforms
- get_test_loader: returns a DataLoader for the global test set
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


# CIFAR-10 statistics computed on the training set (standard values from literature)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_datasets(
        data_root: str = "./data/CIFAR10",
        augment_train: bool = True,
) -> Tuple[Dataset, Dataset]:
    """
    Load CIFAR-10 train and test sets with standard normalization.

    Args:
        data_root: where to download / find CIFAR-10
        augment_train: whether to apply random crop + flip on the training set

    Returns:
        (trainset, testset) torchvision datasets
    """
    # Test transform: only ToTensor + Normalize (no augmentation at test time)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Train transform: optional augmentation (random crop + horizontal flip),
    # then ToTensor + Normalize
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = test_transform

    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,           # ← TRAIN set, fix del bug del repo originale
        download=True,
        transform=train_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return trainset, testset


def get_test_loader(
        testset: Dataset,
        batch_size: int = 256,
        num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the global test set.

    Args:
        testset: the test dataset
        batch_size: batch size for evaluation (can be larger than training)
        num_workers: number of worker processes (use 0 on Windows to avoid issues)

    Returns:
        DataLoader for evaluation
    """
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,         # at test time we don't shuffle
        num_workers=num_workers,
        pin_memory=True,       # faster GPU transfer
    )


if __name__ == "__main__":
    # Quick sanity check: load the dataset and print some info
    trainset, testset = get_cifar10_datasets()
    print(f"Train set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Number of classes: {len(trainset.classes)}")
    print(f"Classes: {trainset.classes}")

    # Get one sample
    img, label = trainset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {img.shape}")    # should be [3, 32, 32]
    print(f"  Image dtype: {img.dtype}")
    print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")  # normalized
    print(f"  Label: {label} ({trainset.classes[label]})")