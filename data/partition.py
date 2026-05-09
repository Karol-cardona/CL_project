"""
Data partitioning across clients.

Provides:
- iid_partition: random uniform partition (homogeneous clients)
- dirichlet_partition: Dirichlet-based label-skew partition (non-IID)
- visualize_partition: plot the class distribution per client
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional
from data.cifar10 import get_cifar10_datasets

def iid_partition(
        dataset: Dataset,
        num_clients: int,
        seed: int = 42,
) -> List[List[int]]:
    """
    Partition the dataset uniformly at random among clients.
    Each client gets a roughly equal-sized random subset.

    Args:
        dataset: torchvision dataset (must support len())
        num_clients: number of clients
        seed: random seed for reproducibility

    Returns:
        List of length num_clients, where each element is a list of indices
        belonging to that client.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n)
    # split as evenly as possible (np.array_split handles non-divisible cases)
    splits = np.array_split(indices, num_clients)
    return [s.tolist() for s in splits]


def dirichlet_partition(
        dataset: Dataset,
        num_clients: int,
        alpha: float,
        seed: int = 42,
        min_size_per_client: int = 10,
) -> List[List[int]]:
    """
    Partition the dataset across clients using a Dirichlet distribution
    on label proportions (label-skew non-IID).

    For each class c, we sample a vector p_c ~ Dirichlet(alpha) of length
    num_clients, then distribute the samples of class c according to p_c.

    Small alpha (e.g. 0.1) -> highly non-IID (each client sees few classes).
    Large alpha (e.g. 100) -> nearly IID.

    Args:
        dataset: torchvision dataset; must expose .targets (list/array of labels)
        num_clients: number of clients
        alpha: Dirichlet concentration parameter (>0)
        seed: random seed for reproducibility
        min_size_per_client: minimum samples per client (we resample if violated)

    Returns:
        List of length num_clients, each is a list of indices for that client.
    """
    rng = np.random.default_rng(seed)

    # Get labels as a numpy array
    targets = np.array(dataset.targets)
    num_classes = int(targets.max() + 1)

    # We may need to resample if some client gets too few samples
    # (this can happen with very small alpha)
    while True:
        # client_indices[k] will be the list of indices assigned to client k
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            # All sample indices belonging to class c
            class_idx = np.where(targets == c)[0]
            rng.shuffle(class_idx)

            # Sample the proportions of class c across the num_clients clients
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)

            # Convert proportions into split points (cumulative sum, then int)
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]

            # Split the class indices according to the sampled proportions
            class_splits = np.split(class_idx, split_points)
            for k in range(num_clients):
                client_indices[k].extend(class_splits[k].tolist())

        # Check minimum size constraint
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size_per_client:
            break
        # else: try again with a different sample (loop)

    # Shuffle the final indices for each client (so that order is not class-sorted)
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices


def get_partition_stats(
        client_indices: List[List[int]],
        targets: np.ndarray,
        num_classes: int,
) -> np.ndarray:
    """
    Compute the class-count matrix: rows are clients, cols are classes.

    Returns:
        Array of shape (num_clients, num_classes), where entry [k, c]
        is the number of samples of class c assigned to client k.
    """
    num_clients = len(client_indices)
    stats = np.zeros((num_clients, num_classes), dtype=int)
    for k, idx in enumerate(client_indices):
        labels_k = targets[idx]
        for c in range(num_classes):
            stats[k, c] = (labels_k == c).sum()
    return stats


def visualize_partition(
        client_indices: List[List[int]],
        dataset: Dataset,
        title: str = "Client class distribution",
        save_path: Optional[str] = None,
) -> None:
    """
    Plot a stacked bar chart showing the class distribution of each client.

    Args:
        client_indices: output of iid_partition or dirichlet_partition
        dataset: the underlying dataset (must expose .targets and .classes)
        title: plot title
        save_path: if given, save the plot to this file (PNG/PDF). Else, show it.
    """
    import matplotlib.pyplot as plt

    targets = np.array(dataset.targets)
    num_classes = int(targets.max() + 1)
    class_names = dataset.classes if hasattr(dataset, "classes") else [str(i) for i in range(num_classes)]

    stats = get_partition_stats(client_indices, targets, num_classes)
    num_clients = len(client_indices)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(num_clients)
    cmap = plt.get_cmap("tab10")

    for c in range(num_classes):
        ax.bar(
            range(num_clients),
            stats[:, c],
            bottom=bottom,
            label=class_names[c],
            color=cmap(c),
        )
        bottom += stats[:, c]

    ax.set_xlabel("Client")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xticks(range(num_clients))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved partition plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    trainset, _ = get_cifar10_datasets(augment_train=False)

    print("=" * 60)
    print("IID partition")
    print("=" * 60)
    iid_idx = iid_partition(trainset, num_clients=10, seed=42)
    for k, idx in enumerate(iid_idx):
        print(f"  Client {k}: {len(idx)} samples")
    print(f"  Total: {sum(len(idx) for idx in iid_idx)} samples")

    print()
    print("=" * 60)
    print("Dirichlet partition (alpha=0.1, highly non-IID)")
    print("=" * 60)
    dir_idx_01 = dirichlet_partition(trainset, num_clients=10, alpha=0.1, seed=42)
    targets = np.array(trainset.targets)
    stats = get_partition_stats(dir_idx_01, targets, num_classes=10)
    for k in range(10):
        print(f"  Client {k}: {len(dir_idx_01[k])} samples, "
              f"class counts: {stats[k].tolist()}")

    print()
    print("=" * 60)
    print("Dirichlet partition (alpha=100, almost IID)")
    print("=" * 60)
    dir_idx_100 = dirichlet_partition(trainset, num_clients=10, alpha=100.0, seed=42)
    stats = get_partition_stats(dir_idx_100, targets, num_classes=10)
    for k in range(10):
        print(f"  Client {k}: {len(dir_idx_100[k])} samples, "
              f"class counts: {stats[k].tolist()}")

    # Save visualizations
    print()
    print("Saving visualizations...")
    visualize_partition(iid_idx, trainset, title="IID partition (random)",
                        save_path="../plots/partition_iid.png")
    visualize_partition(dir_idx_01, trainset, title="Dirichlet partition (alpha=0.1)",
                        save_path="../plots/partition_dirichlet_alpha0.1.png")
    visualize_partition(dir_idx_100, trainset, title="Dirichlet partition (alpha=100)",
                        save_path="../plots/partition_dirichlet_alpha100.png")