"""
FedAvg: Federated Averaging (McMahan et al., 2017).

Standard implementation:
- Each round, the server distributes the global model to all selected clients.
- Each client trains locally for E epochs on its data with SGD.
- The server averages the resulting weights, weighted by local dataset size:
    theta_global = sum_k (n_k / n_total) * theta_k

This file defines two classes:
- Client: encapsulates a client's local dataset and training step.
- FedAvgServer: orchestrates rounds, distributes the model, aggregates weights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import math


def cosine_lr(round_idx: int, total_rounds: int, lr_max: float, lr_min: float = 0.0) -> float:
    """
    Cosine-annealed learning rate.

    Args:
        round_idx: current global round, 1-indexed (i.e. first round is 1, not 0).
        total_rounds: total number of global rounds R.
        lr_max: initial learning rate (used at round 1).
        lr_min: final learning rate (approached at round R).

    Returns:
        lr_r ∈ [lr_min, lr_max], decreasing along a cosine curve.

    Notes:
        - At round_idx=1, returns lr_max.
        - At round_idx=total_rounds, returns lr_min.
        - In between, follows lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(r-1)/(R-1))).
    """
    if total_rounds <= 1:
        return lr_max
    progress = (round_idx - 1) / (total_rounds - 1)
    progress = max(0.0, min(1.0, progress))  # clamp
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


class Client:
    """
    A federated learning client with a local dataset.

    Holds a reference to:
    - the dataset and the indices belonging to this client
    - the device for training
    - hyperparameters for local training (lr, batch_size, momentum, weight_decay)

    Each round, the server calls `local_update(global_state_dict)`:
    - the client copies the global model into a local copy
    - trains it for E epochs on its local data
    - returns (local_state_dict, num_samples) so the server can aggregate
    """

    def __init__(
            self,
            client_id: int,
            dataset: Dataset,
            indices: List[int],
            batch_size: int = 64,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        # Local DataLoader: only the slice of `dataset` indexed by `indices`
        self.subset = Subset(dataset, indices)
        self.loader = DataLoader(
            self.subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    @property
    def num_samples(self) -> int:
        """Number of training samples on this client."""
        return len(self.indices)

    def local_update(
            self,
            global_state_dict: Dict[str, torch.Tensor],
            model_factory,
            local_epochs: int = 2,
            lr_override: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Run local SGD training for `local_epochs` epochs starting from the
        global model weights.

        Args:
            global_state_dict: weights of the current global model
            model_factory: callable that returns a fresh model on the right device
                           (e.g. `lambda: build_resnet(depth=20, device=device)`)
            local_epochs: number of local training epochs

        Returns:
            (local_state_dict, num_samples, avg_train_loss)
        """
        # Build a local copy of the model and load global weights
        model = model_factory()
        model.load_state_dict(global_state_dict)
        model.train()

        # Set up local optimizer (SGD with momentum + weight decay)
        # Use the server-provided LR if available (cosine schedule across rounds),
        # otherwise fall back to the client's default LR (constant schedule).
        effective_lr = lr_override if lr_override is not None else self.lr
        optimizer = optim.SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Train for `local_epochs` epochs
        running_loss = 0.0
        running_samples = 0
        for _ in range(local_epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_samples += inputs.size(0)

        avg_loss = running_loss / max(running_samples, 1)

        # Return updated weights (CPU tensors -> reduces VRAM usage at server)
        local_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return local_state_dict, self.num_samples, avg_loss


class FedAvgServer:
    """
    The FedAvg server: holds the global model, orchestrates rounds,
    aggregates client updates with weighted averaging.
    """

    def __init__(
            self,
            global_model: nn.Module,
            clients: List[Client],
            device: torch.device = torch.device("cpu"),
    ):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the global model state_dict (used to send to clients)."""
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(
            self,
            client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> None:
        """
        Aggregate client weight updates into the global model.

        Performs WEIGHTED averaging: client k contributes proportionally to
        its dataset size (n_k / n_total).
        """
        # Compute total samples and per-client weights
        total_samples = sum(n for _, n in client_updates)
        weights = [n / total_samples for _, n in client_updates]

        # Initialize the aggregated state dict with zeros (same shapes as the first client)
        first_state = client_updates[0][0]
        aggregated = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in first_state.items()}

        # Weighted sum: aggregated[k] = sum_i (weights[i] * client_updates[i][0][k])
        for (state_dict, _), w in zip(client_updates, weights):
            for k, v in state_dict.items():
                aggregated[k] += w * v.float()

        # Cast back to original dtype where needed (e.g. integer counters in BN/GN)
        new_state = {}
        for k, v in aggregated.items():
            new_state[k] = v.to(first_state[k].dtype)

        # Load aggregated weights into the global model on its device
        self.global_model.load_state_dict({
            k: v.to(self.device) for k, v in new_state.items()
        })

    def run_round(
            self,
            round_idx: int,
            model_factory,
            local_epochs: int = 2,
            verbose: bool = True,
            lr_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Run one full FedAvg round:
        1. Get current global state.
        2. Each client trains locally on this state.
        3. Aggregate the updates with weighted averaging.

        Returns:
            dict with statistics about the round (avg client loss, etc.)
        """
        global_state = self.get_global_state()

        client_updates = []
        client_losses = []

        for client in self.clients:
            local_state, n_samples, avg_loss = client.local_update(
                global_state_dict=global_state,
                model_factory=model_factory,
                local_epochs=local_epochs,
                lr_override=lr_override,
            )
            client_updates.append((local_state, n_samples))
            client_losses.append(avg_loss)

            if verbose:
                print(f"  [Round {round_idx}] Client {client.client_id}: "
                      f"n={n_samples}, local_loss={avg_loss:.4f}")

        self.aggregate(client_updates)

        # Round statistics
        total_samples = sum(c.num_samples for c in self.clients)
        weighted_loss = sum(l * c.num_samples for l, c in zip(client_losses, self.clients)) / total_samples

        return {
            "round": round_idx,
            "avg_client_loss": float(sum(client_losses) / len(client_losses)),
            "weighted_client_loss": float(weighted_loss),
            "lr": float(lr_override) if lr_override is not None else float("nan"),
        }


if __name__ == "__main__":
    # Sanity check: 2 clients on tiny IID partition, 3 rounds, expect loss to go down.
    from models.resnet import build_resnet
    from data.cifar10 import get_cifar10_datasets, get_test_loader
    from data.partition import iid_partition
    from utils.seed import set_seed
    from utils.eval import evaluate

    print("=" * 60)
    print("Sanity check: FedAvg with 2 clients, 3 rounds, tiny partition")
    print("Expected: loss decreases, accuracy goes from ~10% to ~20-30%")
    print("=" * 60)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data, take a small subset for the sanity check
    trainset, testset = get_cifar10_datasets(augment_train=False)
    test_loader = get_test_loader(testset, batch_size=256)

    # Take only the first 2000 samples for speed; partition into 2 clients
    from torch.utils.data import Subset
    small_train = Subset(trainset, list(range(2000)))
    # Wrap small_train so it has .targets like the original (needed by partition)
    # Actually we use iid_partition on the small Subset using its dataset attr
    small_train.targets = [trainset.targets[i] for i in range(2000)]

    client_indices = iid_partition(small_train, num_clients=2, seed=42)

    # Build clients
    clients = [
        Client(
            client_id=k,
            dataset=small_train,
            indices=idx,
            batch_size=64,
            lr=0.01,
            device=device,
        )
        for k, idx in enumerate(client_indices)
    ]

    # Build global model + server
    global_model = build_resnet(depth=20, device=device)
    server = FedAvgServer(global_model=global_model, clients=clients, device=device)

    # Helper to build fresh local models
    model_factory = lambda: build_resnet(depth=20, device=device)

    # Initial evaluation
    loss0, acc0 = evaluate(global_model, test_loader, device)
    print(f"\n[Round 0] Initial: loss={loss0:.4f}, acc={acc0*100:.2f}%")

    # Run 3 rounds
    for r in range(1, 4):
        stats = server.run_round(round_idx=r, model_factory=model_factory, local_epochs=1)
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"[Round {r}] Test: loss={loss:.4f}, acc={acc*100:.2f}% | "
              f"avg client loss={stats['avg_client_loss']:.4f}")

    print(f"\n✓ Sanity check completed.")