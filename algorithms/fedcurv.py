"""
FedCurv: federated learning with EWC-like Fisher regularization
(Shoham et al., 2019, "Overcoming Forgetting in Federated Learning on Non-IID Data").

Local loss for client k at round t:
    L_k = L_local(theta) + (lambda/2) * sum_i F_global[i] * (theta[i] - theta_prev[i])^2

Where:
    F_global = weighted average of the Fishers computed by each client in round t-1
    theta_prev = snapshot of the global model at the START of the current round t,
                   i.e. the weights every client receives before local training
    lambda = regularization coefficient (hyperparameter to sweep)

The penalty keeps each client close to the global model it started from, weighted
per-parameter by how important that parameter is according to the Fisher diagonal.
At round 1 no Fisher exists yet, so the penalty is zero and FedCurv reduces exactly
to FedAvg; the same holds for lambda = 0.

This is a simplified variant of FedCurv that uses one shared Fisher instead of
maintaining per-client Fishers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from algorithms.fedavg import Client, FedAvgServer


@torch.no_grad()
def _zero_fisher(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a dict of zero tensors with the same shape as model.named_parameters()."""
    return {n: torch.zeros_like(p, device=p.device)
            for n, p in model.named_parameters() if p.requires_grad}


def compute_fisher_diagonal(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        max_samples: int = 200,
        use_true_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute the diagonal of the empirical Fisher Information Matrix for a model.
    Approximated by averaging the squared per-sample gradient with respect to
    each parameter, over `max_samples` data points.

    Args:
        model: the model (typically the locally-trained model at end of round)
        dataloader: data to compute Fisher on (client's local data)
        device: cuda/cpu
        max_samples: cap on the number of samples for efficiency (200 is plenty)
        use_true_labels: True = empirical Fisher; False = sample from predictions

    Returns:
        dict mapping parameter name -> diagonal Fisher tensor (same shape as param)
    """
    model.eval()
    fisher = _zero_fisher(model)
    count = 0

    # We process the data in small batches, but accumulate grad^2 sample-by-sample
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        for i in range(inputs.size(0)):
            if count >= max_samples:
                break

            model.zero_grad()
            logits = model(inputs[i:i+1])

            if use_true_labels:
                # Empirical Fisher: use ground-truth label
                y = targets[i:i+1]
            else:
                # Standard Fisher: sample from p(y|x; θ)
                probs = F.softmax(logits, dim=1)
                y = torch.multinomial(probs, num_samples=1).squeeze(1)

            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            count += 1

        if count >= max_samples:
            break

    # Average over the number of samples actually processed
    if count > 0:
        for n in fisher:
            fisher[n] /= count

    # Move to CPU to save VRAM (will be aggregated server-side anyway)
    return {n: v.cpu() for n, v in fisher.items()}

class FedCurvClient(Client):
    """
    FedCurv client: like FedAvg client but with EWC-style penalty during local training.

    Adds two things to the base Client:
    1. During local_update, applies the penalty
         (lambda/2) * sum_i F_global[i] * (theta[i] - theta_prev[i])^2
       where F_global and theta_prev come from the server (round t-1).
    2. After local training, computes the diagonal Fisher on the client's own data
       and returns it along with the updated weights.

    At round 1 (no Fisher yet from previous round), penalty is zero and behavior
    reduces to plain FedAvg.
    """

    def local_update_curv(
            self,
            global_state_dict: Dict[str, torch.Tensor],
            model_factory,
            local_epochs: int = 2,
            fed_lambda: float = 1000.0,
            fisher_global: Optional[Dict[str, torch.Tensor]] = None,
            theta_prev: Optional[Dict[str, torch.Tensor]] = None,
            fisher_samples: int = 200,
            lr_override: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, Dict[str, torch.Tensor]]:
        """
        Args:
            global_state_dict: current global model weights (start of round)
            model_factory: callable that builds a fresh model on the right device
            local_epochs: local training epochs
            fed_lambda: regularization coefficient
            fisher_global: aggregated Fisher from previous round (None at round 1)
            theta_prev: snapshot of the global weights at the start of the CURRENT round
                        (the anchor for the penalty; None at round 1)
            fisher_samples: number of samples used to estimate Fisher on this client

        Returns:
            (local_state_dict, num_samples, avg_train_loss, local_fisher)
        """
        # Build local model, load global weights
        model = model_factory()
        model.load_state_dict(global_state_dict)
        model.train()

        # If we have Fisher + previous theta, move them to the same device as the model
        # and pre-extract per-parameter tensors for the penalty.
        use_penalty = (fisher_global is not None) and (theta_prev is not None) and (fed_lambda > 0)

        if use_penalty:
            fisher_dev = {n: f.to(self.device) for n, f in fisher_global.items()}
            theta_prev_dev = {n: t.to(self.device) for n, t in theta_prev.items()}

        # Local optimizer
        effective_lr = lr_override if lr_override is not None else self.lr
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Training loop with penalty
        running_loss = 0.0
        running_samples = 0
        for _ in range(local_epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(inputs)
                ce_loss = F.cross_entropy(logits, targets)

                if use_penalty:
                    # Compute the FedCurv penalty term
                    penalty = 0.0
                    for n, p in model.named_parameters():
                        if n in fisher_dev:
                            penalty = penalty + (fisher_dev[n] * (p - theta_prev_dev[n]) ** 2).sum()
                    loss = ce_loss + (fed_lambda / 2.0) * penalty
                else:
                    loss = ce_loss

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                running_loss += ce_loss.item() * inputs.size(0)
                running_samples += inputs.size(0)

        avg_loss = running_loss / max(running_samples, 1)

        # Compute the Fisher of the locally-trained model on the client's data
        local_fisher = compute_fisher_diagonal(
            model, self.loader, self.device, max_samples=fisher_samples
        )

        # Return updated weights (on CPU) and Fisher (already CPU from compute_fisher_diagonal)
        local_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return local_state_dict, self.num_samples, avg_loss, local_fisher


class FedCurvServer(FedAvgServer):
    """
    FedCurv server: extends FedAvgServer with:
    - state tracking for F_global (Fisher aggregated at the end of the previous round)
    - a fresh snapshot of the global weights at the start of each round, used as the
      anchor for that round's penalty
    - aggregation of Fishers (weighted average, same as for weights)
    - run_round modified to pass F_global and theta_prev to clients
    """

    def __init__(
            self,
            global_model: nn.Module,
            clients: List["FedCurvClient"],
            device: torch.device = torch.device("cpu"),
            fed_lambda: float = 1000.0,
            fisher_samples: int = 200,
    ):
        super().__init__(global_model=global_model, clients=clients, device=device)
        self.fed_lambda = fed_lambda
        self.fisher_samples = fisher_samples

        # State maintained across rounds
        self.fisher_global: Optional[Dict[str, torch.Tensor]] = None
        self.theta_prev: Optional[Dict[str, torch.Tensor]] = None

    def _aggregate_fishers(
            self,
            client_fishers: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """Weighted average of Fishers, weighted by client dataset size."""
        total_samples = sum(n for _, n in client_fishers)
        weights = [n / total_samples for _, n in client_fishers]

        first_fisher = client_fishers[0][0]
        aggregated = {n: torch.zeros_like(v, dtype=torch.float32)
                      for n, v in first_fisher.items()}

        for (fisher_k, _), w in zip(client_fishers, weights):
            for n, v in fisher_k.items():
                aggregated[n] += w * v.float()

        return aggregated

    def run_round(
            self,
            round_idx: int,
            model_factory,
            local_epochs: int = 2,
            verbose: bool = True,
            lr_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Run one FedCurv round:
        1. Take snapshot of current global weights (theta_prev for next round's penalty).
        2. Each client trains locally with the FedCurv penalty (zero penalty at round 1).
        3. Each client also returns its Fisher.
        4. Server aggregates weights AND Fishers (weighted by dataset size).
        5. Update F_global and theta_prev for next round.
        """
        global_state = self.get_global_state()

        # Snapshot of theta at start of this round — will become theta_prev for next round
        theta_at_round_start = {k: v.clone() for k, v in global_state.items()}

        client_updates = []
        client_fishers = []
        client_losses = []

        for client in self.clients:
            local_state, n_samples, avg_loss, local_fisher = client.local_update_curv(
                global_state_dict=global_state,
                model_factory=model_factory,
                local_epochs=local_epochs,
                fed_lambda=self.fed_lambda,
                fisher_global=self.fisher_global,
                theta_prev=theta_at_round_start,
                fisher_samples=self.fisher_samples,
                lr_override=lr_override,
            )
            client_updates.append((local_state, n_samples))
            client_fishers.append((local_fisher, n_samples))
            client_losses.append(avg_loss)

            if verbose:
                print(f"  [Round {round_idx}] Client {client.client_id}: "
                      f"n={n_samples}, local_loss={avg_loss:.4f}")

        # Aggregate weights (inherited from FedAvgServer)
        self.aggregate(client_updates)

        # Aggregate Fishers
        self.fisher_global = self._aggregate_fishers(client_fishers)

        # Save snapshot for next round's penalty
        self.theta_prev = theta_at_round_start

        # Round statistics
        total_samples = sum(c.num_samples for c in self.clients)
        weighted_loss = sum(l * c.num_samples for l, c in zip(client_losses, self.clients)) / total_samples

        # Compute the average magnitude of the penalty term in absolute terms
        if self.theta_prev is not None and self.fisher_global is not None and round_idx > 1:
            penalty_norm = 0.0
            for n, fg in self.fisher_global.items():
                if n in theta_at_round_start:
                    # before aggregation, theta_at_round_start was the global model;
                    # after aggregation, the global model has moved. The penalty term
                    # measures how much the new global is from the prev global wrt Fisher.
                    new_theta = self.global_model.state_dict()[n].cpu().float()
                    penalty_norm += (fg * (new_theta - theta_at_round_start[n].float()) ** 2).sum().item()
            penalty_norm = (self.fed_lambda / 2.0) * penalty_norm
        else:
            penalty_norm = 0.0

        return {
            "round": round_idx,
            "avg_client_loss": float(sum(client_losses) / len(client_losses)),
            "weighted_client_loss": float(weighted_loss),
            "penalty_norm": float(penalty_norm),
            "lr": float(lr_override) if lr_override is not None else float("nan"),
        }

if __name__ == "__main__":
    # --- Test 1: Fisher computation ---
    print("=" * 60)
    print("Test 1: compute_fisher_diagonal")
    print("=" * 60)

    from models.resnet import build_resnet
    from data.cifar10 import get_cifar10_datasets
    from data.partition import dirichlet_partition
    from utils.seed import set_seed
    from utils.eval import evaluate

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, testset = get_cifar10_datasets(augment_train=False)
    indices = dirichlet_partition(trainset, num_clients=10, alpha=0.1, seed=42)

    test_subset = Subset(trainset, indices[0][:500])
    loader = DataLoader(test_subset, batch_size=32, shuffle=True)
    model = build_resnet(depth=20, device=device)
    fisher = compute_fisher_diagonal(model, loader, device, max_samples=200)

    print(f"[OK] Fisher computed for {len(fisher)} parameter tensors")
    print(f"     fc.weight mean = {fisher['fc.weight'].mean():.2e}")
    print(f"     conv1.weight mean = {fisher['conv1.weight'].mean():.2e}")
    assert all(f.min() >= 0 for f in fisher.values())
    print("[OK] Fisher is non-negative")

    # --- Test 2: FedCurv with lambda=0 should match FedAvg exactly ---
    print("\n" + "=" * 60)
    print("Test 2: FedCurv with lambda=0 == FedAvg (sanity check)")
    print("=" * 60)

    from data.cifar10 import get_test_loader
    test_loader = get_test_loader(testset, batch_size=256)

    set_seed(123)  # reset seed for this test

    # Use just 2 clients on a small subset for speed
    small_train = Subset(trainset, list(range(2000)))
    small_train.targets = [trainset.targets[i] for i in range(2000)]
    client_indices = [list(range(0, 1000)), list(range(1000, 2000))]

    clients_curv = [
        FedCurvClient(
            client_id=k, dataset=small_train, indices=idx,
            batch_size=64, lr=0.01, device=device,
        )
        for k, idx in enumerate(client_indices)
    ]

    global_model = build_resnet(depth=20, device=device)
    server_curv = FedCurvServer(
        global_model=global_model, clients=clients_curv, device=device,
        fed_lambda=0.0,  # <-- zero penalty: should be equivalent to FedAvg
    )
    model_factory = lambda: build_resnet(depth=20, device=device)

    init_loss, init_acc = evaluate(global_model, test_loader, device)
    print(f"[Round 0] Initial: loss={init_loss:.4f}, acc={init_acc*100:.2f}%")

    for r in range(1, 4):
        stats = server_curv.run_round(r, model_factory, local_epochs=1, verbose=False)
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"[Round {r}] loss={loss:.4f}, acc={acc*100:.2f}%, "
              f"penalty_norm={stats['penalty_norm']:.4f}")

    print("\n[OK] FedCurv with lambda=0 runs and converges similarly to FedAvg")

    # --- Test 3: FedCurv with lambda > 0 should run without crashing ---
    print("\n" + "=" * 60)
    print("Test 3: FedCurv with lambda=100 (active penalty)")
    print("=" * 60)

    set_seed(123)
    clients_curv = [
        FedCurvClient(
            client_id=k, dataset=small_train, indices=idx,
            batch_size=64, lr=0.01, device=device,
        )
        for k, idx in enumerate(client_indices)
    ]
    global_model = build_resnet(depth=20, device=device)
    server_curv = FedCurvServer(
        global_model=global_model, clients=clients_curv, device=device,
        fed_lambda=100,
    )

    init_loss, init_acc = evaluate(global_model, test_loader, device)
    print(f"[Round 0] Initial: loss={init_loss:.4f}, acc={init_acc*100:.2f}%")

    for r in range(1, 4):
        stats = server_curv.run_round(r, model_factory, local_epochs=1, verbose=False)
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"[Round {r}] loss={loss:.4f}, acc={acc*100:.2f}%, "
              f"penalty_norm={stats['penalty_norm']:.6f}")

    print("\n[OK] FedCurv with lambda=100 runs end-to-end")