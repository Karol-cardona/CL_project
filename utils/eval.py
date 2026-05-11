"""
Model evaluation utilities.

Provides:
- evaluate: compute test loss and accuracy of a model on a DataLoader.
- evaluate_per_class: also break down accuracy by class (useful for non-IID
  analysis: which classes is the global model good/bad at?).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a model on a DataLoader: returns (avg_loss, accuracy).

    Args:
        model: the network to evaluate (will be set to eval mode)
        dataloader: typically the global test set
        device: cuda or cpu

    Returns:
        avg_loss: average cross-entropy loss over the entire dataset
        accuracy: overall accuracy in [0, 1]
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")

        # accumulate (sum over batch, divide at the end)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_per_class(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_classes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate the model and also break down accuracy per class.
    Useful in non-IID FL: shows whether the global model is good on all
    classes or only on a subset (which would indicate failed knowledge fusion).

    Returns:
        dict with keys: "loss", "accuracy", "acc_class_0", ..., "acc_class_{N-1}"
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    correct_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    total_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_samples += targets.size(0)

        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            mask = targets == c
            correct_per_class[c] += (preds[mask] == c).sum()
            total_per_class[c] += mask.sum()

    avg_loss = total_loss / total_samples
    overall_acc = correct_per_class.sum().item() / total_samples

    result = {"loss": avg_loss, "accuracy": overall_acc}
    for c in range(num_classes):
        if total_per_class[c] > 0:
            result[f"acc_class_{c}"] = (correct_per_class[c].float() / total_per_class[c].float()).item()
        else:
            result[f"acc_class_{c}"] = float("nan")
    return result


if __name__ == "__main__":
    # Sanity check: build a model, evaluate on test set, expect ~10% (random guessing)
    from models.resnet import build_resnet
    from data.cifar10 import get_cifar10_datasets, get_test_loader
    from utils.seed import set_seed

    print("=" * 60)
    print("Sanity check: evaluate untrained ResNet-20 on CIFAR-10 test set")
    print("Expected accuracy: ~10% (random guessing on 10 classes)")
    print("=" * 60)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test set
    _, testset = get_cifar10_datasets()
    test_loader = get_test_loader(testset, batch_size=256)

    # Build untrained model
    model = build_resnet(depth=20, device=device)

    # Verify the model is on the correct device
    print(f"\n[DEBUG] Model device: {next(model.parameters()).device}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[DEBUG] Test set size: {len(testset)}")
    print(f"[DEBUG] DataLoader num_workers: {test_loader.num_workers}")

    # Evaluate
    loss, acc = evaluate(model, test_loader, device)
    print(f"\nUntrained model:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc * 100:.2f}%")

    # Per-class breakdown
    metrics = evaluate_per_class(model, test_loader, device, num_classes=10)
    print(f"\nPer-class accuracy:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    for c, name in enumerate(class_names):
        print(f"  {name:<12s}: {metrics[f'acc_class_{c}'] * 100:5.2f}%")

    print(f"\n✓ Sanity check completed.")
    if 0.05 <= acc <= 0.20:
        print(f"  Untrained model accuracy ({acc*100:.1f}%) is in the expected ~10% range.")
    else:
        print(f"  ⚠ Warning: untrained accuracy {acc*100:.1f}% is unusual.")