"""
Model evaluation utilities.

Provides:
- evaluate: lightweight evaluation (loss + accuracy) for round-by-round logging
- evaluate_per_class: per-class accuracy breakdown
- evaluate_extended: full metrics suite (accuracy, macro F1, per-class P/R/F1,
                     fairness metrics) for end-of-training evaluation
- compute_fairness_metrics: derive fairness stats from per-class accuracies

Metric philosophy:
- Accuracy: the standard FL metric; reported in every paper.
- Macro F1: complements accuracy on balanced test sets; penalizes models that
            ignore minority/difficult classes.
- Fairness metrics (std, min, max, gap): measure how uniformly the model
            performs across classes. In FL with non-IID data, class-level
            disparity is a direct signature of client drift — even when the
            average accuracy looks reasonable, some classes may be neglected.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    """
    Lightweight evaluation: returns (avg_loss, accuracy).
    Use this for per-round logging during training (called every round).

    Args:
        model: the network to evaluate
        dataloader: typically the global test set
        device: cuda or cpu

    Returns:
        avg_loss: average cross-entropy loss over the dataset
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
    Evaluate and break down accuracy per class. Kept for backward compatibility.
    For the full metric suite, prefer `evaluate_extended`.
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


def compute_fairness_metrics(per_class_accs: List[float]) -> Dict[str, float]:
    """
    Given a list of per-class accuracies, compute fairness metrics.

    These metrics quantify class-level disparity, which is a key signature
    of client drift in non-IID federated learning. The same average accuracy
    can hide very different per-class behaviour:

      Example: two models with 73% accuracy on CIFAR-10
        Model A: [73, 73, 73, ..., 73]       -> uniform, fair
        Model B: [94, 91, 47, 51, ..., 87]   -> unfair, some classes neglected
      → Both have the same accuracy, but Model B has high fairness_gap.

    Args:
        per_class_accs: list of 10 floats in [0, 1], one per class.

    Returns:
        dict with:
        - macro_acc:     mean of per-class accuracies (= macro recall)
        - std_acc:       std deviation across classes (higher = more disparity)
        - min_acc:       worst-class accuracy
        - max_acc:       best-class accuracy
        - fairness_gap:  max - min (the "spread")
    """
    arr = np.array(per_class_accs)
    return {
        "macro_acc": float(arr.mean()),
        "std_acc": float(arr.std()),
        "min_acc": float(arr.min()),
        "max_acc": float(arr.max()),
        "fairness_gap": float(arr.max() - arr.min()),
    }


@torch.no_grad()
def evaluate_extended(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_classes: int = 10,
) -> Dict[str, float]:
    """
    Full evaluation: accuracy + macro F1 + per-class precision/recall/F1 + fairness.

    Use this at the end of training (or sparsely during training) to get a
    rich picture of model behaviour, especially in non-IID settings.

    Returns:
        dict with keys:
        - loss, accuracy, macro_f1
        - precision_class_{c}, recall_class_{c}, f1_class_{c}, acc_class_{c}
        - macro_acc, std_acc, min_acc, max_acc, fairness_gap
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_samples += targets.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    accuracy = float(accuracy_score(all_targets, all_preds))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    # Per-class accuracy on a balanced test set is equivalent to recall
    per_class_acc = recall.tolist()

    macro_f1 = float(f1.mean())
    fairness = compute_fairness_metrics(per_class_acc)

    result = {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **{f"precision_class_{c}": float(precision[c]) for c in range(num_classes)},
        **{f"recall_class_{c}": float(recall[c]) for c in range(num_classes)},
        **{f"f1_class_{c}": float(f1[c]) for c in range(num_classes)},
        **{f"acc_class_{c}": float(per_class_acc[c]) for c in range(num_classes)},
        **fairness,
    }
    return result


if __name__ == "__main__":
    # Sanity check: full evaluation on an untrained model
    from models.resnet import build_resnet
    from data.cifar10 import get_cifar10_datasets, get_test_loader
    from utils.seed import set_seed

    print("=" * 60)
    print("Sanity check: evaluate_extended on untrained ResNet-20")
    print("Expected: accuracy ~10%, very high fairness_gap (random model)")
    print("=" * 60)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, testset = get_cifar10_datasets()
    test_loader = get_test_loader(testset, batch_size=256)
    model = build_resnet(depth=20, device=device)

    metrics = evaluate_extended(model, test_loader, device, num_classes=10)

    print(f"\nGlobal metrics:")
    print(f"  Loss:          {metrics['loss']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1:      {metrics['macro_f1']*100:.2f}%")

    print(f"\nFairness metrics:")
    print(f"  Macro acc:     {metrics['macro_acc']*100:.2f}%")
    print(f"  Std per-class: {metrics['std_acc']*100:.2f}%")
    print(f"  Min acc:       {metrics['min_acc']*100:.2f}%")
    print(f"  Max acc:       {metrics['max_acc']*100:.2f}%")
    print(f"  Fairness gap:  {metrics['fairness_gap']*100:.2f}%")

    print(f"\nPer-class breakdown:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"  {'Class':<12s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    for c, name in enumerate(class_names):
        print(f"  {name:<12s} "
              f"{metrics[f'acc_class_{c}']*100:6.2f}% "
              f"{metrics[f'precision_class_{c}']*100:6.2f}% "
              f"{metrics[f'recall_class_{c}']*100:6.2f}% "
              f"{metrics[f'f1_class_{c}']*100:6.2f}%")

    print(f"\n✓ Sanity check completed.")