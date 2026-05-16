"""
Generate baseline FedAvg plots from completed experiments.

Reads metrics.csv and summary.json from experiments/results/ and produces:
1. Test accuracy vs round (curves with std band for alpha=0.1, 3 seeds)
2. Best accuracy vs alpha (bars with error bar for alpha=0.1)
3. Per-class accuracy comparison (alpha=0.1 vs alpha=100)
4. Fairness metrics across alpha values (std + fairness gap)

Fairness metrics are computed retroactively from the per-class accuracies
already stored in summary.json.

All plots saved to plots/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_run(run_name: str) -> dict:
    """Load metrics CSV and summary JSON for one run."""
    run_dir = RESULTS_DIR / run_name
    metrics = pd.read_csv(run_dir / "metrics.csv")
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    return {"metrics": metrics, "summary": summary, "name": run_name}


def add_fairness_to_summary(run: dict, num_classes: int = 10) -> dict:
    """
    Compute fairness metrics retroactively from per-class accuracies in summary.
    Mutates run["summary"] in-place and returns the run.

    Useful for older runs that pre-date the extended evaluation.
    """
    summary = run["summary"]
    # Check if already present (newer runs include them natively)
    if "final_fairness_gap" in summary:
        return run

    per_class = []
    for c in range(num_classes):
        key = f"final_acc_class_{c}"
        if key in summary:
            per_class.append(summary[key])
        else:
            return run  # cannot compute, missing data

    arr = np.array(per_class)
    summary["final_macro_acc"] = float(arr.mean())
    summary["final_std_acc"] = float(arr.std())
    summary["final_min_acc"] = float(arr.min())
    summary["final_max_acc"] = float(arr.max())
    summary["final_fairness_gap"] = float(arr.max() - arr.min())
    return run


def plot_convergence_by_alpha(runs_alpha01, run_alpha1, run_alpha100):
    """Plot 1: test accuracy vs round."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    accs_per_seed = np.stack([r["metrics"]["test_acc"].values for r in runs_alpha01])
    rounds = runs_alpha01[0]["metrics"]["round"].values
    mean_acc = accs_per_seed.mean(axis=0) * 100
    std_acc = accs_per_seed.std(axis=0) * 100

    ax.plot(rounds, mean_acc, label=r"$\alpha=0.1$ (strong non-IID, 3 seeds)",
            color="C3", linewidth=2)
    ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                    color="C3", alpha=0.2)

    ax.plot(run_alpha1["metrics"]["round"], run_alpha1["metrics"]["test_acc"] * 100,
            label=r"$\alpha=1.0$ (intermediate)", color="C1", linewidth=2)

    ax.plot(run_alpha100["metrics"]["round"], run_alpha100["metrics"]["test_acc"] * 100,
            label=r"$\alpha=100$ (quasi-IID, upper bound)", color="C2", linewidth=2)

    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("FedAvg convergence on CIFAR-10 with varying non-IID-ness", fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(5, 90)
    ax.set_xlim(0, 100)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_convergence_by_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_best_acc_vs_alpha(runs_alpha01, run_alpha1, run_alpha100):
    """Plot 2: best accuracy vs alpha, bar chart."""
    fig, ax = plt.subplots(figsize=(7, 5))

    alpha_labels = [r"$\alpha=0.1$" + "\n(strong\nnon-IID)",
                    r"$\alpha=1.0$" + "\n(intermediate)",
                    r"$\alpha=100$" + "\n(quasi-IID)"]
    xs = np.arange(3)

    bests_01 = [r["summary"]["best_test_acc"] * 100 for r in runs_alpha01]
    mean_01, std_01 = np.mean(bests_01), np.std(bests_01)

    means = [mean_01,
             run_alpha1["summary"]["best_test_acc"] * 100,
             run_alpha100["summary"]["best_test_acc"] * 100]
    stds = [std_01, 0.0, 0.0]
    colors = ["C3", "C1", "C2"]

    ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85,
           edgecolor="black", linewidth=1.2, capsize=6)

    for x, m, s in zip(xs, means, stds):
        label = f"{m:.1f} ± {s:.1f}%" if s > 0 else f"{m:.1f}%"
        ax.text(x, m + 1.5, label, ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("Best test accuracy (%)", fontsize=12)
    ax.set_title("Effect of non-IID-ness on FedAvg final accuracy", fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_best_acc_vs_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_comparison(run_alpha01, run_alpha100):
    """Plot 3: per-class accuracy alpha=0.1 vs alpha=100."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.38

    accs_01 = [run_alpha01["summary"][f"final_acc_class_{c}"] * 100 for c in range(n_classes)]
    accs_100 = [run_alpha100["summary"][f"final_acc_class_{c}"] * 100 for c in range(n_classes)]

    ax.bar(x - width/2, accs_01, width, label=r"$\alpha=0.1$ (non-IID)",
           color="C3", alpha=0.85, edgecolor="black", linewidth=1)
    ax.bar(x + width/2, accs_100, width, label=r"$\alpha=100$ (quasi-IID)",
           color="C2", alpha=0.85, edgecolor="black", linewidth=1)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("FedAvg per-class accuracy: client drift is unequal across classes",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_per_class_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_fairness_metrics(runs_alpha01, run_alpha1, run_alpha100):
    """Plot 4: fairness metrics (std and gap) across alpha values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    alpha_labels = [r"$\alpha=0.1$" + "\n(non-IID)",
                    r"$\alpha=1.0$",
                    r"$\alpha=100$" + "\n(quasi-IID)"]
    xs = np.arange(3)
    colors = ["C3", "C1", "C2"]

    # --- Left: std of per-class accuracy ---
    stds_01 = [r["summary"]["final_std_acc"] * 100 for r in runs_alpha01]
    means = [np.mean(stds_01),
             run_alpha1["summary"]["final_std_acc"] * 100,
             run_alpha100["summary"]["final_std_acc"] * 100]
    errs = [np.std(stds_01), 0, 0]

    axes[0].bar(xs, means, yerr=errs, color=colors,
                alpha=0.85, edgecolor="black", linewidth=1.2, capsize=6)
    for x, m, s in zip(xs, means, errs):
        label = f"{m:.1f} ± {s:.1f}%" if s > 0 else f"{m:.1f}%"
        axes[0].text(x, m + 0.4, label, ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(alpha_labels)
    axes[0].set_ylabel("Std of per-class accuracy (%)", fontsize=12)
    axes[0].set_title("Class-level heterogeneity\n(lower = more uniform)", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Right: fairness gap ---
    gaps_01 = [r["summary"]["final_fairness_gap"] * 100 for r in runs_alpha01]
    means = [np.mean(gaps_01),
             run_alpha1["summary"]["final_fairness_gap"] * 100,
             run_alpha100["summary"]["final_fairness_gap"] * 100]
    errs = [np.std(gaps_01), 0, 0]

    axes[1].bar(xs, means, yerr=errs, color=colors,
                alpha=0.85, edgecolor="black", linewidth=1.2, capsize=6)
    for x, m, s in zip(xs, means, errs):
        label = f"{m:.1f} ± {s:.1f} pp" if s > 0 else f"{m:.1f} pp"
        axes[1].text(x, m + 1, label, ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(alpha_labels)
    axes[1].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    axes[1].set_title("Class-level disparity\n(lower = more fair)", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("FedAvg: client drift produces class-level inequality",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_fairness_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("Generating FedAvg baseline plots")
    print("=" * 60)

    runs_alpha01 = [
        load_run("fedavg_alpha0.1_E2_seed42"),
        load_run("fedavg_alpha0.1_E2_seed43"),
        load_run("fedavg_alpha0.1_E2_seed44"),
    ]
    run_alpha1 = load_run("fedavg_alpha1.0_E2_seed42")
    run_alpha100 = load_run("fedavg_alpha100.0_E2_seed42")

    # Compute fairness metrics retroactively for runs that don't have them
    runs_alpha01 = [add_fairness_to_summary(r) for r in runs_alpha01]
    run_alpha1 = add_fairness_to_summary(run_alpha1)
    run_alpha100 = add_fairness_to_summary(run_alpha100)

    print("\nLoaded runs:")
    print(f"  alpha=0.1: {len(runs_alpha01)} seeds")
    print(f"  alpha=1.0: 1 seed")
    print(f"  alpha=100: 1 seed")

    bests_01 = [r["summary"]["best_test_acc"] * 100 for r in runs_alpha01]
    gaps_01 = [r["summary"]["final_fairness_gap"] * 100 for r in runs_alpha01]
    stds_01 = [r["summary"]["final_std_acc"] * 100 for r in runs_alpha01]

    print(f"\nFedAvg summary:")
    print(f"  alpha=0.1:  best acc = {np.mean(bests_01):.2f} ± {np.std(bests_01):.2f}%  "
          f"(seeds: {[f'{b:.1f}' for b in bests_01]})")
    print(f"  alpha=1.0:  best acc = {run_alpha1['summary']['best_test_acc']*100:.2f}%")
    print(f"  alpha=100:  best acc = {run_alpha100['summary']['best_test_acc']*100:.2f}%")
    print(f"  Drift gap (alpha=100 - alpha=0.1): "
          f"{run_alpha100['summary']['best_test_acc']*100 - np.mean(bests_01):.2f} pp")

    print(f"\nFairness summary (final, per-class):")
    print(f"  alpha=0.1: std={np.mean(stds_01):.2f}%, gap={np.mean(gaps_01):.2f} pp")
    print(f"  alpha=1.0: std={run_alpha1['summary']['final_std_acc']*100:.2f}%, "
          f"gap={run_alpha1['summary']['final_fairness_gap']*100:.2f} pp")
    print(f"  alpha=100: std={run_alpha100['summary']['final_std_acc']*100:.2f}%, "
          f"gap={run_alpha100['summary']['final_fairness_gap']*100:.2f} pp")

    print("\nGenerating plots...")
    plot_convergence_by_alpha(runs_alpha01, run_alpha1, run_alpha100)
    plot_best_acc_vs_alpha(runs_alpha01, run_alpha1, run_alpha100)
    plot_per_class_comparison(runs_alpha01[0], run_alpha100)
    plot_fairness_metrics(runs_alpha01, run_alpha1, run_alpha100)

    print("\n✓ Done. Plots are in plots/")


if __name__ == "__main__":
    main()