"""
Generate FedAvg baseline plots from completed experiments.

Reads metrics.csv and summary.json from experiments/results/ and produces
8 plots covering:
- Ablation 1: effect of non-IID-ness (alpha)
- Ablation 2: effect of local epochs (E)
- Diagnostic views: fairness and acc–F1 gap evolution in time

Plots produced:
  1. fedavg_convergence_by_alpha.png       — acc + macro F1 vs round, by alpha
  2. fedavg_best_acc_vs_alpha.png          — best acc + macro F1 by alpha (bars)
  3. fedavg_per_class_comparison.png       — per-class acc, alpha=0.1 vs 1.0 vs 100
  4. fedavg_fairness_metrics.png           — std + gap per alpha (bars)
  5. fedavg_convergence_by_E.png           — acc + macro F1 vs round, by E
  6. fedavg_tradeoff_E.png                 — best acc + gap by E (bars)
  7. fedavg_fairness_gap_vs_round.png      — fairness_gap evolution by alpha
  8. fedavg_acc_f1_gap_vs_round.png        — (acc - macro F1) evolution by alpha

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


# ============================================================
# Data loading
# ============================================================

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
    Mutates run["summary"] in-place. Useful for runs predating extended eval.
    """
    summary = run["summary"]
    if "final_fairness_gap" in summary:
        return run
    per_class = []
    for c in range(num_classes):
        key = f"final_acc_class_{c}"
        if key in summary:
            per_class.append(summary[key])
        else:
            return run
    arr = np.array(per_class)
    summary["final_macro_acc"] = float(arr.mean())
    summary["final_std_acc"] = float(arr.std())
    summary["final_min_acc"] = float(arr.min())
    summary["final_max_acc"] = float(arr.max())
    summary["final_fairness_gap"] = float(arr.max() - arr.min())
    return run


# ============================================================
# PLOT 1 — Convergence by alpha (accuracy + macro F1)
# ============================================================

def plot_convergence_by_alpha(runs_alpha01, run_alpha1, run_alpha100):
    """Test accuracy AND macro F1 vs round. Two side-by-side panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    rounds = runs_alpha01[0]["metrics"]["round"].values

    # --- Left: accuracy ---
    accs_per_seed = np.stack([r["metrics"]["test_acc"].values for r in runs_alpha01])
    mean_acc = accs_per_seed.mean(axis=0) * 100
    std_acc = accs_per_seed.std(axis=0) * 100

    axes[0].plot(rounds, mean_acc, label=r"$\alpha=0.1$ (non-IID, 3 seeds)",
                 color="C3", linewidth=2)
    axes[0].fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                         color="C3", alpha=0.2)
    axes[0].plot(run_alpha1["metrics"]["round"], run_alpha1["metrics"]["test_acc"] * 100,
                 label=r"$\alpha=1.0$ (intermediate)", color="C1", linewidth=2)
    axes[0].plot(run_alpha100["metrics"]["round"], run_alpha100["metrics"]["test_acc"] * 100,
                 label=r"$\alpha=100$ (quasi-IID)", color="C2", linewidth=2)

    axes[0].set_xlabel("Communication round", fontsize=12)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(5, 90)

    # --- Right: macro F1 ---
    # macro_f1 is logged only every K rounds (NaN elsewhere) — dropna to get valid points
    for r, label, color in [
        (runs_alpha01[0], r"$\alpha=0.1$ (seed 42)", "C3"),
        (run_alpha1, r"$\alpha=1.0$", "C1"),
        (run_alpha100, r"$\alpha=100$", "C2"),
    ]:
        m = r["metrics"][["round", "macro_f1"]].dropna()
        axes[1].plot(m["round"], m["macro_f1"] * 100, label=label,
                     color=color, linewidth=2, marker="o", markersize=4)

    axes[1].set_xlabel("Communication round", fontsize=12)
    axes[1].set_ylabel("Macro F1 (%)", fontsize=12)
    axes[1].set_title("Macro F1", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=10, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(5, 90)

    fig.suptitle("FedAvg convergence on CIFAR-10: client drift widens the acc–F1 gap",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_convergence_by_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 2 — Best acc + macro F1 by alpha (bar chart)
# ============================================================

def plot_best_acc_vs_alpha(runs_alpha01, run_alpha1, run_alpha100):
    """Side-by-side bars: best accuracy AND macro F1 per alpha."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    alpha_labels = [r"$\alpha=0.1$" + "\n(non-IID)",
                    r"$\alpha=1.0$" + "\n(intermediate)",
                    r"$\alpha=100$" + "\n(quasi-IID)"]
    xs = np.arange(3)
    width = 0.38

    # Accuracy values (mean ± std for alpha=0.1)
    accs_01 = [r["summary"]["best_test_acc"] * 100 for r in runs_alpha01]
    mean_acc_01, std_acc_01 = np.mean(accs_01), np.std(accs_01)
    acc_means = [mean_acc_01,
                 run_alpha1["summary"]["best_test_acc"] * 100,
                 run_alpha100["summary"]["best_test_acc"] * 100]
    acc_stds = [std_acc_01, 0.0, 0.0]

    # Macro F1 values
    f1_01 = [r["summary"]["final_macro_f1"] * 100 for r in runs_alpha01]
    mean_f1_01, std_f1_01 = np.mean(f1_01), np.std(f1_01)
    f1_means = [mean_f1_01,
                run_alpha1["summary"]["final_macro_f1"] * 100,
                run_alpha100["summary"]["final_macro_f1"] * 100]
    f1_stds = [std_f1_01, 0.0, 0.0]

    bars_acc = ax.bar(xs - width/2, acc_means, width, yerr=acc_stds,
                      label="Best accuracy", color="steelblue", alpha=0.85,
                      edgecolor="black", linewidth=1.2, capsize=6)
    bars_f1 = ax.bar(xs + width/2, f1_means, width, yerr=f1_stds,
                     label="Macro F1", color="coral", alpha=0.85,
                     edgecolor="black", linewidth=1.2, capsize=6)

    # Annotate values
    for x, m, s in zip(xs - width/2, acc_means, acc_stds):
        label = f"{m:.1f}±{s:.1f}" if s > 0 else f"{m:.1f}"
        ax.text(x, m + 1.5, label, ha="center", fontsize=9, fontweight="bold")
    for x, m, s in zip(xs + width/2, f1_means, f1_stds):
        label = f"{m:.1f}±{s:.1f}" if s > 0 else f"{m:.1f}"
        ax.text(x, m + 1.5, label, ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("FedAvg final performance: accuracy vs macro F1 across non-IID levels",
                 fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_best_acc_vs_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 3 — Per-class accuracy across three alpha values
# ============================================================

def plot_per_class_comparison(run_alpha01, run_alpha1, run_alpha100):
    """Per-class accuracy: alpha=0.1 vs 1.0 vs 100."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.27

    accs_01 = [run_alpha01["summary"][f"final_acc_class_{c}"] * 100 for c in range(n_classes)]
    accs_1 = [run_alpha1["summary"][f"final_acc_class_{c}"] * 100 for c in range(n_classes)]
    accs_100 = [run_alpha100["summary"][f"final_acc_class_{c}"] * 100 for c in range(n_classes)]

    ax.bar(x - width, accs_01, width, label=r"$\alpha=0.1$ (non-IID)",
           color="C3", alpha=0.85, edgecolor="black", linewidth=1)
    ax.bar(x, accs_1, width, label=r"$\alpha=1.0$ (intermediate)",
           color="C1", alpha=0.85, edgecolor="black", linewidth=1)
    ax.bar(x + width, accs_100, width, label=r"$\alpha=100$ (quasi-IID)",
           color="C2", alpha=0.85, edgecolor="black", linewidth=1)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("FedAvg per-class accuracy: client drift is unequal across classes",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_per_class_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 4 — Fairness metrics by alpha (std + gap, bar charts)
# ============================================================

def plot_fairness_metrics(runs_alpha01, run_alpha1, run_alpha100):
    """Two-panel bar chart: std of per-class acc, and fairness gap, vs alpha."""
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


# ============================================================
# PLOT 5 — Convergence by E (local epochs), alpha=0.1 fixed
# ============================================================

def plot_convergence_by_E(run_E1, run_E2, run_E5):
    """Test accuracy AND macro F1 vs round, varying E. alpha=0.1 fixed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    runs = [(run_E1, "E=1", "C0"),
            (run_E2, "E=2", "C4"),
            (run_E5, "E=5", "C6")]

    # --- Left: accuracy ---
    for r, label, color in runs:
        axes[0].plot(r["metrics"]["round"], r["metrics"]["test_acc"] * 100,
                     label=label, color=color, linewidth=2)
    axes[0].set_xlabel("Communication round", fontsize=12)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=11, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(5, 85)

    # --- Right: macro F1 ---
    for r, label, color in runs:
        m = r["metrics"][["round", "macro_f1"]].dropna()
        axes[1].plot(m["round"], m["macro_f1"] * 100, label=label,
                     color=color, linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Communication round", fontsize=12)
    axes[1].set_ylabel("Macro F1 (%)", fontsize=12)
    axes[1].set_title("Macro F1", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=11, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(5, 85)

    fig.suptitle(r"FedAvg convergence at $\alpha=0.1$, varying local epochs $E$",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_convergence_by_E.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 6 — Trade-off E ↔ accuracy / fairness
# ============================================================

def plot_tradeoff_E(run_E1, run_E2, run_E5):
    """Two-panel bar chart: best_acc and fairness_gap by E. alpha=0.1 fixed."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    E_labels = ["E=1", "E=2", "E=5"]
    xs = np.arange(3)
    colors = ["C0", "C4", "C6"]

    accs = [run_E1["summary"]["best_test_acc"] * 100,
            run_E2["summary"]["best_test_acc"] * 100,
            run_E5["summary"]["best_test_acc"] * 100]
    f1s = [run_E1["summary"]["final_macro_f1"] * 100,
           run_E2["summary"]["final_macro_f1"] * 100,
           run_E5["summary"]["final_macro_f1"] * 100]
    gaps = [run_E1["summary"]["final_fairness_gap"] * 100,
            run_E2["summary"]["final_fairness_gap"] * 100,
            run_E5["summary"]["final_fairness_gap"] * 100]

    # --- Left: accuracy + F1 ---
    width = 0.38
    axes[0].bar(xs - width/2, accs, width, label="Best accuracy",
                color="steelblue", alpha=0.85, edgecolor="black", linewidth=1.2)
    axes[0].bar(xs + width/2, f1s, width, label="Macro F1",
                color="coral", alpha=0.85, edgecolor="black", linewidth=1.2)
    for x, m in zip(xs - width/2, accs):
        axes[0].text(x, m + 1.5, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
    for x, m in zip(xs + width/2, f1s):
        axes[0].text(x, m + 1.5, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(E_labels)
    axes[0].set_ylabel("Performance (%)", fontsize=12)
    axes[0].set_title(r"Higher $E$ helps in our setting" + "\n(more total local training)",
                      fontsize=12)
    axes[0].legend(fontsize=11, loc="lower right")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, 95)

    # --- Right: fairness gap ---
    axes[1].bar(xs, gaps, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.2)
    for x, m in zip(xs, gaps):
        axes[1].text(x, m + 1.5, f"{m:.1f} pp", ha="center",
                     fontsize=10, fontweight="bold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(E_labels)
    axes[1].set_ylabel("Fairness gap (pp)", fontsize=12)
    axes[1].set_title(r"And reduces class-level disparity" + "\n(lower = more fair)",
                      fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 100)

    fig.suptitle(r"Effect of local epochs $E$ at $\alpha=0.1$ (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_tradeoff_E.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 7 — Fairness gap evolution in time by alpha
# ============================================================

def plot_fairness_gap_vs_round(runs_alpha01, run_alpha1, run_alpha100):
    """fairness_gap vs round, by alpha. Uses logged values (every K rounds)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # alpha=0.1 — mean across seeds
    seed_data = []
    for r in runs_alpha01:
        df = r["metrics"][["round", "fairness_gap"]].dropna()
        seed_data.append(df.set_index("round")["fairness_gap"])
    # Align on common rounds
    common = seed_data[0].index
    for s in seed_data[1:]:
        common = common.intersection(s.index)
    arr = np.stack([s.loc[common].values for s in seed_data]) * 100
    mean_gap = arr.mean(axis=0)
    std_gap = arr.std(axis=0)

    ax.plot(common, mean_gap, label=r"$\alpha=0.1$ (3 seeds)",
            color="C3", linewidth=2, marker="o", markersize=5)
    ax.fill_between(common, mean_gap - std_gap, mean_gap + std_gap,
                    color="C3", alpha=0.2)

    for r, label, color in [(run_alpha1, r"$\alpha=1.0$", "C1"),
                            (run_alpha100, r"$\alpha=100$", "C2")]:
        df = r["metrics"][["round", "fairness_gap"]].dropna()
        ax.plot(df["round"], df["fairness_gap"] * 100, label=label,
                color=color, linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    ax.set_title("Fairness gap evolution: non-IID training does NOT close the gap",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_fairness_gap_vs_round.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 8 — (Accuracy - Macro F1) gap evolution in time
# ============================================================

def plot_acc_f1_gap_vs_round(runs_alpha01, run_alpha1, run_alpha100):
    """
    (test_acc - macro_f1) vs round. This 'metric gap' opens precisely when
    the model is sacrificing some classes to boost others — a diagnostic
    signature of client drift that complements the fairness_gap view.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    def get_gap_series(run):
        """Build (round, acc - f1) from rows where macro_f1 is logged."""
        df = run["metrics"][["round", "test_acc", "macro_f1"]].dropna()
        return df["round"].values, (df["test_acc"] - df["macro_f1"]).values * 100

    # alpha=0.1 — mean across seeds
    all_x = None
    seed_gaps = []
    for r in runs_alpha01:
        x, g = get_gap_series(r)
        seed_gaps.append((x, g))
    common_x = seed_gaps[0][0]
    for x, _ in seed_gaps[1:]:
        common_x = np.intersect1d(common_x, x)
    arr = np.stack([
        np.array([g[np.where(x == cx)[0][0]] for cx in common_x])
        for x, g in seed_gaps
    ])
    mean_g = arr.mean(axis=0)
    std_g = arr.std(axis=0)

    ax.plot(common_x, mean_g, label=r"$\alpha=0.1$ (3 seeds)",
            color="C3", linewidth=2, marker="o", markersize=5)
    ax.fill_between(common_x, mean_g - std_g, mean_g + std_g,
                    color="C3", alpha=0.2)

    for r, label, color in [(run_alpha1, r"$\alpha=1.0$", "C1"),
                            (run_alpha100, r"$\alpha=100$", "C2")]:
        x, g = get_gap_series(r)
        ax.plot(x, g, label=label, color=color,
                linewidth=2, marker="o", markersize=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Accuracy − Macro F1 (pp)", fontsize=12)
    ax.set_title("Diagnostic: accuracy hides class-level neglect under client drift",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = PLOTS_DIR / "fedavg_acc_f1_gap_vs_round.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Generating FedAvg baseline plots")
    print("=" * 60)

    # --- Load alpha-ablation runs ---
    runs_alpha01 = [
        load_run("fedavg_alpha0.1_E2_seed42"),
        load_run("fedavg_alpha0.1_E2_seed43"),
        load_run("fedavg_alpha0.1_E2_seed44"),
    ]
    run_alpha1 = load_run("fedavg_alpha1.0_E2_seed42")
    run_alpha100 = load_run("fedavg_alpha100.0_E2_seed42")

    # --- Load E-ablation runs (alpha=0.1, seed=42 fixed) ---
    # E=2 is reused from the alpha ablation
    run_E1 = load_run("fedavg_alpha0.1_E1_seed42")
    run_E2 = runs_alpha01[0]
    run_E5 = load_run("fedavg_alpha0.1_E5_seed42")

    # Retroactive fairness fill-in (no-op for new runs)
    runs_alpha01 = [add_fairness_to_summary(r) for r in runs_alpha01]
    run_alpha1 = add_fairness_to_summary(run_alpha1)
    run_alpha100 = add_fairness_to_summary(run_alpha100)
    run_E1 = add_fairness_to_summary(run_E1)
    run_E5 = add_fairness_to_summary(run_E5)

    print("\nLoaded runs:")
    print(f"  Alpha sweep:  alpha=0.1 (3 seeds), alpha=1.0, alpha=100")
    print(f"  E sweep:      E=1, E=2, E=5  (all at alpha=0.1, seed=42)")

    # ---- Summary numbers ----
    bests_01 = [r["summary"]["best_test_acc"] * 100 for r in runs_alpha01]
    f1s_01 = [r["summary"]["final_macro_f1"] * 100 for r in runs_alpha01]
    gaps_01 = [r["summary"]["final_fairness_gap"] * 100 for r in runs_alpha01]

    print(f"\n--- Alpha ablation (E=2) ---")
    print(f"  alpha=0.1:  acc={np.mean(bests_01):.2f}±{np.std(bests_01):.2f}%  "
          f"F1={np.mean(f1s_01):.2f}±{np.std(f1s_01):.2f}%  "
          f"gap={np.mean(gaps_01):.2f}±{np.std(gaps_01):.2f}pp")
    print(f"  alpha=1.0:  acc={run_alpha1['summary']['best_test_acc']*100:.2f}%  "
          f"F1={run_alpha1['summary']['final_macro_f1']*100:.2f}%  "
          f"gap={run_alpha1['summary']['final_fairness_gap']*100:.2f}pp")
    print(f"  alpha=100:  acc={run_alpha100['summary']['best_test_acc']*100:.2f}%  "
          f"F1={run_alpha100['summary']['final_macro_f1']*100:.2f}%  "
          f"gap={run_alpha100['summary']['final_fairness_gap']*100:.2f}pp")

    print(f"\n--- E ablation (alpha=0.1, seed=42) ---")
    for label, r in [("E=1", run_E1), ("E=2", run_E2), ("E=5", run_E5)]:
        print(f"  {label}:  acc={r['summary']['best_test_acc']*100:.2f}%  "
              f"F1={r['summary']['final_macro_f1']*100:.2f}%  "
              f"gap={r['summary']['final_fairness_gap']*100:.2f}pp")

    # ---- Generate plots ----
    print("\nGenerating plots...")
    plot_convergence_by_alpha(runs_alpha01, run_alpha1, run_alpha100)
    plot_best_acc_vs_alpha(runs_alpha01, run_alpha1, run_alpha100)
    plot_per_class_comparison(runs_alpha01[0], run_alpha1, run_alpha100)
    plot_fairness_metrics(runs_alpha01, run_alpha1, run_alpha100)
    plot_convergence_by_E(run_E1, run_E2, run_E5)
    plot_tradeoff_E(run_E1, run_E2, run_E5)
    plot_fairness_gap_vs_round(runs_alpha01, run_alpha1, run_alpha100)
    plot_acc_f1_gap_vs_round(runs_alpha01, run_alpha1, run_alpha100)

    print("\nDone. Plots are in plots/")


if __name__ == "__main__":
    main()