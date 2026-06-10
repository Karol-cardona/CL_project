"""
Generate FedAvg baseline plots from completed experiments.

Reads metrics.csv and summary.json from experiments/results/ and produces
the plots used in the FedAvg part of the report.

Changes vs. the first version:
- 3 seeds are now aggregated for EVERY alpha (0.1, 1.0, 100), not just 0.1.
  This gives real mean +/- std error bars across the whole alpha sweep.
- New compute-matched E plot: accuracy and fairness gap vs *cumulative local
  epochs* (round * E), using the constant-LR runs E1/R200, E2/R100, E5/R40.
  This is the honest counterpart to the round-fixed E ablation.
- New worst-class (min per-class accuracy) vs round plot: the most direct
  "forgetting" signal to pair with the per-class bar chart.
- Titles corrected: the acc - macro F1 gap is small on a balanced test set,
  so it is demoted from "main diagnostic" to a complementary view; class
  disparity is carried by the per-class / std / fairness-gap plots.

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

# Consistent colors
C_A01, C_A1, C_A100 = "C3", "C1", "C2"     # alpha = 0.1, 1.0, 100
C_E1, C_E2, C_E5 = "C0", "C4", "C6"        # E = 1, 2, 5
SEEDS = [42, 43, 44]


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


def load_seeds(alpha_str: str, E: int, seeds=SEEDS, suffix: str = None) -> list:
    """Load the seed-runs for one (alpha, E) configuration."""
    runs = []
    for s in seeds:
        name = f"fedavg_alpha{alpha_str}_E{E}_seed{s}"
        if suffix:
            name += f"_{suffix}"
        runs.append(add_fairness_to_summary(load_run(name)))
    return runs


# ---- aggregation helpers ----

def agg(runs: list, key: str, scale: float = 100.0):
    """Mean and std of a summary scalar across seed-runs."""
    vals = np.array([r["summary"][key] for r in runs], dtype=float) * scale
    return float(vals.mean()), float(vals.std())


def dense_mean(runs: list, col: str, scale: float = 100.0):
    """Mean/std of a per-round column logged every round (aligned rounds)."""
    rounds = runs[0]["metrics"]["round"].values
    arr = np.stack([r["metrics"][col].values for r in runs]) * scale
    return rounds, arr.mean(axis=0), arr.std(axis=0)


def sparse_mean(runs: list, col: str, scale: float = 100.0):
    """
    Mean/std of a per-round column logged only every K rounds (NaN elsewhere).
    Aligns seeds on their common set of logged rounds.
    """
    series = [r["metrics"][["round", col]].dropna().set_index("round")[col]
              for r in runs]
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    arr = np.stack([s.loc[common].values for s in series]) * scale
    return common.values, arr.mean(axis=0), arr.std(axis=0)


def mean_per_class(runs: list, num_classes: int = 10, scale: float = 100.0):
    """Mean per-class accuracy across seed-runs."""
    arr = np.array([[r["summary"][f"final_acc_class_{c}"] for c in range(num_classes)]
                    for r in runs], dtype=float) * scale
    return arr.mean(axis=0)


# ============================================================
# PLOT 1 -- Convergence by alpha (accuracy + macro F1), 3 seeds each
# ============================================================

def plot_convergence_by_alpha(runs_a01, runs_a1, runs_a100):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    groups = [(runs_a01, r"$\alpha=0.1$ (non-IID)", C_A01),
              (runs_a1, r"$\alpha=1.0$ (intermediate)", C_A1),
              (runs_a100, r"$\alpha=100$ (quasi-IID)", C_A100)]

    # --- Left: accuracy (mean +/- std band, 3 seeds) ---
    for runs, label, color in groups:
        rounds, mean_acc, std_acc = dense_mean(runs, "test_acc")
        axes[0].plot(rounds, mean_acc, label=label, color=color, linewidth=2)
        axes[0].fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                             color=color, alpha=0.18)
    axes[0].set_xlabel("Communication round", fontsize=12)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy (mean +/- std, 3 seeds)", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(5, 90)

    # --- Right: macro F1 (mean +/- std band, 3 seeds) ---
    for runs, label, color in groups:
        x, mean_f1, std_f1 = sparse_mean(runs, "macro_f1")
        axes[1].plot(x, mean_f1, label=label, color=color,
                     linewidth=2, marker="o", markersize=4)
        axes[1].fill_between(x, mean_f1 - std_f1, mean_f1 + std_f1,
                             color=color, alpha=0.18)
    axes[1].set_xlabel("Communication round", fontsize=12)
    axes[1].set_ylabel("Macro F1 (%)", fontsize=12)
    axes[1].set_title("Macro F1 (mean +/- std, 3 seeds)", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=10, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(5, 90)

    fig.suptitle("FedAvg convergence on CIFAR-10: accuracy and macro F1 across non-IID levels",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_convergence_by_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 2 -- Best acc + macro F1 by alpha (bars), 3 seeds each
# ============================================================

def plot_best_acc_vs_alpha(runs_a01, runs_a1, runs_a100):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    alpha_labels = [r"$\alpha=0.1$" + "\n(non-IID)",
                    r"$\alpha=1.0$" + "\n(intermediate)",
                    r"$\alpha=100$" + "\n(quasi-IID)"]
    xs = np.arange(3)
    width = 0.38

    acc_means, acc_stds, f1_means, f1_stds = [], [], [], []
    for runs in (runs_a01, runs_a1, runs_a100):
        m, s = agg(runs, "best_test_acc"); acc_means.append(m); acc_stds.append(s)
        m, s = agg(runs, "final_macro_f1"); f1_means.append(m); f1_stds.append(s)

    ax.bar(xs - width/2, acc_means, width, yerr=acc_stds,
           label="Best accuracy", color="steelblue", alpha=0.85,
           edgecolor="black", linewidth=1.2, capsize=6)
    ax.bar(xs + width/2, f1_means, width, yerr=f1_stds,
           label="Macro F1", color="coral", alpha=0.85,
           edgecolor="black", linewidth=1.2, capsize=6)

    for x, m, s in zip(xs - width/2, acc_means, acc_stds):
        ax.text(x, m + s + 1.5, f"{m:.1f}+/-{s:.1f}", ha="center",
                fontsize=9, fontweight="bold")
    for x, m, s in zip(xs + width/2, f1_means, f1_stds):
        ax.text(x, m + s + 1.5, f"{m:.1f}+/-{s:.1f}", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("FedAvg final performance: accuracy vs macro F1 across non-IID levels (3 seeds)",
                 fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_best_acc_vs_alpha.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 3 -- Per-class accuracy across three alpha values (3-seed mean)
# ============================================================

def plot_per_class_comparison(runs_a01, runs_a1, runs_a100):
    fig, ax = plt.subplots(figsize=(13, 5.5))

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.27

    accs_01 = mean_per_class(runs_a01, n_classes)
    accs_1 = mean_per_class(runs_a1, n_classes)
    accs_100 = mean_per_class(runs_a100, n_classes)

    ax.bar(x - width, accs_01, width, label=r"$\alpha=0.1$ (non-IID)",
           color=C_A01, alpha=0.85, edgecolor="black", linewidth=1)
    ax.bar(x, accs_1, width, label=r"$\alpha=1.0$ (intermediate)",
           color=C_A1, alpha=0.85, edgecolor="black", linewidth=1)
    ax.bar(x + width, accs_100, width, label=r"$\alpha=100$ (quasi-IID)",
           color=C_A100, alpha=0.85, edgecolor="black", linewidth=1)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("FedAvg per-class accuracy (3-seed mean): client drift is unequal across classes",
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
# PLOT 4 -- Fairness metrics by alpha (std primary, gap complementary)
# ============================================================

def plot_fairness_metrics(runs_a01, runs_a1, runs_a100):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    alpha_labels = [r"$\alpha=0.1$" + "\n(non-IID)",
                    r"$\alpha=1.0$",
                    r"$\alpha=100$" + "\n(quasi-IID)"]
    xs = np.arange(3)
    colors = [C_A01, C_A1, C_A100]
    groups = (runs_a01, runs_a1, runs_a100)

    # --- Left: std of per-class accuracy (primary, monotone) ---
    means, errs = zip(*[agg(g, "final_std_acc") for g in groups])
    axes[0].bar(xs, means, yerr=errs, color=colors,
                alpha=0.85, edgecolor="black", linewidth=1.2, capsize=6)
    for x, m, s in zip(xs, means, errs):
        axes[0].text(x, m + s + 0.4, f"{m:.1f}+/-{s:.1f}%", ha="center",
                     fontsize=10, fontweight="bold")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(alpha_labels)
    axes[0].set_ylabel("Std of per-class accuracy (%)", fontsize=12)
    axes[0].set_title("Class-level heterogeneity -- primary metric\n(lower = more uniform)",
                      fontsize=12)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Right: fairness gap (complementary, sensitive to a single class) ---
    means, errs = zip(*[agg(g, "final_fairness_gap") for g in groups])
    axes[1].bar(xs, means, yerr=errs, color=colors,
                alpha=0.85, edgecolor="black", linewidth=1.2, capsize=6)
    for x, m, s in zip(xs, means, errs):
        axes[1].text(x, m + s + 1, f"{m:.1f}+/-{s:.1f} pp", ha="center",
                     fontsize=10, fontweight="bold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(alpha_labels)
    axes[1].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    axes[1].set_title("Class-level disparity -- complementary\n(max-min, sensitive to one class)",
                      fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("FedAvg: client drift produces class-level inequality (3 seeds)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_fairness_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 5 -- Convergence by E (local epochs), alpha=0.1, ROUND-FIXED
# ============================================================

def plot_convergence_by_E(run_E1, run_E2, run_E5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    runs = [(run_E1, "E=1", C_E1), (run_E2, "E=2", C_E2), (run_E5, "E=5", C_E5)]

    for r, label, color in runs:
        axes[0].plot(r["metrics"]["round"], r["metrics"]["test_acc"] * 100,
                     label=label, color=color, linewidth=2)
    axes[0].set_xlabel("Communication round", fontsize=12)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=11, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(5, 85)

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

    fig.suptitle(r"FedAvg convergence at $\alpha=0.1$, varying $E$ (round-fixed: more E = more total compute)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_convergence_by_E.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 6 -- Trade-off E (ROUND-FIXED, NOT compute-matched)
# ============================================================

def plot_tradeoff_E(run_E1, run_E2, run_E5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    E_labels = ["E=1", "E=2", "E=5"]
    xs = np.arange(3)
    colors = [C_E1, C_E2, C_E5]

    accs = [run_E1["summary"]["best_test_acc"] * 100,
            run_E2["summary"]["best_test_acc"] * 100,
            run_E5["summary"]["best_test_acc"] * 100]
    f1s = [run_E1["summary"]["final_macro_f1"] * 100,
           run_E2["summary"]["final_macro_f1"] * 100,
           run_E5["summary"]["final_macro_f1"] * 100]
    gaps = [run_E1["summary"]["final_fairness_gap"] * 100,
            run_E2["summary"]["final_fairness_gap"] * 100,
            run_E5["summary"]["final_fairness_gap"] * 100]

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
    axes[0].set_title("Higher E helps at fixed rounds\n(because total local training grows with E)",
                      fontsize=12)
    axes[0].legend(fontsize=11, loc="lower right")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, 95)

    axes[1].bar(xs, gaps, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.2)
    for x, m in zip(xs, gaps):
        axes[1].text(x, m + 1.5, f"{m:.1f} pp", ha="center",
                     fontsize=10, fontweight="bold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(E_labels)
    axes[1].set_ylabel("Fairness gap (pp)", fontsize=12)
    axes[1].set_title("Gap shrinks here, but only because of extra compute",
                      fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 100)

    fig.suptitle(r"Effect of $E$ at $\alpha=0.1$ -- ROUND-FIXED (NOT compute-matched; see compute-matched plot)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_tradeoff_E.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 7 -- Fairness gap evolution in time by alpha (3 seeds each)
# ============================================================

def plot_fairness_gap_vs_round(runs_a01, runs_a1, runs_a100):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    groups = [(runs_a01, r"$\alpha=0.1$", C_A01),
              (runs_a1, r"$\alpha=1.0$", C_A1),
              (runs_a100, r"$\alpha=100$", C_A100)]
    for runs, label, color in groups:
        x, mean_gap, std_gap = sparse_mean(runs, "fairness_gap")
        ax.plot(x, mean_gap, label=label + " (3 seeds)", color=color,
                linewidth=2, marker="o", markersize=5)
        ax.fill_between(x, mean_gap - std_gap, mean_gap + std_gap,
                        color=color, alpha=0.18)

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
# PLOT 8 -- (Accuracy - Macro F1) gap in time (complementary, small on balanced test set)
# ============================================================

def plot_acc_f1_gap_vs_round(runs_a01, runs_a1, runs_a100):
    """
    On a class-balanced test set, accuracy ~= macro F1 at convergence, so this
    gap is small. It is kept only as a complementary diagnostic: a positive gap
    in early training signals the model favouring some classes over others.
    Class disparity is carried mainly by the per-class / std / fairness-gap plots.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    def gap_series(runs):
        # acc - f1 on rows where macro_f1 is logged; mean across seeds on common rounds
        per = [r["metrics"][["round", "test_acc", "macro_f1"]].dropna() for r in runs]
        s = [p.assign(g=(p["test_acc"] - p["macro_f1"]) * 100).set_index("round")["g"]
             for p in per]
        common = s[0].index
        for si in s[1:]:
            common = common.intersection(si.index)
        arr = np.stack([si.loc[common].values for si in s])
        return common.values, arr.mean(axis=0), arr.std(axis=0)

    groups = [(runs_a01, r"$\alpha=0.1$", C_A01),
              (runs_a1, r"$\alpha=1.0$", C_A1),
              (runs_a100, r"$\alpha=100$", C_A100)]
    for runs, label, color in groups:
        x, mean_g, std_g = gap_series(runs)
        ax.plot(x, mean_g, label=label + " (3 seeds)", color=color,
                linewidth=2, marker="o", markersize=5)
        ax.fill_between(x, mean_g - std_g, mean_g + std_g, color=color, alpha=0.18)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Accuracy - Macro F1 (pp)", fontsize=12)
    ax.set_title("Accuracy - Macro F1 gap (modest on a balanced test set; complementary view)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_acc_f1_gap_vs_round.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 9 (NEW) -- Compute-matched E: metrics vs CUMULATIVE local epochs
# ============================================================

def plot_compute_matched_E(cm_E1, cm_E2, cm_E5):
    """
    The honest counterpart to the round-fixed E ablation.
    All three runs perform ~the same total local work (E1 x R200, E2 x R100,
    E5 x R40 = ~200 local epochs), constant LR. The x-axis is cumulative local
    epochs (round * E), so the three are directly comparable at equal compute.
    Expectation from drift theory: larger E => worse fairness gap at equal compute.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    runs = [(cm_E1, 1, "E=1 (R=200)", C_E1),
            (cm_E2, 2, "E=2 (R=100)", C_E2),
            (cm_E5, 5, "E=5 (R=40)", C_E5)]

    # --- Left: accuracy vs cumulative local epochs ---
    for r, E, label, color in runs:
        m = r["metrics"]
        x = m["round"].values * E
        axes[0].plot(x, m["test_acc"].values * 100, label=label,
                     color=color, linewidth=2)
    axes[0].set_xlabel("Cumulative local epochs (round x E)", fontsize=12)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy at equal compute", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=11, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(5, 85)

    # --- Right: fairness gap vs cumulative local epochs ---
    for r, E, label, color in runs:
        m = r["metrics"][["round", "fairness_gap"]].dropna()
        x = m["round"].values * E
        axes[1].plot(x, m["fairness_gap"].values * 100, label=label,
                     color=color, linewidth=2, marker="o", markersize=5)
    axes[1].set_xlabel("Cumulative local epochs (round x E)", fontsize=12)
    axes[1].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    axes[1].set_title("Fairness gap at equal compute\n(larger E = more drift = wider gap)",
                      fontsize=12)
    axes[1].legend(loc="upper right", fontsize=11, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    fig.suptitle(r"COMPUTE-MATCHED $E$ at $\alpha=0.1$ (constant LR, ~200 total local epochs)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_compute_matched_E.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 10 (NEW) -- Worst-class accuracy (min per-class) vs round
# ============================================================

def plot_min_acc_vs_round(runs_a01, runs_a1, runs_a100):
    """
    The most direct 'forgetting' signal: the accuracy of the worst class over
    time. Under heavy non-IID (alpha=0.1) the global model can keep a whole
    class stuck near zero even as overall accuracy climbs.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    groups = [(runs_a01, r"$\alpha=0.1$", C_A01),
              (runs_a1, r"$\alpha=1.0$", C_A1),
              (runs_a100, r"$\alpha=100$", C_A100)]
    for runs, label, color in groups:
        x, mean_min, std_min = sparse_mean(runs, "min_acc")
        ax.plot(x, mean_min, label=label + " (3 seeds)", color=color,
                linewidth=2, marker="o", markersize=5)
        ax.fill_between(x, mean_min - std_min, mean_min + std_min,
                        color=color, alpha=0.18)

    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Worst-class accuracy (min per-class, %)", fontsize=12)
    ax.set_title("Worst-class accuracy over time: the direct signature of forgetting",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_min_acc_vs_round.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# PLOT 11 (NEW) -- Fairness gap: round-fixed vs compute-matched (thesis figure)
# ============================================================

def plot_gap_roundfixed_vs_compute(run_E1, run_E2, run_E5, cm_E1, cm_E2, cm_E5):
    """
    The thesis figure: the SAME E sweep under two budgeting regimes.
    Left  (round-fixed, 100 rounds): more E = more total compute  -> gap DOWN.
    Right (compute-matched, ~200 local epochs): equal compute      -> gap UP with E.
    The fairness effect of E flips sign once compute is held fixed: that flip is
    the whole point. Shared y-axis so the bar heights are directly comparable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    E_labels = ["E=1", "E=2", "E=5"]
    xs = np.arange(3)
    colors = [C_E1, C_E2, C_E5]

    rf_gaps = [run_E1["summary"]["final_fairness_gap"] * 100,
               run_E2["summary"]["final_fairness_gap"] * 100,
               run_E5["summary"]["final_fairness_gap"] * 100]
    cm_gaps = [cm_E1["summary"]["final_fairness_gap"] * 100,
               cm_E2["summary"]["final_fairness_gap"] * 100,
               cm_E5["summary"]["final_fairness_gap"] * 100]

    panels = [
        (axes[0], rf_gaps, "Round-fixed (100 rounds)",
         "more E = more total compute -> gap DOWN"),
        (axes[1], cm_gaps, "Compute-matched (~200 local epochs)",
         "equal compute -> gap UP with E"),
    ]
    for ax, gaps, title, sub in panels:
        ax.bar(xs, gaps, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
        for x, m in zip(xs, gaps):
            ax.text(x, m + 1.5, f"{m:.1f} pp", ha="center",
                    fontsize=11, fontweight="bold")
        # trend arrow across the three bars (visual cue for the direction)
        ax.annotate("", xy=(2, gaps[2]), xytext=(0, gaps[0]),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.6, alpha=0.6))
        ax.set_xticks(xs)
        ax.set_xticklabels(E_labels)
        ax.set_title(f"{title}\n({sub})", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)

    fig.suptitle(r"Local epochs $E$ at $\alpha=0.1$: the fairness effect FLIPS once compute is matched",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = PLOTS_DIR / "fedavg_gap_roundfixed_vs_compute.png"
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

    # --- Alpha sweep: 3 seeds for ALL alphas (E=2, cosine LR) ---
    runs_a01 = load_seeds("0.1", 2)
    runs_a1 = load_seeds("1.0", 2)
    runs_a100 = load_seeds("100.0", 2)

    # --- E ablation, round-fixed (alpha=0.1, seed 42, cosine LR) ---
    run_E1 = add_fairness_to_summary(load_run("fedavg_alpha0.1_E1_seed42"))
    run_E2 = runs_a01[0]  # reuse seed-42 E=2 from the alpha sweep
    run_E5 = add_fairness_to_summary(load_run("fedavg_alpha0.1_E5_seed42"))

    # --- E ablation, compute-matched (alpha=0.1, seed 42, constant LR) ---
    cm = "compute_matched_constLR"
    cm_E1 = add_fairness_to_summary(load_run(f"fedavg_alpha0.1_E1_seed42_{cm}"))
    cm_E2 = add_fairness_to_summary(load_run(f"fedavg_alpha0.1_E2_seed42_{cm}"))
    cm_E5 = add_fairness_to_summary(load_run(f"fedavg_alpha0.1_E5_seed42_{cm}"))

    print("\nLoaded runs:")
    print("  Alpha sweep:        alpha=0.1, 1.0, 100  (3 seeds each, E=2)")
    print("  E ablation (rounds): E=1, 2, 5  (alpha=0.1, seed 42)")
    print("  E ablation (compute): E=1/R200, E=2/R100, E=5/R40  (const LR)")

    # ---- Summary numbers (3 seeds for every alpha now) ----
    print("\n--- Alpha ablation (E=2, 3 seeds) ---")
    for name, runs in [("alpha=0.1", runs_a01), ("alpha=1.0", runs_a1),
                       ("alpha=100", runs_a100)]:
        am, asd = agg(runs, "best_test_acc")
        fm, fsd = agg(runs, "final_macro_f1")
        gm, gsd = agg(runs, "final_fairness_gap")
        sm, ssd = agg(runs, "final_std_acc")
        print(f"  {name:<10s} acc={am:.2f}+/-{asd:.2f}%  F1={fm:.2f}+/-{fsd:.2f}%  "
              f"gap={gm:.2f}+/-{gsd:.2f}pp  std={sm:.2f}+/-{ssd:.2f}%")

    print("\n--- E ablation: ROUND-FIXED vs COMPUTE-MATCHED (alpha=0.1, seed 42) ---")
    print("  round-fixed (100 rounds, more E = more compute):")
    for label, r in [("E=1", run_E1), ("E=2", run_E2), ("E=5", run_E5)]:
        print(f"    {label}: acc={r['summary']['best_test_acc']*100:.2f}%  "
              f"gap={r['summary']['final_fairness_gap']*100:.2f}pp")
    print("  compute-matched (~200 local epochs, const LR):")
    for label, r in [("E=1", cm_E1), ("E=2", cm_E2), ("E=5", cm_E5)]:
        print(f"    {label}: acc={r['summary']['best_test_acc']*100:.2f}%  "
              f"gap={r['summary']['final_fairness_gap']*100:.2f}pp")

    # ---- Generate plots ----
    print("\nGenerating plots...")
    plot_convergence_by_alpha(runs_a01, runs_a1, runs_a100)
    plot_best_acc_vs_alpha(runs_a01, runs_a1, runs_a100)
    plot_per_class_comparison(runs_a01, runs_a1, runs_a100)
    plot_fairness_metrics(runs_a01, runs_a1, runs_a100)
    plot_convergence_by_E(run_E1, run_E2, run_E5)
    plot_tradeoff_E(run_E1, run_E2, run_E5)
    plot_fairness_gap_vs_round(runs_a01, runs_a1, runs_a100)
    plot_acc_f1_gap_vs_round(runs_a01, runs_a1, runs_a100)
    plot_compute_matched_E(cm_E1, cm_E2, cm_E5)
    plot_min_acc_vs_round(runs_a01, runs_a1, runs_a100)
    plot_gap_roundfixed_vs_compute(run_E1, run_E2, run_E5, cm_E1, cm_E2, cm_E5)

    print("\nDone. Plots are in plots/")


if __name__ == "__main__":
    main()