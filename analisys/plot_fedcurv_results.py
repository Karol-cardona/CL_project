"""
Generate FedCurv plots for the presentation.

Two figures:
  1) fedcurv_lambda_ablation.png  -- E=2 lambda sweep (seed 42):
       accuracy and fairness gap vs lambda, with the FedAvg (seed 42)
       reference line. Shows: lambda=1 is best but ~FedAvg; large lambda
       over-regularizes and collapses the model.
  2) fedcurv_vs_fedavg_by_E.png   -- FedCurv(lambda=1) vs FedAvg at E=2 and E=5
       (3 seeds, mean +/- std). Shows the E-dependent benefit: no gain at
       E=2, a modest fairness-gap improvement at E=5.

Numbers are taken directly from the run summaries (final round, best acc,
extended metrics). They are hard-coded here so the figure is reproducible
without depending on the exact results-folder names.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data (from run summaries)
# ------------------------------------------------------------------

# FedCurv E=2 lambda ablation, seed 42
LAMBDAS      = [0.1, 1, 10, 100, 1000]
FC_E2_ACC    = [69.34, 69.97, 69.18, 60.22, 41.94]   # best accuracy (%)
FC_E2_GAP    = [75.60, 68.40, 69.80, 71.60, 61.70]   # fairness gap (pp)
FC_E2_WORST  = [16.60, 23.50, 22.60, 14.40,  4.50]   # worst-class acc (%)

# FedAvg E=2 seed-42 reference (same seed as the ablation)
FA_E2_ACC_S42 = 70.37
FA_E2_GAP_S42 = 65.80

# 3-seed aggregates (mean, std) for the E=2 vs E=5 comparison
# accuracy (%)
ACC = {
    ("FedAvg", "E=2"): (68.8, 2.3),
    ("FedCurv", "E=2"): (68.4, 2.7),
    ("FedAvg", "E=5"): (76.6, 2.1),
    ("FedCurv", "E=5"): (77.0, 1.6),
}
# fairness gap (pp)
GAP = {
    ("FedAvg", "E=2"): (59.4, 19.0),
    ("FedCurv", "E=2"): (58.8, 18.5),
    ("FedAvg", "E=5"): (50.3, 16.9),
    ("FedCurv", "E=5"): (47.3, 14.1),
}

C_FA = "steelblue"    # FedAvg
C_FC = "indianred"    # FedCurv


# ------------------------------------------------------------------
# Figure 1 -- lambda ablation at E=2
# ------------------------------------------------------------------

def plot_lambda_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(LAMBDAS))
    xticklabels = [str(l) for l in LAMBDAS]

    # Left: accuracy vs lambda
    axes[0].plot(x, FC_E2_ACC, "o-", color=C_FC, linewidth=2,
                 markersize=8, label="FedCurv (seed 42)")
    axes[0].axhline(FA_E2_ACC_S42, color=C_FA, linestyle="--", linewidth=2,
                    label=f"FedAvg baseline ({FA_E2_ACC_S42:.1f}%)")
    for xi, v in zip(x, FC_E2_ACC):
        axes[0].annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(xticklabels)
    axes[0].set_xlabel(r"Regularization coefficient $\lambda$ (log grid)", fontsize=12)
    axes[0].set_ylabel("Best test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy vs $\\lambda$\n(large $\\lambda$ over-regularizes)", fontsize=12)
    axes[0].legend(fontsize=10, loc="lower left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(35, 80)

    # Right: fairness gap vs lambda
    axes[1].plot(x, FC_E2_GAP, "o-", color=C_FC, linewidth=2,
                 markersize=8, label="FedCurv (seed 42)")
    axes[1].axhline(FA_E2_GAP_S42, color=C_FA, linestyle="--", linewidth=2,
                    label=f"FedAvg baseline ({FA_E2_GAP_S42:.1f} pp)")
    for xi, v in zip(x, FC_E2_GAP):
        axes[1].annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(xticklabels)
    axes[1].set_xlabel(r"Regularization coefficient $\lambda$ (log grid)", fontsize=12)
    axes[1].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    axes[1].set_title("Fairness gap vs $\\lambda$\n(no $\\lambda$ beats FedAvg at E=2)", fontsize=12)
    axes[1].legend(fontsize=10, loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(40, 85)

    fig.suptitle(r"FedCurv $\lambda$ ablation at $\alpha=0.1$, E=2: is the regularization working?",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    p = PLOTS_DIR / "fedcurv_lambda_ablation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ------------------------------------------------------------------
# Figure 2 -- FedCurv vs FedAvg at E=2 and E=5 (3 seeds)
# ------------------------------------------------------------------

def plot_fedcurv_vs_fedavg_by_E():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    Es = ["E=2", "E=5"]
    x = np.arange(len(Es))
    width = 0.36

    # Left: accuracy
    fa = [ACC[("FedAvg", e)] for e in Es]
    fc = [ACC[("FedCurv", e)] for e in Es]
    axes[0].bar(x - width/2, [m for m, _ in fa], width, yerr=[s for _, s in fa],
                capsize=6, color=C_FA, alpha=0.85, edgecolor="black", label="FedAvg")
    axes[0].bar(x + width/2, [m for m, _ in fc], width, yerr=[s for _, s in fc],
                capsize=6, color=C_FC, alpha=0.85, edgecolor="black", label="FedCurv ($\\lambda$=1)")
    for xi, (m, s) in zip(x - width/2, fa):
        axes[0].text(xi, m + s + 0.6, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    for xi, (m, s) in zip(x + width/2, fc):
        axes[0].text(xi, m + s + 0.6, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(Es)
    axes[0].set_ylabel("Best test accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy (3 seeds)\ncomparable at both E", fontsize=12)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, 90)

    # Right: fairness gap
    fa = [GAP[("FedAvg", e)] for e in Es]
    fc = [GAP[("FedCurv", e)] for e in Es]
    axes[1].bar(x - width/2, [m for m, _ in fa], width, yerr=[s for _, s in fa],
                capsize=6, color=C_FA, alpha=0.85, edgecolor="black", label="FedAvg")
    axes[1].bar(x + width/2, [m for m, _ in fc], width, yerr=[s for _, s in fc],
                capsize=6, color=C_FC, alpha=0.85, edgecolor="black", label="FedCurv ($\\lambda$=1)")
    for xi, (m, s) in zip(x - width/2, fa):
        axes[1].text(xi, m + s + 1.2, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    for xi, (m, s) in zip(x + width/2, fc):
        axes[1].text(xi, m + s + 1.2, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(Es)
    axes[1].set_ylabel("Fairness gap: max - min per-class acc (pp)", fontsize=12)
    axes[1].set_title("Fairness gap (3 seeds, lower = better)\nFedCurv helps only at E=5", fontsize=12)
    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 90)

    fig.suptitle(r"FedCurv vs FedAvg at $\alpha=0.1$: the benefit is E-dependent (emerges at larger E)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    p = PLOTS_DIR / "fedcurv_vs_fedavg_by_E.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def main():
    print("Generating FedCurv plots...")
    plot_lambda_ablation()
    plot_fedcurv_vs_fedavg_by_E()
    print("Done. Plots in plots/")


if __name__ == "__main__":
    main()