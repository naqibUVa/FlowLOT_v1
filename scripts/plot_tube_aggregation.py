#!/usr/bin/env python3
"""Plot tube aggregation (multi-tube fusion) results for FlowLOT benchmark."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 300,
})

def main():
    res_dir = Path("data/results/repeated_classification_blast110/aggregate")
    summary_path = res_dir / "summary.csv"
    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    model_colors = {
        "random_forest": "#1f77b4",
        "nsc": "#2ca02c",
        "nsc_energy": "#17becf",
        "logistic": "#9467bd",
        "extra_trees": "#e377c2",
        "linear_svm": "#7f7f7f",
    }
    model_labels = {
        "random_forest": "Random Forest",
        "nsc": "NSC",
        "nsc_energy": "NSC (Energy)",
        "logistic": "Logistic",
        "extra_trees": "Extra Trees",
        "linear_svm": "Linear SVM",
    }

    # Panel 1: Early Mean (Feature Fusion across tubes P1-P4)
    ax1 = axes[0]
    early_df = df[df["aggregation"] == "early_mean"]
    for model, grp in early_df.groupby("model"):
        grp = grp.sort_values("k")
        ax1.plot(
            grp["k"], grp["balanced_accuracy_mean"],
            label=model_labels.get(model, model),
            color=model_colors.get(model, "#333"),
            marker="o",
        )
    ax1.set_title("(a) Early Feature Fusion (early_mean)", fontweight="bold")
    ax1.set_xlabel("Training Patients Per Class (k)")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_xticks([2, 4, 6, 8])
    ax1.set_ylim(0.28, 0.80)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Late Soft (Decision Probability Fusion across tubes P1-P4)
    ax2 = axes[1]
    late_df = df[df["aggregation"] == "late_soft"]
    for model, grp in late_df.groupby("model"):
        grp = grp.sort_values("k")
        ax2.plot(
            grp["k"], grp["balanced_accuracy_mean"],
            label=model_labels.get(model, model),
            color=model_colors.get(model, "#333"),
            marker="s",
        )
    ax2.set_title("(b) Late Decision Fusion (late_soft)", fontweight="bold")
    ax2.set_xlabel("Training Patients Per Class (k)")
    ax2.set_xticks([2, 4, 6, 8])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    # Panel 3: Single Tube vs. Tube Aggregated Comparison (Random Forest & NSC)
    ax3 = axes[2]
    rf_df = df[df["model"] == "random_forest"]
    tube_colors = {
        "single P1": "#aec7e8",
        "single P2": "#3182bd",
        "single P3": "#9ecae1",
        "single P4": "#c6dbef",
        "early_mean ALL": "#d62728",
        "late_soft ALL": "#ff7f0e",
    }
    for (agg, tube), grp in rf_df.groupby(["aggregation", "tube"]):
        label = f"{agg} {tube}"
        grp = grp.sort_values("k")
        is_agg = "ALL" in label
        ax3.plot(
            grp["k"], grp["balanced_accuracy_mean"],
            label=label,
            color=tube_colors.get(label, "#333"),
            marker="^" if is_agg else "o",
            linestyle="-" if is_agg else ":",
            linewidth=2.4 if is_agg else 1.3,
        )
    ax3.set_title("(c) Multi-Tube vs. Individual Tubes (RF)", fontweight="bold")
    ax3.set_xlabel("Training Patients Per Class (k)")
    ax3.set_xticks([2, 4, 6, 8])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", frameon=True, fontsize=8)

    sns.despine(fig)
    plt.tight_layout()

    out_file = res_dir / "tube_aggregation_curves.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(res_dir / "tube_aggregation_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tube aggregation plot to {out_file}")

if __name__ == "__main__":
    main()
