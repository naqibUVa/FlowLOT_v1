#!/usr/bin/env python3
"""Generate publication-ready clean benchmark visualizations for FlowLOT."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure aesthetic style
sns.set_theme(style="ticks", font_scale=1.05)
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi": 300,
})

def main():
    res_dir = Path("data/results/repeated_classification_blast110/aggregate")
    summary_path = res_dir / "summary.csv"
    if not summary_path.exists():
        print(f"Error: {summary_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(summary_path)

    # Classify models into categories
    lot_models = ["logistic", "linear_svm", "random_forest", "extra_trees", "nsc", "nsc_energy"]
    deep_models = ["attention_mil", "cellcnn", "cytoset", "dgcnn", "pointnet2"]
    baseline_models = ["flowsom"]

    df["model_type"] = "Other"
    df.loc[df["model"].isin(lot_models), "model_type"] = "FlowLOT (LOT)"
    df.loc[df["model"].isin(deep_models), "model_type"] = "Deep Sets / MIL"
    df.loc[df["model"].isin(baseline_models), "model_type"] = "FlowSOM (Gating)"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # --- Panel A: Top LOT vs Top Deep Models (Single Tube P2 & P1) ---
    ax_a = axes[0, 0]
    palette_a = {
        "LOT: Random Forest (P2)": "#1f77b4",
        "LOT: NSC (P1)": "#2ca02c",
        "LOT: Logistic (P1)": "#17becf",
        "Deep: DGCNN (P2)": "#ff7f0e",
        "Deep: CytoSet (P2)": "#e377c2",
        "Deep: Attention-MIL (P2)": "#d62728",
        "Baseline: FlowSOM (P2)": "#7f7f7f",
    }

    curve_configs = [
        ("random_forest", "single", "P2", "LOT: Random Forest (P2)", "o"),
        ("nsc", "single", "P1", "LOT: NSC (P1)", "s"),
        ("logistic", "single", "P1", "LOT: Logistic (P1)", "^"),
        ("dgcnn", "single", "P2", "Deep: DGCNN (P2)", "D"),
        ("cytoset", "single", "P2", "Deep: CytoSet (P2)", "v"),
        ("attention_mil", "single", "P2", "Deep: Attention-MIL (P2)", "x"),
        ("flowsom", "single", "P2", "Baseline: FlowSOM (P2)", "*"),
    ]

    for model, agg, tube, label, marker in curve_configs:
        sub = df[(df["model"] == model) & (df["aggregation"] == agg) & (df["tube"] == tube)].sort_values("k")
        if not sub.empty:
            ax_a.errorbar(
                sub["k"],
                sub["balanced_accuracy_mean"],
                yerr=sub["balanced_accuracy_std"],
                label=label,
                color=palette_a[label],
                marker=marker,
                capsize=3,
            )

    ax_a.set_title("(a) Top LOT Embeddings vs. Deep Learning Baselines", fontweight="bold")
    ax_a.set_xlabel("Training Patients Per Class (k)")
    ax_a.set_ylabel("Balanced Accuracy")
    ax_a.set_ylim(0.20, 0.90)
    ax_a.set_xticks([2, 4, 6, 8])
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="lower right", frameon=True, fontsize=8)

    # --- Panel B: Single Tube vs Multi-Tube Fusion (Random Forest & NSC) ---
    ax_b = axes[0, 1]
    rf_df = df[df["model"] == "random_forest"]
    fusion_colors = {
        "single P1": "#aec7e8",
        "single P2": "#1f77b4",
        "single P3": "#9edae5",
        "single P4": "#bcbd22",
        "early_mean ALL": "#d62728",
        "late_soft ALL": "#ff7f0e",
    }
    for (agg, tube), sub in rf_df.groupby(["aggregation", "tube"]):
        label = f"{agg} {tube}"
        sub = sub.sort_values("k")
        ax_b.plot(
            sub["k"],
            sub["balanced_accuracy_mean"],
            label=label,
            color=fusion_colors.get(label, "#333333"),
            marker="o",
            linestyle="-" if "ALL" in label else "--",
            linewidth=2.2 if "ALL" in label else 1.4,
        )

    ax_b.set_title("(b) Multi-Tube Fusion vs. Single Tubes (Random Forest)", fontweight="bold")
    ax_b.set_xlabel("Training Patients Per Class (k)")
    ax_b.set_ylabel("Balanced Accuracy")
    ax_b.set_ylim(0.50, 0.85)
    ax_b.set_xticks([2, 4, 6, 8])
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="lower right", frameon=True, fontsize=8)

    # --- Panel C: Accuracy vs. Compute Runtime (Efficiency Frontier) ---
    ax_c = axes[1, 0]
    k8 = df[df["k"] == 8].copy()
    model_summary = k8.groupby(["model", "model_type"])[["elapsed_seconds_mean", "balanced_accuracy_mean"]].mean().reset_index()

    type_colors = {"FlowLOT (LOT)": "#1f77b4", "Deep Sets / MIL": "#d62728", "FlowSOM (Gating)": "#2ca02c"}
    for mtype, grp in model_summary.groupby("model_type"):
        ax_c.scatter(
            grp["elapsed_seconds_mean"],
            grp["balanced_accuracy_mean"],
            label=mtype,
            color=type_colors[mtype],
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.8,
        )
        for _, row in grp.iterrows():
            offset_x = 1.12
            offset_y = 0.002
            ax_c.annotate(
                row["model"],
                (row["elapsed_seconds_mean"], row["balanced_accuracy_mean"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8,
            )

    ax_c.set_xscale("log")
    ax_c.set_title("(c) Performance vs. Compute Runtime (k=8)", fontweight="bold")
    ax_c.set_xlabel("Mean Training Time per Run (seconds, log scale)")
    ax_c.set_ylabel("Balanced Accuracy")
    ax_c.grid(True, linestyle="--", alpha=0.5, which="both")
    ax_c.legend(loc="lower right", frameon=True, fontsize=9)

    # --- Panel D: Bar chart ranking of all models at k=8 ---
    ax_d = axes[1, 1]
    sorted_models = model_summary.sort_values("balanced_accuracy_mean", ascending=True)
    colors_d = [type_colors[t] for t in sorted_models["model_type"]]
    bars = ax_d.barh(
        sorted_models["model"],
        sorted_models["balanced_accuracy_mean"],
        color=colors_d,
        edgecolor="black",
        linewidth=0.6,
        height=0.65,
    )
    ax_d.set_xlim(0.55, 0.78)
    ax_d.set_xlabel("Balanced Accuracy (k=8, mean across tube/fusion configs)")
    ax_d.set_title("(d) Model Ranking across BLAST110 Benchmark", fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax_d.text(w + 0.003, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    sns.despine(fig)
    plt.tight_layout()

    out_file = res_dir / "benchmark_clean_overview.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(res_dir / "benchmark_clean_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clean overview plot to {out_file}")

if __name__ == "__main__":
    main()
