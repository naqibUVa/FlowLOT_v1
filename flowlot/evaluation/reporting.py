"""Nature-style vector figures for prediction and transport diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_nature_style() -> None:
    mpl.rcParams.update(NATURE_STYLE)
    sns.set_palette("colorblind")


def save_figure(fig: plt.Figure, output_prefix: str | Path) -> list[Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = prefix.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        outputs.append(path)
    return outputs


def _binary_roc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    bootstrap: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0, 1, 201)
    curves = []
    rng = np.random.default_rng(random_state)
    for _ in range(bootstrap):
        indices = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[indices], scores[indices])
        curves.append(np.interp(grid, fpr, tpr))
    if not curves:
        nan = np.full_like(grid, np.nan)
        return grid, nan, nan, nan
    matrix = np.stack(curves)
    return grid, np.mean(matrix, 0), np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


def plot_classification_suite(
    labels: ArrayLike,
    probabilities: ArrayLike,
    output_prefix: str | Path,
    class_names: Sequence[str] | None = None,
    bootstrap: int = 1000,
    random_state: int = 0,
) -> list[Path]:
    apply_nature_style()
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.arange(probabilities.shape[1])
    names = list(class_names or [str(value) for value in classes])
    targets = label_binarize(labels, classes=classes)
    if probabilities.shape[1] == 2:
        targets = np.column_stack([1 - labels, labels])
    colors = sns.color_palette("colorblind", probabilities.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for class_index, (name, color) in enumerate(zip(names, colors)):
        binary = targets[:, class_index]
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, probabilities[:, class_index])
        axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        if bootstrap:
            grid, _, lower, upper = _binary_roc_ci(
                binary, probabilities[:, class_index], bootstrap, random_state + class_index
            )
            axes[0].fill_between(grid, lower, upper, color=color, alpha=0.12, lw=0)
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, class_index])
        axes[1].plot(recall, precision, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (95% bootstrap CI)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(labels, predicted, labels=classes, normalize="true")
    sns.heatmap(
        matrix,
        ax=axes[2],
        cmap="cividis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Fraction"},
    )
    axes[2].set(xlabel="Predicted", ylabel="Observed", title="Normalized confusion")
    sns.despine(fig)
    fig.tight_layout()
    outputs = save_figure(fig, output_prefix)
    plt.close(fig)
    return outputs


def plot_regression_suite(
    labels: ArrayLike,
    predictions: ArrayLike,
    output_prefix: str | Path,
    target_name: str = "Outcome",
) -> list[Path]:
    apply_nature_style()
    labels = np.asarray(labels, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    residuals = predictions - labels
    means = (predictions + labels) / 2
    bias, spread = residuals.mean(), residuals.std(ddof=1)
    color = sns.color_palette("colorblind")[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    axes[0].scatter(labels, predictions, s=18, alpha=0.75, color=color, edgecolor="white", lw=0.3)
    limits = [min(labels.min(), predictions.min()), max(labels.max(), predictions.max())]
    axes[0].plot(limits, limits, "--", color="0.3", lw=0.9)
    axes[0].set(xlabel=f"Observed {target_name}", ylabel=f"Predicted {target_name}", title="Prediction parity")
    sns.histplot(residuals, kde=True, ax=axes[1], color=color, edgecolor="white")
    axes[1].axvline(0, ls="--", color="0.3", lw=0.9)
    axes[1].set(xlabel="Residual", title="Residual distribution")
    axes[2].scatter(means, residuals, s=18, alpha=0.75, color=color, edgecolor="white", lw=0.3)
    for value, style in [(bias, "-"), (bias - 1.96 * spread, "--"), (bias + 1.96 * spread, "--")]:
        axes[2].axhline(value, color="0.3", ls=style, lw=0.9)
    axes[2].set(xlabel="Pair mean", ylabel="Predicted − observed", title="Bland–Altman")
    sns.despine(fig)
    fig.tight_layout()
    outputs = save_figure(fig, output_prefix)
    plt.close(fig)
    return outputs


def plot_transport_geometry(
    reference: ArrayLike,
    sample: ArrayLike,
    transported: ArrayLike,
    output_prefix: str | Path,
    max_vectors: int = 100,
    projection: str = "pca",
    dimensions: int = 2,
) -> list[Path]:
    """Plot reference, sample, and LOT vectors in a 2D/3D PCA or UMAP projection."""

    apply_nature_style()
    reference = np.asarray(reference)
    sample = np.asarray(sample)
    transported = np.asarray(transported)
    if dimensions not in {2, 3}:
        raise ValueError("dimensions must be 2 or 3")
    combined = np.vstack([reference, sample, transported])
    if projection.lower() == "pca":
        projector = PCA(dimensions).fit(combined)
        axis_prefix = "PC"
    elif projection.lower() == "umap":
        try:
            from umap import UMAP
        except ImportError as error:
            raise ImportError("UMAP projection requires `pip install flowlot[notebook]`") from error
        projector = UMAP(n_components=dimensions, random_state=0).fit(combined)
        axis_prefix = "UMAP"
    else:
        raise ValueError("projection must be 'pca' or 'umap'")
    ref_projected, sample_projected, moved_projected = map(
        projector.transform, [reference, sample, transported]
    )
    fig = plt.figure(figsize=(3.5, 3.0))
    ax = fig.add_subplot(111, projection="3d" if dimensions == 3 else None)
    coordinates = list(range(dimensions))
    ax.scatter(
        *[sample_projected[:, value] for value in coordinates],
        s=5,
        alpha=0.18,
        color="#999999",
        label="Sample cells",
    )
    ax.scatter(
        *[ref_projected[:, value] for value in coordinates],
        s=9,
        color="#0072B2",
        label="Reference",
    )
    selected = np.linspace(
        0, len(ref_projected) - 1, min(max_vectors, len(ref_projected)), dtype=int
    )
    delta = moved_projected[selected] - ref_projected[selected]
    if dimensions == 2:
        ax.quiver(
            ref_projected[selected, 0], ref_projected[selected, 1], delta[:, 0], delta[:, 1],
            angles="xy", scale_units="xy", scale=1, width=0.003, color="#D55E00", alpha=0.75,
            label="LOT displacement",
        )
        ax.set(xlabel=f"{axis_prefix}1", ylabel=f"{axis_prefix}2", title="Transport geometry")
    else:
        ax.quiver(
            ref_projected[selected, 0], ref_projected[selected, 1], ref_projected[selected, 2],
            delta[:, 0], delta[:, 1], delta[:, 2], color="#D55E00", alpha=0.75,
            label="LOT displacement",
        )
        ax.set(
            xlabel=f"{axis_prefix}1",
            ylabel=f"{axis_prefix}2",
            zlabel=f"{axis_prefix}3",
            title="Transport geometry",
        )
    ax.legend(frameon=False)
    if dimensions == 2:
        sns.despine(ax=ax)
    fig.tight_layout()
    outputs = save_figure(fig, output_prefix)
    plt.close(fig)
    return outputs


def plot_clinical_threshold_suite(
    observed: ArrayLike,
    predicted: ArrayLike,
    output_prefix: str | Path,
    thresholds: Sequence[float] = (0.1, 1.0, 5.0),
    groups: Sequence[str] | None = None,
) -> list[Path]:
    """Reproduce the legacy BLAST/LAIP decision, Youden, and ROC panel logic."""

    apply_nature_style()
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    groups_array = np.asarray(groups if groups is not None else ["sample"] * len(observed))
    fig, axes = plt.subplots(3, len(thresholds), figsize=(7.2, 6.4), squeeze=False)
    palette = {"TP": "#009E73", "TN": "#0072B2", "FP": "#E69F00", "FN": "#D55E00"}
    markers = {group: marker for group, marker in zip(np.unique(groups_array), ["o", "^", "s", "D"])}
    for column, threshold in enumerate(thresholds):
        truth = observed >= threshold
        grid = np.geomspace(max(predicted[predicted > 0].min(initial=1e-6), 1e-6), max(predicted.max(), 1e-5), 250)
        sensitivity, specificity = [], []
        for cutoff in grid:
            call = predicted >= cutoff
            sensitivity.append(np.sum(truth & call) / max(np.sum(truth), 1))
            specificity.append(np.sum(~truth & ~call) / max(np.sum(~truth), 1))
        sensitivity, specificity = np.asarray(sensitivity), np.asarray(specificity)
        best = int(np.argmax(sensitivity + specificity - 1))
        cutoff = grid[best]
        call = predicted >= cutoff
        category = np.where(truth & call, "TP", np.where(~truth & ~call, "TN", np.where(~truth & call, "FP", "FN")))
        for group, marker in markers.items():
            for name, color in palette.items():
                mask = (groups_array == group) & (category == name)
                if np.any(mask):
                    axes[0, column].scatter(observed[mask], predicted[mask], s=22, marker=marker, color=color, edgecolor="white", lw=0.3)
        axes[0, column].axvline(threshold, ls=":", color="0.4")
        axes[0, column].axhline(cutoff, ls="--", color="#CC79A7")
        axes[0, column].set(xscale="log", yscale="log", xlabel="Observed (%)", ylabel="Predicted (%)", title=f"Decision @ {threshold:g}%")
        axes[1, column].plot(grid, sensitivity, label="Sensitivity", color="#D55E00")
        axes[1, column].plot(grid, specificity, label="Specificity", color="#0072B2")
        axes[1, column].plot(grid, sensitivity + specificity - 1, label="Youden J", color="0.4", ls="--")
        axes[1, column].axvline(cutoff, color="#CC79A7", ls=":")
        axes[1, column].set(xscale="log", xlabel="Predicted cutoff (%)", ylabel="Rate", title=f"Optimal cutoff {cutoff:.2g}%")
        if len(np.unique(truth)) == 2:
            fpr, tpr, _ = roc_curve(truth.astype(int), predicted)
            axes[2, column].plot(fpr, tpr, color="#009E73", label=f"AUC={auc(fpr, tpr):.2f}")
        else:
            axes[2, column].text(0.5, 0.5, "One observed class", ha="center", va="center")
        axes[2, column].plot([0, 1], [0, 1], "--", color="0.5")
        axes[2, column].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC @ {threshold:g}%")
        axes[1, column].legend(frameon=False)
        axes[2, column].legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    outputs = save_figure(fig, output_prefix)
    plt.close(fig)
    return outputs
