"""Standardized classification and regression evaluators."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(labels: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("probabilities must have shape [samples, classes]")
    observed_classes = np.unique(labels)
    if np.issubdtype(labels.dtype, np.integer) and np.all(
        (labels >= 0) & (labels < probabilities.shape[1])
    ):
        # Encoded CV folds may legitimately omit one class while the estimator
        # still emits a column for every global class.
        classes = np.arange(probabilities.shape[1])
    elif len(observed_classes) == probabilities.shape[1]:
        classes = observed_classes
    else:
        raise ValueError("Probability columns must match the encoded label space")
    predicted = classes[probabilities.argmax(1)]
    result = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if probabilities.shape[1] == 2:
                binary = (labels == classes[1]).astype(int)
                result["roc_auc"] = float(roc_auc_score(binary, probabilities[:, 1]))
                result["pr_auc"] = float(average_precision_score(binary, probabilities[:, 1]))
            else:
                targets = label_binarize(labels, classes=classes)
                result["roc_auc"] = float(
                    roc_auc_score(targets, probabilities, average="macro", multi_class="ovr")
                )
                result["pr_auc"] = float(
                    average_precision_score(targets, probabilities, average="macro")
                )
    except ValueError:
        result.update(roc_auc=float("nan"), pr_auc=float("nan"))
    return result


def regression_metrics(labels: ArrayLike, predictions: ArrayLike) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if labels.shape != predictions.shape or not len(labels):
        raise ValueError("labels and predictions must be non-empty equal-length vectors")
    pearson = pearsonr(labels, predictions).statistic if len(labels) > 1 else float("nan")
    spearman = spearmanr(labels, predictions).statistic if len(labels) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(mean_squared_error(labels, predictions) ** 0.5),
        "r2": float(r2_score(labels, predictions)),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
    }
