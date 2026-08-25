"""Standard sample-level classification metrics."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    labels: ArrayLike, probabilities: ArrayLike
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("probabilities must be [samples, classes]")
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if probabilities.shape[1] == 2:
                result["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
                result["pr_auc"] = float(average_precision_score(labels, probabilities[:, 1]))
            else:
                classes = np.arange(probabilities.shape[1])
                targets = label_binarize(labels, classes=classes)
                result["roc_auc"] = float(
                    roc_auc_score(targets, probabilities, average="macro", multi_class="ovr")
                )
                result["pr_auc"] = float(
                    average_precision_score(targets, probabilities, average="macro")
                )
    except ValueError:
        # A split without every class cannot define one-vs-rest AUC.
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result
