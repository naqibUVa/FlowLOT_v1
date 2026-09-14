"""Classical LOT-vector classifiers, including legacy nearest-subspace variants."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FloatArray = NDArray[np.float64]


class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator):
    """Mean-centered class subspaces scored by reconstruction distance."""

    def __init__(
        self,
        n_components: int | str | None = 5,
        energy_threshold: float | None = None,
        centered: bool = True,
    ) -> None:
        self.n_components = n_components
        self.energy_threshold = energy_threshold
        self.centered = centered

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "NearestSubspaceClassifier":
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.means_: dict[Any, FloatArray] = {}
        self.subspaces_: dict[Any, FloatArray] = {}
        for label in self.classes_:
            values = features[labels == label]
            if self.centered:
                mean = values.mean(axis=0)
                centered = values - mean
            else:
                mean = np.zeros(values.shape[1], dtype=float)
                centered = values
            basis, singular, _ = np.linalg.svd(centered.T, full_matrices=False)
            if self.energy_threshold is not None and np.square(singular).sum() > 0:
                cumulative = np.cumsum(np.square(singular)) / np.square(singular).sum()
                rank = int(np.searchsorted(cumulative, self.energy_threshold) + 1)
            elif (
                self.n_components is not None
                and self.n_components != "full"
                and isinstance(self.n_components, (int, np.integer))
                and self.n_components > 0
            ):
                rank = int(self.n_components)
            else:
                rank = basis.shape[1]
            max_possible_rank = max(len(values) - 1, 1) if self.centered else len(values)
            rank = max(1, min(rank, max_possible_rank, basis.shape[1]))
            self.means_[label] = mean
            self.subspaces_[label] = basis[:, :rank]
        return self

    def decision_function(self, features: ArrayLike) -> FloatArray:
        features = np.asarray(features, dtype=float)
        distances = np.empty((len(features), len(self.classes_)))
        for column, label in enumerate(self.classes_):
            centered = features - self.means_[label]
            basis = self.subspaces_[label]
            residual = centered - (centered @ basis) @ basis.T
            distances[:, column] = np.linalg.norm(residual, axis=1)
        return -distances

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        scores = self.decision_function(features)
        scores -= scores.max(axis=1, keepdims=True)
        exponential = np.exp(scores)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def predict(self, features: ArrayLike) -> np.ndarray:
        return self.classes_[self.decision_function(features).argmax(axis=1)]


CLASSICAL_MODELS = (
    "logistic",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "nsc",
    "nsc_energy",
    "nsc_full",
    "xgboost",
)
CELL_MODELS = ("flowsom", "cellcnn", "attention_mil", "cytoset", "dgcnn", "pointnet2")


def make_classical_classifier(name: str, random_state: int = 0) -> Any:
    name = name.lower()
    if name == "logistic":
        estimator = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=random_state
        )
        return make_pipeline(StandardScaler(), estimator)
    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    if name == "nsc":
        return make_pipeline(StandardScaler(), NearestSubspaceClassifier(n_components=5))
    if name == "nsc_energy":
        return make_pipeline(
            StandardScaler(), NearestSubspaceClassifier(energy_threshold=0.95)
        )
    if name == "nsc_full":
        return make_pipeline(
            StandardScaler(), NearestSubspaceClassifier(n_components="full")
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("XGBoost requires `pip install flowlot[xgboost]`") from error
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unknown classical classifier: {name}")
