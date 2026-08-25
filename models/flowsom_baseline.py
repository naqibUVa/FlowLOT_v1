"""FlowSOM-style sample features with a leakage-safe downstream classifier.

This implementation keeps the essential FlowSOM workflow self-contained: marker
standardisation, a two-dimensional self-organising map, meta-clustering of SOM
codes, and sample-level cluster abundance/median-intensity features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]


@dataclass
class SelfOrganizingMap:
    grid: tuple[int, int] = (10, 10)
    iterations: int = 10_000
    learning_rate: float = 0.05
    sigma: float | None = None
    random_state: int = 0

    def fit(self, x: FloatArray) -> "SelfOrganizingMap":
        if x.ndim != 2 or not len(x):
            raise ValueError("SOM input must be a non-empty 2D array")
        rng = np.random.default_rng(self.random_state)
        node_count = self.grid[0] * self.grid[1]
        initial = rng.choice(len(x), node_count, replace=len(x) < node_count)
        self.codebook_ = x[initial].copy()
        rows, columns = np.unravel_index(np.arange(node_count), self.grid)
        locations = np.column_stack([rows, columns])
        grid_distance = ((locations[:, None] - locations[None, :]) ** 2).sum(axis=-1)
        initial_sigma = self.sigma or max(self.grid) / 2
        for step in range(self.iterations):
            observation = x[rng.integers(len(x))]
            winner = np.square(self.codebook_ - observation).sum(axis=1).argmin()
            progress = step / max(self.iterations - 1, 1)
            rate = self.learning_rate * np.exp(-3 * progress)
            radius = max(initial_sigma * np.exp(-3 * progress), 0.5)
            influence = np.exp(-grid_distance[winner] / (2 * radius**2))[:, None]
            self.codebook_ += rate * influence * (observation - self.codebook_)
        return self

    def predict(self, x: FloatArray, chunk_size: int = 8192) -> NDArray[np.int64]:
        if not hasattr(self, "codebook_"):
            raise RuntimeError("SOM has not been fitted")
        assignments = []
        for start in range(0, len(x), chunk_size):
            values = x[start : start + chunk_size]
            distance = ((values[:, None] - self.codebook_[None, :]) ** 2).sum(axis=-1)
            assignments.append(distance.argmin(axis=1))
        return np.concatenate(assignments).astype(np.int64, copy=False)


class FlowSOMFeatureExtractor:
    def __init__(
        self,
        grid: tuple[int, int] = (10, 10),
        n_meta_clusters: int = 20,
        som_iterations: int = 10_000,
        max_training_cells: int = 200_000,
        max_cells_per_sample: int = 20_000,
        random_state: int = 0,
    ) -> None:
        self.grid = grid
        self.n_meta_clusters = n_meta_clusters
        self.som_iterations = som_iterations
        self.max_training_cells = max_training_cells
        self.max_cells_per_sample = max_cells_per_sample
        self.random_state = random_state

    @staticmethod
    def _validate(samples: Sequence[ArrayLike]) -> list[FloatArray]:
        arrays = [np.asarray(sample, dtype=np.float64) for sample in samples]
        if not arrays or any(value.ndim != 2 or not len(value) for value in arrays):
            raise ValueError("samples must contain non-empty [cells, markers] arrays")
        dimensions = {value.shape[1] for value in arrays}
        if len(dimensions) != 1:
            raise ValueError("All samples must have the same marker count")
        return arrays

    def fit(self, samples: Sequence[ArrayLike]) -> "FlowSOMFeatureExtractor":
        arrays = self._validate(samples)
        rng = np.random.default_rng(self.random_state)
        selected = []
        for values in arrays:
            count = min(len(values), self.max_cells_per_sample)
            selected.append(values[rng.choice(len(values), count, replace=False)])
        pooled = np.concatenate(selected)
        if len(pooled) > self.max_training_cells:
            pooled = pooled[rng.choice(len(pooled), self.max_training_cells, replace=False)]
        self.scaler_ = StandardScaler().fit(pooled)
        pooled = self.scaler_.transform(pooled)
        self.global_median_ = np.median(pooled, axis=0)
        self.som_ = SelfOrganizingMap(
            self.grid, self.som_iterations, random_state=self.random_state
        ).fit(pooled)
        cluster_count = min(self.n_meta_clusters, len(self.som_.codebook_))
        self.meta_cluster_ = KMeans(
            cluster_count, n_init=10, random_state=self.random_state
        ).fit(self.som_.codebook_)
        self.n_meta_clusters_ = cluster_count
        return self

    def transform(self, samples: Sequence[ArrayLike]) -> FloatArray:
        if not hasattr(self, "som_"):
            raise RuntimeError("FlowSOMFeatureExtractor has not been fitted")
        arrays = self._validate(samples)
        rows = []
        for sample in arrays:
            values = self.scaler_.transform(sample)
            nodes = self.som_.predict(values)
            clusters = self.meta_cluster_.labels_[nodes]
            abundance = np.bincount(clusters, minlength=self.n_meta_clusters_) / len(values)
            medians = np.stack(
                [
                    np.median(values[clusters == cluster], axis=0)
                    if np.any(clusters == cluster)
                    else self.global_median_
                    for cluster in range(self.n_meta_clusters_)
                ]
            ).reshape(-1)
            rows.append(np.concatenate([abundance, medians]))
        return np.stack(rows)

    def fit_transform(self, samples: Sequence[ArrayLike]) -> FloatArray:
        return self.fit(samples).transform(samples)


class FlowSOMClassifier:
    """FlowSOM feature extractor plus class-balanced logistic regression."""

    def __init__(self, random_state: int = 0, **extractor_kwargs: object) -> None:
        self.extractor = FlowSOMFeatureExtractor(random_state=random_state, **extractor_kwargs)
        self.classifier = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        )

    def fit(self, samples: Sequence[ArrayLike], labels: ArrayLike) -> "FlowSOMClassifier":
        features = self.extractor.fit_transform(samples)
        self.feature_scaler_ = StandardScaler().fit(features)
        self.classifier.fit(self.feature_scaler_.transform(features), np.asarray(labels))
        return self

    def predict_proba(self, samples: Sequence[ArrayLike]) -> FloatArray:
        features = self.extractor.transform(samples)
        return self.classifier.predict_proba(self.feature_scaler_.transform(features))

    def predict(self, samples: Sequence[ArrayLike]) -> NDArray[np.int64]:
        return self.classifier.classes_[self.predict_proba(samples).argmax(axis=1)]
