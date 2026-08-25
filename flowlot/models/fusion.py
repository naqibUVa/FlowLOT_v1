"""Missing-tube-aware feature- and decision-level fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


FloatArray = NDArray[np.float64]
TubeData = Mapping[str, Mapping[str, ArrayLike]]


class EarlyTubeFusion:
    """Concatenate tube embeddings with zero/mean imputation and indicators."""

    def __init__(self, imputation: str = "mean", add_indicators: bool = True) -> None:
        if imputation not in {"mean", "zero"}:
            raise ValueError("imputation must be 'mean' or 'zero'")
        self.imputation = imputation
        self.add_indicators = add_indicators

    def fit(self, features: TubeData) -> "EarlyTubeFusion":
        self.tubes_ = sorted(features)
        self.dimensions_: dict[str, int] = {}
        self.fill_: dict[str, FloatArray] = {}
        for tube in self.tubes_:
            vectors = [np.asarray(value, dtype=float).reshape(-1) for value in features[tube].values()]
            if not vectors or len({len(value) for value in vectors}) != 1:
                raise ValueError(f"Tube {tube} has no vectors or inconsistent dimensions")
            self.dimensions_[tube] = len(vectors[0])
            self.fill_[tube] = (
                np.mean(vectors, axis=0) if self.imputation == "mean" else np.zeros(len(vectors[0]))
            )
        return self

    def transform(self, features: TubeData, patient_ids: Sequence[str]) -> FloatArray:
        if not hasattr(self, "tubes_"):
            raise RuntimeError("EarlyTubeFusion has not been fitted")
        rows = []
        for patient_id in patient_ids:
            values, indicators = [], []
            for tube in self.tubes_:
                available = patient_id in features.get(tube, {})
                vector = (
                    np.asarray(features[tube][patient_id], dtype=float).reshape(-1)
                    if available
                    else self.fill_[tube]
                )
                if len(vector) != self.dimensions_[tube]:
                    raise ValueError(f"Tube {tube}/{patient_id} has an unexpected dimension")
                values.append(vector)
                indicators.append(float(available))
            row = np.concatenate(values)
            if self.add_indicators:
                row = np.concatenate([row, indicators])
            rows.append(row)
        return np.stack(rows)

    def fit_transform(self, features: TubeData, patient_ids: Sequence[str]) -> FloatArray:
        return self.fit(features).transform(features, patient_ids)


class LateTubeFusion:
    """Fit per-tube estimators and aggregate only available patient tubes.

    ``method='stacking'`` builds out-of-fold tube predictions for its meta-model,
    avoiding the optimistic in-sample stacking used in many ad-hoc notebooks.
    """

    def __init__(
        self,
        estimator: Any,
        task: str = "classification",
        method: str = "mean",
        cv: int = 5,
        random_state: int = 0,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        allowed = {"classification": {"mean", "soft", "stacking"}, "regression": {"mean", "median", "stacking"}}
        if method not in allowed[task]:
            raise ValueError(f"Unsupported {task} fusion method: {method}")
        self.estimator = estimator
        self.task = task
        self.method = method
        self.cv = cv
        self.random_state = random_state

    def fit(
        self, features: TubeData, targets: Mapping[str, int | float]
    ) -> "LateTubeFusion":
        self.patient_ids_ = sorted(targets)
        def eligible(values: Mapping[str, ArrayLike]) -> bool:
            available = [targets[patient] for patient in self.patient_ids_ if patient in values]
            return len(available) >= 2 and (
                self.task == "regression" or len(np.unique(available)) >= 2
            )

        self.tubes_ = sorted(tube for tube, values in features.items() if eligible(values))
        if not self.tubes_:
            raise ValueError("No tube has features for the requested training patients")
        y_all = np.asarray([targets[patient] for patient in self.patient_ids_])
        self.classes_ = np.unique(y_all) if self.task == "classification" else None
        self.models_: dict[str, Any] = {}
        meta = self._empty_meta(len(self.patient_ids_))
        indicators = np.zeros((len(self.patient_ids_), len(self.tubes_)))
        patient_index = {patient: i for i, patient in enumerate(self.patient_ids_)}
        for tube_index, tube in enumerate(self.tubes_):
            ids = [patient for patient in self.patient_ids_ if patient in features[tube]]
            x = np.stack([np.asarray(features[tube][patient]).reshape(-1) for patient in ids])
            y = np.asarray([targets[patient] for patient in ids])
            if self.method == "stacking":
                self._fill_oof(meta, indicators, tube_index, ids, x, y, patient_index)
            model = clone(self.estimator).fit(x, y)
            self.models_[tube] = model
        if self.method == "stacking":
            meta = np.concatenate([meta, indicators], axis=1)
            self.meta_model_ = (
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state)
                if self.task == "classification"
                else Ridge(alpha=1.0)
            ).fit(meta, y_all)
        return self

    def _empty_meta(self, count: int) -> FloatArray:
        width = len(self.tubes_) * (len(self.classes_) if self.task == "classification" else 1)
        if self.task == "classification":
            prior = np.full(len(self.classes_), 1 / len(self.classes_))
            return np.tile(prior, (count, len(self.tubes_)))
        return np.zeros((count, width))

    def _prediction(self, model: Any, x: FloatArray) -> FloatArray:
        if self.task == "regression":
            return np.asarray(model.predict(x), dtype=float).reshape(-1, 1)
        raw = model.predict_proba(x)
        aligned = np.zeros((len(x), len(self.classes_)))
        for column, label in enumerate(model.classes_):
            aligned[:, np.where(self.classes_ == label)[0][0]] = raw[:, column]
        return aligned

    def _fill_oof(
        self,
        meta: FloatArray,
        indicators: FloatArray,
        tube_index: int,
        ids: list[str],
        x: FloatArray,
        y: np.ndarray,
        patient_index: Mapping[str, int],
    ) -> None:
        minimum_class = np.unique(y, return_counts=True)[1].min() if self.task == "classification" else len(y)
        folds = min(self.cv, len(y), int(minimum_class))
        if folds < 2:
            raise ValueError("Stacking needs at least two samples per fold/class in every tube")
        splitter = (
            StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
            if self.task == "classification"
            else KFold(folds, shuffle=True, random_state=self.random_state)
        )
        width = len(self.classes_) if self.task == "classification" else 1
        offset = tube_index * width
        for train, validation in splitter.split(x, y if self.task == "classification" else None):
            model = clone(self.estimator).fit(x[train], y[train])
            predictions = self._prediction(model, x[validation])
            for local, prediction in zip(validation, predictions):
                row = patient_index[ids[local]]
                meta[row, offset : offset + width] = prediction
                indicators[row, tube_index] = 1

    def _meta_features(self, features: TubeData, patient_ids: Sequence[str]) -> tuple[FloatArray, list[list[FloatArray]]]:
        meta = self._empty_meta(len(patient_ids))
        indicators = np.zeros((len(patient_ids), len(self.tubes_)))
        available: list[list[FloatArray]] = [[] for _ in patient_ids]
        width = len(self.classes_) if self.task == "classification" else 1
        for tube_index, tube in enumerate(self.tubes_):
            rows = [index for index, patient in enumerate(patient_ids) if patient in features.get(tube, {})]
            if not rows:
                continue
            x = np.stack([np.asarray(features[tube][patient_ids[index]]).reshape(-1) for index in rows])
            predictions = self._prediction(self.models_[tube], x)
            offset = tube_index * width
            for row, prediction in zip(rows, predictions):
                meta[row, offset : offset + width] = prediction
                indicators[row, tube_index] = 1
                available[row].append(prediction)
        if np.any(indicators.sum(1) == 0):
            raise ValueError("At least one requested patient has no available tube")
        return np.concatenate([meta, indicators], axis=1), available

    def predict_proba(self, features: TubeData, patient_ids: Sequence[str]) -> FloatArray:
        if self.task != "classification":
            raise AttributeError("predict_proba is classification-only")
        meta, available = self._meta_features(features, patient_ids)
        if self.method == "stacking":
            return self.meta_model_.predict_proba(meta)
        return np.stack([np.mean(values, axis=0) for values in available])

    def predict(self, features: TubeData, patient_ids: Sequence[str]) -> np.ndarray:
        meta, available = self._meta_features(features, patient_ids)
        if self.method == "stacking":
            return self.meta_model_.predict(meta)
        if self.task == "classification":
            return self.classes_[np.stack([np.mean(values, axis=0) for values in available]).argmax(1)]
        reducer = np.median if self.method == "median" else np.mean
        return np.asarray([reducer(np.concatenate(values)) for values in available])
