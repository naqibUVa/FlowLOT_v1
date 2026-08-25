"""Reference distribution generation for LOT embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _samples(values: Sequence[ArrayLike]) -> list[FloatArray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays or any(value.ndim != 2 or not len(value) for value in arrays):
        raise ValueError("Reference samples must be non-empty [cells, markers] arrays")
    if len({value.shape[1] for value in arrays}) != 1:
        raise ValueError("Reference samples have inconsistent marker dimensions")
    return arrays


class ReferenceFactory:
    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state

    def patient(self, samples: Sequence[ArrayLike], index: int = 0, size: int | None = None) -> FloatArray:
        arrays = _samples(samples)
        if not -len(arrays) <= index < len(arrays):
            raise IndexError("Patient reference index is out of range")
        return self._resize(arrays[index], size)

    def uniform(self, samples: Sequence[ArrayLike], size: int) -> FloatArray:
        arrays = _samples(samples)
        lower = np.minimum.reduce([array.min(axis=0) for array in arrays])
        upper = np.maximum.reduce([array.max(axis=0) for array in arrays])
        rng = np.random.default_rng(self.random_state)
        return rng.uniform(lower, upper, size=(size, arrays[0].shape[1]))

    def gaussian(self, samples: Sequence[ArrayLike], size: int, jitter: float = 1e-6) -> FloatArray:
        arrays = _samples(samples)
        count = sum(len(array) for array in arrays)
        total = sum((array.sum(axis=0) for array in arrays), np.zeros(arrays[0].shape[1]))
        second_moment = sum(
            (array.T @ array for array in arrays),
            np.zeros((arrays[0].shape[1], arrays[0].shape[1])),
        )
        mean = total / count
        covariance = (second_moment - count * np.outer(mean, mean)) / max(count - 1, 1)
        covariance += jitter * np.eye(arrays[0].shape[1])
        return np.random.default_rng(self.random_state).multivariate_normal(
            mean, covariance, size=size
        )

    def pooled(self, samples: Sequence[ArrayLike], size: int) -> FloatArray:
        arrays = _samples(samples)
        lengths = np.asarray([len(array) for array in arrays])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        total = int(offsets[-1])
        rng = np.random.default_rng(self.random_state)
        global_indices = rng.choice(total, size=size, replace=total < size)
        sample_indices = np.searchsorted(offsets[1:], global_indices, side="right")
        local_indices = global_indices - offsets[sample_indices]
        return np.stack(
            [arrays[sample][local] for sample, local in zip(sample_indices, local_indices)]
        )

    def barycenter(
        self,
        samples: Sequence[ArrayLike],
        size: int,
        max_patients: int = 10,
        max_cells_per_patient: int = 2048,
        max_iter: int = 50,
    ) -> FloatArray:
        arrays = _samples(samples)[:max_patients]
        rng = np.random.default_rng(self.random_state)
        arrays = [
            array[
                rng.choice(
                    len(array),
                    size=min(len(array), max_cells_per_patient),
                    replace=False,
                )
            ]
            for array in arrays
        ]
        try:
            import ot
        except ImportError as error:
            raise ImportError("Wasserstein barycenters require the POT dependency") from error
        initial = self.pooled(arrays, size)
        weights = [np.full(len(array), 1.0 / len(array)) for array in arrays]
        bary_weights = np.full(size, 1.0 / size)
        return np.asarray(
            ot.lp.free_support_barycenter(
                arrays,
                weights,
                initial,
                b=bary_weights,
                numItermax=max_iter,
                verbose=False,
            ),
            dtype=np.float64,
        )

    def create(
        self,
        kind: str,
        samples: Sequence[ArrayLike],
        size: int,
        **kwargs: Any,
    ) -> FloatArray:
        normalized = kind.lower().replace("_", "")
        if normalized in {"patient", "patient0", "patientreference"}:
            return self.patient(samples, size=size, **kwargs)
        if normalized in {"uniform", "syntheticuniform"}:
            return self.uniform(samples, size=size)
        if normalized in {"gaussian", "syntheticgaussian"}:
            return self.gaussian(samples, size=size, **kwargs)
        if normalized in {"pooled", "pooledsubsample", "subsampled"}:
            return self.pooled(samples, size=size)
        if normalized in {"barycenter", "wassersteinbarycenter"}:
            return self.barycenter(samples, size=size, **kwargs)
        raise ValueError(f"Unknown reference type: {kind}")

    def _resize(self, values: FloatArray, size: int | None) -> FloatArray:
        if size is None or size == len(values):
            return values.copy()
        rng = np.random.default_rng(self.random_state)
        return values[rng.choice(len(values), size=size, replace=len(values) < size)].copy()
