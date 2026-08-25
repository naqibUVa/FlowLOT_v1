"""Read-only convenience accessor for Stage 2 analytics files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


class Stage2Loader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.h5: h5py.File | None = None

    def __enter__(self) -> "Stage2Loader":
        self.h5 = h5py.File(self.path, "r")
        if self.h5.attrs.get("schema") != "flowlot-stage2":
            raise ValueError(f"{self.path} is not a FlowLOT Stage 2 file")
        return self

    def __exit__(self, *_: object) -> None:
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def _file(self) -> h5py.File:
        if self.h5 is None:
            raise RuntimeError("Use Stage2Loader as a context manager")
        return self.h5

    def tubes(self, dataset: str, cell_count: str | int) -> list[str]:
        return sorted(self._file()[f"{dataset}/{cell_count}"].keys())

    def metadata(self, dataset: str, cell_count: str | int, tube: str) -> dict[str, object]:
        group = self._file()[f"{dataset}/{cell_count}/{tube}/metadata"]
        labels = group["labels"][...]
        if labels.dtype.kind in {"S", "O", "U"}:
            labels = np.asarray(_strings(labels))
        return {
            "patient_ids": _strings(group["patient_ids"][...]),
            "labels": labels,
            "cell_counts": group["cell_counts"][...],
            "marker_descriptions": _strings(group["marker_descriptions"][...]),
        }

    def cells(
        self,
        dataset: str,
        cell_count: str | int,
        tube: str,
        patient_id: str,
        preprocess_id: str | None = None,
    ) -> np.ndarray:
        if preprocess_id is None:
            path = f"{dataset}/{cell_count}/{tube}/raw/{patient_id}/raw_cell_matrix"
        else:
            path = f"{dataset}/{cell_count}/{tube}/preprocess_{preprocess_id}/{patient_id}/processed_matrix"
        return self._file()[path][...]

    def embeddings(
        self,
        dataset: str,
        cell_count: str | int,
        tube: str,
        preprocess_id: str,
        embedding_id: str,
    ) -> tuple[list[str], np.ndarray]:
        group = self._file()[
            f"{dataset}/{cell_count}/{tube}/lot_embeddings/{preprocess_id}/{embedding_id}"
        ]
        return _strings(group["patient_ids"][...]), group["embeddings"][...]
