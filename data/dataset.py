"""Unified datasets and batching for cytometry bags.

Each subject is represented by an ``[n_cells, n_markers]`` matrix.  A manifest is
a CSV file with at least ``path,label,split`` columns; ``sample_id`` is optional.
Supported sample files are ``.npy``, ``.npz`` (key ``cells`` or its sole array),
``.pt``/``.pth`` tensors, and numeric ``.csv``/``.txt`` matrices.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    split: str
    sample_id: str


def _read_cells(path: Path, npz_key: str = "cells") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif suffix == ".npz":
        with np.load(path) as archive:
            if npz_key in archive:
                array = archive[npz_key]
            elif len(archive.files) == 1:
                array = archive[archive.files[0]]
            else:
                raise KeyError(f"{path} has multiple arrays and no {npz_key!r} key")
    elif suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, Mapping):
            value = value.get(npz_key, value.get("x"))
        if value is None or not torch.is_tensor(value):
            raise TypeError(f"{path} does not contain a tensor")
        array = value.detach().cpu().numpy()
    elif suffix in {".csv", ".txt"}:
        array = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported sample format: {path.suffix}")
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"Expected a non-empty 2D cell matrix in {path}, got {array.shape}")
    return array


def _subsample_indices(n: int, maximum: int | None, random: bool, seed: int) -> np.ndarray:
    if maximum is None or n <= maximum:
        return np.arange(n)
    if random:
        # NumPy's process RNG is seeded by the trainer (and by DataLoader workers),
        # giving fresh epoch subsets while preserving run-level reproducibility.
        return np.random.choice(n, size=maximum, replace=False)
    return np.random.default_rng(seed).choice(n, size=maximum, replace=False)


class CytometryDataset(Dataset[dict[str, Any]]):
    """Lazy manifest-backed cytometry dataset.

    Training datasets normally set ``random_subsample=True`` so a different cell
    subset is observed each epoch. Validation and test datasets use a deterministic
    subset derived from ``seed``.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        max_cells: int | None = 4096,
        random_subsample: bool = False,
        seed: int = 0,
        npz_key: str = "cells",
    ) -> None:
        if not records:
            raise ValueError("The dataset contains no samples")
        if max_cells is not None and max_cells < 1:
            raise ValueError("max_cells must be positive or None")
        self.records = list(records)
        self.max_cells = max_cells
        self.random_subsample = random_subsample
        self.seed = seed
        self.npz_key = npz_key

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        split: str,
        **kwargs: Any,
    ) -> "CytometryDataset":
        manifest = Path(manifest).resolve()
        records: list[SampleRecord] = []
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "label", "split"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
            for row in reader:
                if row["split"].strip().lower() != split.lower():
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                sample_id = row.get("sample_id") or path.stem
                records.append(SampleRecord(path, int(row["label"]), split, sample_id))
        return cls(records, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cells = _read_cells(record.path, self.npz_key)
        chosen = _subsample_indices(
            len(cells), self.max_cells, self.random_subsample, self.seed + index
        )
        # Copy after indexing to detach tensors from read-only memory maps.
        tensor = torch.as_tensor(np.asarray(cells[chosen], dtype=np.float32).copy())
        return {"cells": tensor, "label": record.label, "sample_id": record.sample_id}


class InMemoryCytometryDataset(Dataset[dict[str, Any]]):
    """Convenient in-memory equivalent, useful for notebooks and tests."""

    def __init__(
        self,
        samples: Sequence[np.ndarray | torch.Tensor],
        labels: Sequence[int],
        max_cells: int | None = None,
        random_subsample: bool = False,
        seed: int = 0,
        sample_ids: Sequence[str] | None = None,
    ) -> None:
        if len(samples) != len(labels) or not samples:
            raise ValueError("samples and labels must be non-empty and have equal length")
        self.samples = samples
        self.labels = labels
        self.max_cells = max_cells
        self.random_subsample = random_subsample
        self.seed = seed
        self.sample_ids = sample_ids or [str(i) for i in range(len(samples))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cells = torch.as_tensor(self.samples[index], dtype=torch.float32)
        if cells.ndim != 2 or not cells.numel():
            raise ValueError(f"Sample {index} must be a non-empty 2D matrix")
        chosen = _subsample_indices(
            cells.shape[0], self.max_cells, self.random_subsample, self.seed + index
        )
        return {
            "cells": cells[torch.as_tensor(chosen)].clone(),
            "label": int(self.labels[index]),
            "sample_id": self.sample_ids[index],
        }


def cytometry_collate(batch: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad variable-size bags and return an explicit valid-cell mask."""

    items = list(batch)
    if not items:
        raise ValueError("Cannot collate an empty batch")
    dimensions = {item["cells"].shape[1] for item in items}
    if len(dimensions) != 1:
        raise ValueError(f"All samples must use the same marker count, got {dimensions}")
    batch_size = len(items)
    max_cells = max(item["cells"].shape[0] for item in items)
    n_features = dimensions.pop()
    cells = torch.zeros(batch_size, max_cells, n_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item["cells"].shape[0]
        cells[i, :n] = item["cells"]
        mask[i, :n] = True
    return {
        "cells": cells,
        "mask": mask,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "sample_ids": [str(item["sample_id"]) for item in items],
    }
