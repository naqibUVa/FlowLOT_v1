"""Build the raw, patient-centric Stage 1 HDF5 file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray


UTF8 = h5py.string_dtype("utf-8")


def _safe_key(value: object) -> str:
    text = str(value).strip().replace("/", "_")
    if not text:
        raise ValueError("HDF5 identifiers cannot be empty")
    return text


def _load_input(path: Path, marker_names: Sequence[str] | None) -> tuple[NDArray[np.float32], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        cells = np.load(path)
        markers = list(marker_names or [])
    elif suffix == ".npz":
        with np.load(path) as archive:
            cells = archive["cells"] if "cells" in archive else archive[archive.files[0]]
            stored = archive["markers"].tolist() if "markers" in archive else []
        markers = list(marker_names or stored)
    elif suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            first = next(reader)
            try:
                numeric_first = [float(value) for value in first]
                rows = [numeric_first, *([float(value) for value in row] for row in reader)]
                cells = np.asarray(rows)
                markers = list(marker_names or [f"marker_{i}" for i in range(cells.shape[1])])
            except ValueError:
                markers = list(marker_names or first)
                cells = np.asarray([[float(value) for value in row] for row in reader])
    elif suffix == ".fcs":
        try:
            from flowio import FlowData
        except ImportError as error:
            raise ImportError("FCS ingestion requires `pip install flowlot[fcs]`") from error
        flow = FlowData(str(path))
        cells = np.asarray(flow.events, dtype=np.float32).reshape(-1, flow.channel_count)
        channels = flow.channels
        def channel(index: int) -> dict[str, object]:
            return channels.get(str(index), channels.get(index, {}))
        inferred = [
            str(channel(i + 1).get("PnS") or channel(i + 1).get("PnN") or f"channel_{i + 1}")
            for i in range(flow.channel_count)
        ]
        markers = list(marker_names or inferred)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    cells = np.asarray(cells, dtype=np.float32)
    if cells.ndim != 2 or not cells.size:
        raise ValueError(f"{path} must contain a non-empty [cells, markers] matrix")
    if not markers:
        markers = [f"marker_{i}" for i in range(cells.shape[1])]
    if len(markers) != cells.shape[1]:
        raise ValueError(f"{path}: {len(markers)} marker names for {cells.shape[1]} columns")
    return cells, [str(marker) for marker in markers]


class Stage1Builder:
    """Incremental writer for the Stage 1 hierarchy.

    Existing sample/tube datasets are replaced atomically at the group level;
    unrelated patients and tubes remain untouched.
    """

    def __init__(self, output: str | Path, mode: str = "a") -> None:
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, mode)
        self.h5.attrs.update(schema="flowlot-stage1", schema_version="1.0")

    def __enter__(self) -> "Stage1Builder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.h5:
            self.h5.close()

    def add_sample(
        self,
        dataset_name: str,
        subsampled_cell_count: str | int,
        patient_id: str,
        label: str | int | float,
        tube_id: str,
        cells: NDArray[np.floating[Any]],
        marker_descriptions: Sequence[str],
        original_count: int | None = None,
        overwrite: bool = True,
    ) -> str:
        cells = np.asarray(cells, dtype=np.float32)
        if cells.ndim != 2 or not cells.size or len(marker_descriptions) != cells.shape[1]:
            raise ValueError("cells must be non-empty 2D and match marker_descriptions")
        patient_key = f"{_safe_key(patient_id)}_label_{_safe_key(label)}"
        path = "/".join(
            map(_safe_key, [dataset_name, subsampled_cell_count, patient_key, tube_id])
        )
        if path in self.h5:
            if not overwrite:
                raise ValueError(f"Stage 1 group already exists: /{path}")
            del self.h5[path]
        group = self.h5.require_group(path)
        group.attrs.update(
            patient_id=str(patient_id),
            label=str(label),
            tube_id=str(tube_id),
            counts=int(original_count if original_count is not None else len(cells)),
        )
        group.create_dataset("raw_cell_matrix", data=cells, compression="gzip", shuffle=True)
        group.create_dataset("marker_descriptions", data=np.asarray(marker_descriptions, dtype=UTF8))
        return f"/{path}"

    def add_file(
        self,
        path: str | Path,
        *,
        dataset_name: str,
        subsampled_cell_count: str | int,
        patient_id: str,
        label: str | int | float,
        tube_id: str,
        marker_names: Sequence[str] | None = None,
        seed: int = 0,
    ) -> str:
        cells, markers = _load_input(Path(path), marker_names)
        original_count = len(cells)
        if str(subsampled_cell_count).lower() != "all":
            count = int(subsampled_cell_count)
            if count < 1:
                raise ValueError("subsampled_cell_count must be positive or 'all'")
            if len(cells) > count:
                indices = np.random.default_rng(seed).choice(len(cells), count, replace=False)
                cells = cells[indices]
        return self.add_sample(
            dataset_name,
            subsampled_cell_count,
            patient_id,
            label,
            tube_id,
            cells,
            markers,
            original_count,
        )


def build_stage1_from_manifest(
    manifest: str | Path,
    output: str | Path,
    dataset_name: str,
    subsampled_cell_count: str | int = "all",
    seed: int = 0,
) -> Path:
    """Ingest a CSV with ``patient_id,tube_id,path,label[,markers]`` columns."""

    manifest = Path(manifest).resolve()
    with manifest.open(newline="", encoding="utf-8") as handle, Stage1Builder(output, "w") as builder:
        reader = csv.DictReader(handle)
        required = {"patient_id", "tube_id", "path", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            source = Path(row["path"])
            if not source.is_absolute():
                source = manifest.parent / source
            markers = [item.strip() for item in row.get("markers", "").split(";") if item.strip()]
            builder.add_file(
                source,
                dataset_name=dataset_name,
                subsampled_cell_count=subsampled_cell_count,
                patient_id=row["patient_id"],
                label=row["label"],
                tube_id=row["tube_id"],
                marker_names=markers or None,
                seed=seed + row_index,
            )
    return Path(output)


def import_legacy_flowcode_hdf5(
    legacy_path: str | Path,
    output: str | Path,
    dataset_name: str | None = None,
    sample_level: str | None = None,
) -> Path:
    """Migrate legacy ``/Dataset/.../patient_*/tube_*`` raw matrices to Stage 1."""

    def text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray) and value.shape == ():
            return text(value.item())
        return str(value)

    with h5py.File(legacy_path, "r") as source, Stage1Builder(output, "w") as target:
        if "Dataset" not in source:
            raise ValueError("Legacy file has no /Dataset group")
        datasets = [dataset_name] if dataset_name else list(source["Dataset"])
        for name in datasets:
            if name not in source["Dataset"]:
                raise KeyError(f"Legacy dataset {name!r} does not exist")
            levels = [sample_level] if sample_level else list(source[f"Dataset/{name}"])
            for legacy_level in levels:
                if legacy_level not in source[f"Dataset/{name}"]:
                    raise KeyError(f"Legacy sample level {legacy_level!r} does not exist")
                cell_level = legacy_level.removeprefix("sample_")
                for patient_key, patient in source[f"Dataset/{name}/{legacy_level}"].items():
                    if not isinstance(patient, h5py.Group):
                        continue
                    patient_id = patient_key.removeprefix("patient_")
                    label = text(patient["label"][()]) if "label" in patient else "unknown"
                    for tube_key, tube in patient.items():
                        if not isinstance(tube, h5py.Group) or "data" not in tube:
                            continue
                        marker_key = "markers" if "markers" in tube else "desc"
                        if marker_key not in tube:
                            raise KeyError(f"{tube.name} has no markers/desc dataset")
                        markers = [text(value) for value in tube[marker_key][...]]
                        original_count = len(tube["data"])
                        if "counts" in tube:
                            finite = np.asarray(tube["counts"][...], dtype=float)
                            finite = finite[np.isfinite(finite)]
                            if len(finite):
                                original_count = max(original_count, int(np.max(finite)))
                        target.add_sample(
                            name,
                            cell_level,
                            patient_id,
                            label,
                            tube_key.removeprefix("tube_"),
                            tube["data"][...],
                            markers,
                            original_count,
                        )
    return Path(output)
