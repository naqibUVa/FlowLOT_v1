"""Build the raw, patient-centric Stage 1 HDF5 file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray
import pandas as pd


UTF8 = h5py.string_dtype("utf-8")


def _safe_key(value: object) -> str:
    text = str(value).strip().replace("/", "_")
    if not text:
        raise ValueError("HDF5 identifiers cannot be empty")
    return text


def _normalized_name(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _event_id(value: object) -> str:
    """Normalize CSV/FCS representations such as ``42``, ``42.0``, and whitespace."""

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    rounded = round(number)
    return str(int(rounded)) if np.isfinite(number) and abs(number - rounded) < 1e-6 else text


def _event_annotation_alignment(
    cells: NDArray[np.float32],
    markers: Sequence[str],
    labels_path: str | Path,
    event_id_column: str,
    population_columns: Sequence[str],
) -> tuple[NDArray[np.int64], list[str], pd.DataFrame, NDArray[np.float64]]:
    """Match per-event CSV labels to raw matrix rows using a stable event identifier."""

    marker_matches = [
        index
        for index, marker in enumerate(markers)
        if _normalized_name(marker) == _normalized_name(event_id_column)
    ]
    if len(marker_matches) != 1:
        raise ValueError(
            f"Expected one {event_id_column!r} channel, found {len(marker_matches)} in {markers}"
        )
    raw_event_ids = [_event_id(value) for value in cells[:, marker_matches[0]]]
    if len(raw_event_ids) != len(set(raw_event_ids)):
        raise ValueError("Raw event IDs are not unique; event-level labels cannot be aligned safely")

    annotations = pd.read_csv(labels_path)
    requested = [event_id_column, *population_columns]
    resolved: dict[str, str] = {}
    for name in requested:
        matches = [
            column
            for column in annotations.columns
            if _normalized_name(column) == _normalized_name(name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {name!r} column in {labels_path}, found {len(matches)}"
            )
        resolved[name] = matches[0]
    annotations = annotations[[resolved[name] for name in requested]].rename(
        columns={resolved[name]: name for name in requested}
    )
    annotations[event_id_column] = annotations[event_id_column].map(_event_id)
    if annotations[event_id_column].duplicated().any():
        raise ValueError(f"Duplicate {event_id_column!r} values in {labels_path}")
    for population in population_columns:
        annotations[population] = pd.to_numeric(annotations[population], errors="raise")
        if not np.isfinite(annotations[population].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in population column {population!r}")
    original_counts = annotations[list(population_columns)].sum().to_numpy(dtype=np.float64)
    annotations = annotations.set_index(event_id_column)
    annotated_ids = set(annotations.index)
    eligible = np.asarray(
        [index for index, identifier in enumerate(raw_event_ids) if identifier in annotated_ids],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError(f"No event IDs overlap between the raw matrix and {labels_path}")
    return eligible, raw_event_ids, annotations, original_counts


def load_cytometry_file(
    path: str | Path, marker_names: Sequence[str] | None = None
) -> tuple[NDArray[np.float32], list[str]]:
    """Read FCS/CSV/TSV/TXT/NPY/NPZ into a cells-by-markers float32 matrix."""

    path = Path(path)
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
            from flowio import FlowData, read_multiple_data_sets
            from flowio.exceptions import MultipleDataSetsError
        except ImportError as error:
            raise ImportError("FCS ingestion requires `pip install flowlot[fcs]`") from error
        try:
            flow = FlowData(str(path), ignore_offset_error=True)
        except MultipleDataSetsError:
            data_sets = read_multiple_data_sets(str(path), ignore_offset_error=True)
            if not data_sets:
                raise ValueError(f"{path} contains no readable FCS data sets")
            flow = data_sets[0]
        cells = np.asarray(flow.events, dtype=np.float32).reshape(-1, flow.channel_count)
        channels = flow.channels

        def channel(index: int) -> dict[str, object]:
            return channels.get(str(index), channels.get(index, {}))

        inferred = []
        for index in range(flow.channel_count):
            metadata = channel(index + 1)
            pnn = str(metadata.get("PnN") or "")
            pns = str(metadata.get("PnS") or "")
            # Preserve the join key even when PnS contains a different description.
            if _normalized_name(pnn) in {"eventid", "eventidentifier"}:
                inferred.append(pnn)
            else:
                inferred.append(pns or pnn or f"channel_{index + 1}")
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
        sample_event_ids: Sequence[str] | None = None,
        sample_source_indices: NDArray[np.integer[Any]] | None = None,
        population_names: Sequence[str] | None = None,
        population_annotations: NDArray[np.floating[Any]] | None = None,
        original_population_counts: NDArray[np.floating[Any]] | None = None,
        annotated_event_count: int | None = None,
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
        if population_names:
            names = list(population_names)
            annotations = np.asarray(population_annotations, dtype=np.float32)
            original = np.asarray(original_population_counts, dtype=np.float64)
            if annotations.shape != (len(cells), len(names)) or original.shape != (len(names),):
                raise ValueError("Population annotation/count shapes do not match cells and names")
            if sample_event_ids is None or len(sample_event_ids) != len(cells):
                raise ValueError("sample_event_ids must align with annotated cells")
            if sample_source_indices is None or len(sample_source_indices) != len(cells):
                raise ValueError("sample_source_indices must align with annotated cells")
            sampled = annotations.sum(axis=0, dtype=np.float64)
            denominator = next(
                (index for index, name in enumerate(names) if _normalized_name(name) == "wbc"),
                None,
            )
            if denominator is None or sampled[denominator] == 0:
                sampled_percent = np.full(len(names), np.nan)
            else:
                sampled_percent = 100.0 * sampled / sampled[denominator]
            if denominator is None or original[denominator] == 0:
                original_percent = np.full(len(names), np.nan)
            else:
                original_percent = 100.0 * original / original[denominator]
            population_counts = np.column_stack(
                (sampled, original, sampled_percent, original_percent)
            )
            group.attrs["annotated_event_count"] = int(annotated_event_count or len(cells))
            group.create_dataset("sample_event_ids", data=np.asarray(sample_event_ids, dtype=UTF8))
            group.create_dataset(
                "sample_source_indices", data=np.asarray(sample_source_indices, dtype=np.int64)
            )
            annotation_dataset = group.create_dataset(
                "population_annotations", data=annotations, compression="gzip", shuffle=True
            )
            annotation_dataset.attrs["population_names"] = np.asarray(names, dtype="S")
            counts_dataset = group.create_dataset("population_counts", data=population_counts)
            counts_dataset.attrs["population_names"] = np.asarray(names, dtype="S")
            counts_dataset.attrs["metric_names"] = np.asarray(
                ["sampled_count", "original_count", "sampled_pct_wbc", "original_pct_wbc"],
                dtype="S",
            )
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
        event_labels_path: str | Path | None = None,
        event_id_column: str = "event_ID",
        population_columns: Sequence[str] = (),
    ) -> str:
        cells, markers = load_cytometry_file(path, marker_names)
        original_count = len(cells)
        eligible = np.arange(original_count, dtype=np.int64)
        raw_event_ids: list[str] | None = None
        annotations: pd.DataFrame | None = None
        original_population_counts: NDArray[np.float64] | None = None
        if event_labels_path is not None:
            if not population_columns:
                raise ValueError("population_columns are required with event_labels_path")
            eligible, raw_event_ids, annotations, original_population_counts = (
                _event_annotation_alignment(
                    cells,
                    markers,
                    event_labels_path,
                    event_id_column,
                    population_columns,
                )
            )
        if str(subsampled_cell_count).lower() != "all":
            count = int(subsampled_cell_count)
            if count < 1:
                raise ValueError("subsampled_cell_count must be positive or 'all'")
            if len(eligible) > count:
                # A seed-specific full permutation makes separately generated
                # cell-count levels nested (e.g. 500 ⊂ 1000 ⊂ 2000).
                selected = np.random.default_rng(seed).permutation(len(eligible))[:count]
                eligible = eligible[selected]
        cells = cells[eligible]
        sample_event_ids = None
        population_annotations = None
        if annotations is not None and raw_event_ids is not None:
            sample_event_ids = [raw_event_ids[index] for index in eligible]
            population_annotations = annotations.loc[
                sample_event_ids, list(population_columns)
            ].to_numpy(dtype=np.float32)
        return self.add_sample(
            dataset_name,
            subsampled_cell_count,
            patient_id,
            label,
            tube_id,
            cells,
            markers,
            original_count,
            sample_event_ids=sample_event_ids,
            sample_source_indices=eligible if annotations is not None else None,
            population_names=population_columns,
            population_annotations=population_annotations,
            original_population_counts=original_population_counts,
            annotated_event_count=len(annotations) if annotations is not None else None,
        )


def build_stage1_from_manifest(
    manifest: str | Path,
    output: str | Path,
    dataset_name: str,
    subsampled_cell_count: str | int = "all",
    seed: int = 0,
    mode: str = "w",
) -> Path:
    """Ingest a CSV with ``patient_id,tube_id,path,label[,markers]`` columns."""

    if mode not in {"w", "a"}:
        raise ValueError("mode must be 'w' or 'a'")
    manifest = Path(manifest).resolve()
    with manifest.open(newline="", encoding="utf-8") as handle, Stage1Builder(output, mode) as builder:
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
            event_labels_path = row.get("event_labels_path", "").strip()
            if event_labels_path:
                event_labels = Path(event_labels_path)
                if not event_labels.is_absolute():
                    event_labels = manifest.parent / event_labels
            else:
                event_labels = None
            populations = [
                item.strip()
                for item in row.get("population_columns", "").split(";")
                if item.strip()
            ]
            builder.add_file(
                source,
                dataset_name=dataset_name,
                subsampled_cell_count=subsampled_cell_count,
                patient_id=row["patient_id"],
                label=row["label"],
                tube_id=row["tube_id"],
                marker_names=markers or None,
                seed=seed + row_index,
                event_labels_path=event_labels,
                event_id_column=row.get("event_id_column", "").strip() or "event_ID",
                population_columns=populations,
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
