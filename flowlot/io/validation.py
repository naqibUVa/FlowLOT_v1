"""Memory-conscious manifest and HDF5 diagnostics for interactive workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ISSUE_COLUMNS = ["severity", "location", "check", "message"]
SUPPORTED_INPUTS = {".fcs", ".csv", ".tsv", ".txt", ".npy", ".npz"}


def _issues_frame(issues: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def _add_issue(
    issues: list[dict[str, str]], severity: str, location: str, check: str, message: str
) -> None:
    issues.append(
        {"severity": severity, "location": location, "check": check, "message": message}
    )


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _matrix_stats(dataset: h5py.Dataset, chunk_rows: int = 65_536) -> dict[str, Any]:
    if dataset.ndim != 2:
        return {
            "n_cells": dataset.shape[0] if dataset.shape else 0,
            "n_markers": dataset.shape[1] if dataset.ndim > 1 else 0,
            "finite_fraction": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }
    finite_count = 0
    total = int(np.prod(dataset.shape))
    minimum, maximum = float("inf"), float("-inf")
    for start in range(0, dataset.shape[0], chunk_rows):
        values = np.asarray(dataset[start : start + chunk_rows])
        finite = np.isfinite(values)
        finite_count += int(finite.sum())
        if finite.any():
            minimum = min(minimum, float(values[finite].min()))
            maximum = max(maximum, float(values[finite].max()))
    return {
        "n_cells": int(dataset.shape[0]),
        "n_markers": int(dataset.shape[1]),
        "finite_fraction": finite_count / total if total else float("nan"),
        "minimum": minimum if finite_count else float("nan"),
        "maximum": maximum if finite_count else float("nan"),
    }


def audit_manifest(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve and validate a raw-ingestion CSV before writing Stage 1."""

    manifest = Path(path).resolve()
    if not manifest.exists():
        issue = [{
            "severity": "error",
            "location": str(manifest),
            "check": "manifest_exists",
            "message": "Manifest file does not exist",
        }]
        return pd.DataFrame(), _issues_frame(issue)
    inventory = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"patient_id", "tube_id", "path", "label"}
    missing = required.difference(inventory.columns)
    issues: list[dict[str, str]] = []
    if missing:
        _add_issue(
            issues,
            "error",
            str(manifest),
            "required_columns",
            f"Missing columns: {sorted(missing)}",
        )
        return inventory, _issues_frame(issues)
    resolved = []
    for index, row in inventory.iterrows():
        source = Path(row["path"])
        if not source.is_absolute():
            source = manifest.parent / source
        source = source.resolve()
        resolved.append(str(source))
        location = f"row {index + 2}: {source}"
        for column in ("patient_id", "tube_id", "path", "label"):
            if not str(row[column]).strip():
                _add_issue(issues, "error", location, "nonempty_fields", f"{column} is empty")
        if not source.exists():
            _add_issue(issues, "error", location, "source_exists", "Input file is missing")
        if source.suffix.lower() not in SUPPORTED_INPUTS:
            _add_issue(
                issues,
                "error",
                location,
                "source_format",
                f"Unsupported extension {source.suffix!r}",
            )
    inventory = inventory.copy()
    inventory["resolved_path"] = resolved
    duplicates = inventory.duplicated(["patient_id", "tube_id"], keep=False)
    for index in inventory.index[duplicates]:
        _add_issue(
            issues,
            "error",
            f"row {index + 2}",
            "unique_patient_tube",
            "Duplicate patient_id/tube_id key",
        )
    label_counts = inventory.groupby("patient_id")["label"].nunique()
    for patient, count in label_counts[label_counts > 1].items():
        _add_issue(
            issues,
            "error",
            str(patient),
            "consistent_patient_label",
            f"Patient has {count} labels across tubes",
        )
    return inventory, _issues_frame(issues)


def audit_stage1(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one inventory row per Stage 1 patient/tube and invariant failures."""

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "flowlot-stage1":
            _add_issue(issues, "error", "/", "root_schema", "Expected flowlot-stage1")
        patient_labels: dict[tuple[str, str, str], str] = {}
        for dataset_name, dataset_group in handle.items():
            if not isinstance(dataset_group, h5py.Group):
                continue
            for cell_count, count_group in dataset_group.items():
                for patient_key, patient_group in count_group.items():
                    for tube_id, tube in patient_group.items():
                        location = tube.name
                        required = {"raw_cell_matrix", "marker_descriptions"}
                        missing = required.difference(tube.keys())
                        if missing:
                            _add_issue(
                                issues,
                                "error",
                                location,
                                "required_datasets",
                                f"Missing {sorted(missing)}",
                            )
                            continue
                        matrix = tube["raw_cell_matrix"]
                        markers = _decode(tube["marker_descriptions"][...])
                        statistics = _matrix_stats(matrix)
                        patient_id = str(tube.attrs.get("patient_id", patient_key))
                        label = str(tube.attrs.get("label", ""))
                        original_count = int(tube.attrs.get("counts", statistics["n_cells"]))
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "cell_count": cell_count,
                                "patient_id": patient_id,
                                "label": label,
                                "tube": tube_id,
                                "original_count": original_count,
                                **statistics,
                            }
                        )
                        key = (dataset_name, cell_count, patient_id)
                        if key in patient_labels and patient_labels[key] != label:
                            _add_issue(
                                issues,
                                "error",
                                location,
                                "consistent_patient_label",
                                "Label differs across tubes",
                            )
                        patient_labels[key] = label
                        if matrix.ndim != 2 or statistics["n_markers"] != len(markers):
                            _add_issue(
                                issues,
                                "error",
                                location,
                                "matrix_marker_shape",
                                f"Matrix {matrix.shape} versus {len(markers)} markers",
                            )
                        if len(markers) != len(set(markers)):
                            _add_issue(
                                issues, "error", location, "unique_markers", "Duplicate markers"
                            )
                        if original_count < statistics["n_cells"]:
                            _add_issue(
                                issues,
                                "error",
                                location,
                                "original_count",
                                "Original count is smaller than stored cell count",
                            )
                        if statistics["finite_fraction"] < 1:
                            _add_issue(
                                issues,
                                "error",
                                location,
                                "finite_values",
                                f"Finite fraction is {statistics['finite_fraction']:.6f}",
                            )
    return pd.DataFrame(rows), _issues_frame(issues)


def audit_stage2(path: str | Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Inventory Stage 2 raw, preprocess, and LOT groups and verify alignment."""

    raw_rows: list[dict[str, Any]] = []
    preprocess_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "flowlot-stage2":
            _add_issue(issues, "error", "/", "root_schema", "Expected flowlot-stage2")
        patient_labels: dict[tuple[str, str, str], object] = {}
        for dataset_name, dataset_group in handle.items():
            if not isinstance(dataset_group, h5py.Group):
                continue
            for cell_count, count_group in dataset_group.items():
                for tube_id, tube in count_group.items():
                    location = tube.name
                    if "metadata" not in tube or "raw" not in tube:
                        _add_issue(
                            issues,
                            "error",
                            location,
                            "required_groups",
                            "Tube requires metadata and raw groups",
                        )
                        continue
                    metadata = tube["metadata"]
                    required_metadata = {
                        "patient_ids", "labels", "cell_counts", "marker_descriptions"
                    }
                    missing = required_metadata.difference(metadata.keys())
                    if missing:
                        _add_issue(
                            issues,
                            "error",
                            metadata.name,
                            "metadata_datasets",
                            f"Missing {sorted(missing)}",
                        )
                        continue
                    patient_ids = _decode(metadata["patient_ids"][...])
                    labels = metadata["labels"][...]
                    labels = _decode(labels) if labels.dtype.kind in {"S", "O", "U"} else labels.tolist()
                    counts = metadata["cell_counts"][...]
                    markers = _decode(metadata["marker_descriptions"][...])
                    if not (len(patient_ids) == len(labels) == len(counts)):
                        _add_issue(
                            issues,
                            "error",
                            metadata.name,
                            "metadata_lengths",
                            "patient_ids, labels, and cell_counts differ in length",
                        )
                    if len(patient_ids) != len(set(patient_ids)):
                        _add_issue(
                            issues, "error", metadata.name, "unique_patients", "Duplicate IDs"
                        )
                    if len(markers) != len(set(markers)):
                        _add_issue(
                            issues,
                            "error",
                            metadata.name,
                            "unique_markers",
                            "Duplicate marker descriptions",
                        )
                    for patient_id, label in zip(patient_ids, labels):
                        key = (dataset_name, cell_count, patient_id)
                        if key in patient_labels and patient_labels[key] != label:
                            _add_issue(
                                issues,
                                "error",
                                metadata.name,
                                "consistent_patient_label",
                                f"Patient {patient_id} has inconsistent labels across tubes",
                            )
                        patient_labels[key] = label
                    raw_ids = set(tube["raw"].keys())
                    if raw_ids != set(patient_ids):
                        _add_issue(
                            issues,
                            "error",
                            tube["raw"].name,
                            "raw_metadata_alignment",
                            "Raw patient IDs differ from metadata",
                        )
                    label_map = dict(zip(patient_ids, labels))
                    count_map = dict(zip(patient_ids, counts))
                    for patient_id, patient in tube["raw"].items():
                        matrix = patient.get("raw_cell_matrix")
                        if matrix is None:
                            _add_issue(
                                issues,
                                "error",
                                patient.name,
                                "raw_matrix",
                                "Missing raw_cell_matrix",
                            )
                            continue
                        statistics = _matrix_stats(matrix)
                        raw_rows.append(
                            {
                                "dataset": dataset_name,
                                "cell_count": cell_count,
                                "tube": tube_id,
                                "patient_id": patient_id,
                                "label": label_map.get(patient_id),
                                "original_count": count_map.get(patient_id),
                                **statistics,
                            }
                        )
                        if statistics["n_markers"] != len(markers):
                            _add_issue(
                                issues,
                                "error",
                                matrix.name,
                                "raw_marker_shape",
                                f"Matrix has {statistics['n_markers']} columns; expected {len(markers)}",
                            )
                        if statistics["finite_fraction"] < 1:
                            _add_issue(
                                issues,
                                "error",
                                matrix.name,
                                "finite_values",
                                f"Finite fraction is {statistics['finite_fraction']:.6f}",
                            )
                        if patient_id in count_map and count_map[patient_id] < statistics["n_cells"]:
                            _add_issue(
                                issues,
                                "error",
                                matrix.name,
                                "original_count",
                                "Metadata count is smaller than stored cell count",
                            )
                    for group_name, group in tube.items():
                        if group_name.startswith("preprocess_"):
                            preprocess_id = group_name.removeprefix("preprocess_")
                            if "marker_subset" not in group:
                                _add_issue(
                                    issues,
                                    "error",
                                    group.name,
                                    "marker_subset",
                                    "Missing marker_subset",
                                )
                                continue
                            selected = _decode(group["marker_subset"][...])
                            if len(selected) != len(set(selected)):
                                _add_issue(
                                    issues,
                                    "error",
                                    group.name,
                                    "unique_markers",
                                    "Duplicate markers in marker_subset",
                                )
                            processed_ids = {key for key in group if key != "marker_subset"}
                            if processed_ids != raw_ids:
                                _add_issue(
                                    issues,
                                    "error",
                                    group.name,
                                    "preprocess_coverage",
                                    "Processed patient IDs differ from raw IDs",
                                )
                            for patient_id in sorted(processed_ids):
                                matrix = group[patient_id].get("processed_matrix")
                                if matrix is None:
                                    _add_issue(
                                        issues,
                                        "error",
                                        group[patient_id].name,
                                        "processed_matrix",
                                        "Missing processed_matrix",
                                    )
                                    continue
                                statistics = _matrix_stats(matrix)
                                preprocess_rows.append(
                                    {
                                        "dataset": dataset_name,
                                        "cell_count": cell_count,
                                        "tube": tube_id,
                                        "preprocess": preprocess_id,
                                        "patient_id": patient_id,
                                        **statistics,
                                    }
                                )
                                if statistics["n_markers"] != len(selected):
                                    _add_issue(
                                        issues,
                                        "error",
                                        matrix.name,
                                        "processed_marker_shape",
                                        f"Matrix has {statistics['n_markers']} columns; expected {len(selected)}",
                                    )
                                if statistics["finite_fraction"] < 1:
                                    _add_issue(
                                        issues,
                                        "error",
                                        matrix.name,
                                        "finite_values",
                                        f"Finite fraction is {statistics['finite_fraction']:.6f}",
                                    )
                        if group_name == "lot_embeddings":
                            for preprocess_id, preprocess_group in group.items():
                                for embedding_id, embedding in preprocess_group.items():
                                    required = {
                                        "patient_ids",
                                        "reference_patient_ids",
                                        "reference_matrix",
                                        "embeddings",
                                        "transport_costs",
                                        "converged",
                                        "sorted_cell_matrices",
                                    }
                                    missing_embedding = required.difference(embedding.keys())
                                    if missing_embedding:
                                        _add_issue(
                                            issues,
                                            "error",
                                            embedding.name,
                                            "embedding_datasets",
                                            f"Missing {sorted(missing_embedding)}",
                                        )
                                        continue
                                    embedding_ids = _decode(embedding["patient_ids"][...])
                                    values = embedding["embeddings"]
                                    reference = embedding["reference_matrix"]
                                    costs = embedding["transport_costs"]
                                    converged = embedding["converged"]
                                    reference_cells = reference.shape[0] if reference.ndim else 0
                                    reference_markers = (
                                        reference.shape[1] if reference.ndim == 2 else 0
                                    )
                                    expected_width = (
                                        int(np.prod(reference.shape))
                                        if reference.ndim == 2
                                        else -1
                                    )
                                    value_statistics = _matrix_stats(values)
                                    embedding_rows.append(
                                        {
                                            "dataset": dataset_name,
                                            "cell_count": cell_count,
                                            "tube": tube_id,
                                            "preprocess": preprocess_id,
                                            "embedding": embedding_id,
                                            "n_patients": len(embedding_ids),
                                            "embedding_width": values.shape[1] if values.ndim == 2 else 0,
                                            "reference_cells": reference_cells,
                                            "reference_markers": reference_markers,
                                            "finite_fraction": value_statistics["finite_fraction"],
                                            "mean_transport_cost": (
                                                float(np.mean(costs)) if len(costs) else float("nan")
                                            ),
                                            "median_transport_cost": (
                                                float(np.median(costs))
                                                if len(costs)
                                                else float("nan")
                                            ),
                                            "converged_fraction": (
                                                float(np.mean(converged))
                                                if len(converged)
                                                else float("nan")
                                            ),
                                            "transport_matrices_stored": "transport_matrices" in embedding,
                                        }
                                    )
                                    if values.ndim != 2 or values.shape[0] != len(embedding_ids):
                                        _add_issue(
                                            issues,
                                            "error",
                                            values.name,
                                            "embedding_rows",
                                            "Embedding rows differ from patient_ids",
                                        )
                                    if values.ndim != 2 or values.shape[1] != expected_width:
                                        _add_issue(
                                            issues,
                                            "error",
                                            values.name,
                                            "embedding_width",
                                            f"Expected flattened width {expected_width}",
                                        )
                                    if reference.ndim != 2:
                                        _add_issue(
                                            issues,
                                            "error",
                                            reference.name,
                                            "reference_shape",
                                            f"Expected 2D reference; found {reference.shape}",
                                        )
                                    if len(costs) != len(embedding_ids) or len(converged) != len(
                                        embedding_ids
                                    ):
                                        _add_issue(
                                            issues,
                                            "error",
                                            embedding.name,
                                            "embedding_metadata_lengths",
                                            "Costs/convergence lengths differ from patient_ids",
                                        )
                                    sorted_ids = set(embedding["sorted_cell_matrices"].keys())
                                    if sorted_ids != set(embedding_ids):
                                        _add_issue(
                                            issues,
                                            "error",
                                            embedding["sorted_cell_matrices"].name,
                                            "sorted_matrix_coverage",
                                            "Sorted matrix patient IDs differ from embedding IDs",
                                        )
                                    if value_statistics["finite_fraction"] < 1:
                                        _add_issue(
                                            issues,
                                            "error",
                                            values.name,
                                            "finite_values",
                                            f"Finite fraction is {value_statistics['finite_fraction']:.6f}",
                                        )
                                    if not np.isfinite(costs[...]).all():
                                        _add_issue(
                                            issues,
                                            "error",
                                            costs.name,
                                            "finite_transport_costs",
                                            "Transport costs contain non-finite values",
                                        )
                                    if not np.isfinite(reference[...]).all():
                                        _add_issue(
                                            issues,
                                            "error",
                                            reference.name,
                                            "finite_reference",
                                            "Reference matrix contains non-finite values",
                                        )
    inventories = {
        "raw": pd.DataFrame(raw_rows),
        "preprocess": pd.DataFrame(preprocess_rows),
        "embeddings": pd.DataFrame(embedding_rows),
    }
    return inventories, _issues_frame(issues)
