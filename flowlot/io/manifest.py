"""Create explicit ingestion manifests from heterogeneous raw-data folders."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd


SUPPORTED_INPUTS = {".fcs", ".csv", ".tsv", ".txt", ".npy", ".npz"}


def _metadata_text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def create_manifest_from_metadata(
    raw_root: str | Path,
    metadata: str | Path | pd.DataFrame,
    output: str | Path,
    *,
    file_column: str,
    patient_id_column: str,
    tube_id_column: str,
    label_column: str,
    tube_prefix: str = "",
    markers_by_tube: Mapping[str, str | Sequence[str]] | None = None,
    relative_paths: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create a manifest when patient/tube identity is supplied by a metadata table."""

    root = Path(raw_root).resolve()
    destination = Path(output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw-data folder does not exist: {root}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists: {destination}; set overwrite=True")
    table = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.read_csv(metadata)
    required = {file_column, patient_id_column, tube_id_column, label_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Metadata table is missing columns: {sorted(missing)}")
    available: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUTS:
            continue
        key = path.name.lower()
        if key in available:
            raise ValueError(f"Duplicate raw filename below {root}: {path.name}")
        available[key] = path.resolve()

    marker_mapping = markers_by_tube or {}
    rows: list[dict[str, str]] = []
    missing_files: list[str] = []
    for row in table.itertuples(index=False, name=None):
        record = dict(zip(table.columns, row, strict=True))
        filename = _metadata_text(record[file_column])
        candidates = [filename]
        if not Path(filename).suffix:
            candidates.extend(f"{filename}{suffix}" for suffix in (".fcs", ".FCS"))
        source = next((available[name.lower()] for name in candidates if name.lower() in available), None)
        if source is None:
            missing_files.append(filename)
            continue
        tube_value = _metadata_text(record[tube_id_column])
        tube = tube_value if not tube_prefix or tube_value.startswith(tube_prefix) else tube_prefix + tube_value
        markers = marker_mapping.get(tube, "")
        marker_text = markers if isinstance(markers, str) else ";".join(map(str, markers))
        rows.append(
            {
                "patient_id": _metadata_text(record[patient_id_column]),
                "tube_id": tube,
                "path": str(source),
                "label": _metadata_text(record[label_column]),
                "markers": marker_text,
            }
        )
    if missing_files:
        raise FileNotFoundError(
            f"Metadata references {len(missing_files)} missing raw files; examples={missing_files[:10]}"
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty or (manifest[["patient_id", "tube_id", "label"]] == "").any().any():
        raise ValueError("Metadata produced no rows or empty patient/tube/label values")
    duplicates = manifest.duplicated(["patient_id", "tube_id"], keep=False)
    if duplicates.any():
        raise ValueError("Metadata contains duplicate patient/tube rows")
    patient_labels = manifest.groupby("patient_id")["label"].nunique()
    if (patient_labels > 1).any():
        patients = patient_labels[patient_labels > 1].index.tolist()
        raise ValueError(f"Patients have conflicting labels across tubes: {patients[:10]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relative_paths:
        manifest["path"] = [
            os.path.relpath(path, start=destination.parent) for path in manifest["path"]
        ]
    manifest = manifest.sort_values(["patient_id", "tube_id"]).reset_index(drop=True)
    manifest.to_csv(destination, index=False)
    return manifest


def create_manifest_from_folder(
    raw_root: str | Path,
    output: str | Path,
    filename_pattern: str,
    labels: str | Path | pd.DataFrame | None = None,
    markers_by_tube: Mapping[str, str | Sequence[str]] | None = None,
    labels_id_column: str = "patient_id",
    labels_label_column: str = "label",
    labels_match: str = "patient_id",
    event_labels_root: str | Path | None = None,
    event_labels_pattern: str = "{stem}.csv",
    event_id_column: str = "event_ID",
    population_columns: Sequence[str] = (),
    recursive: bool = True,
    relative_paths: bool = True,
    strict: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Discover raw files and map them to patient/tube/label manifest rows.

    ``filename_pattern`` is searched against each path relative to ``raw_root``
    and must contain named groups ``patient_id`` and ``tube_id``. Labels should
    normally come from a CSV/DataFrame. ``labels_match='file_id'`` supports
    legacy ``sample_info.csv`` tables keyed by the raw filename stem. Optional
    event-level sidecars are resolved below ``event_labels_root`` and recorded
    for event-ID alignment during Stage 1 construction.
    """

    root = Path(raw_root).resolve()
    destination = Path(output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw-data folder does not exist: {root}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists: {destination}; set overwrite=True")
    expression = re.compile(filename_pattern)
    required_groups = {"patient_id", "tube_id"}
    missing_groups = required_groups.difference(expression.groupindex)
    if missing_groups:
        raise ValueError(f"filename_pattern needs named groups: {sorted(missing_groups)}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
    )
    if not files:
        raise ValueError(f"No supported raw files found below {root}")
    unmatched: list[str] = []
    parsed: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        match = expression.search(relative)
        if match is None:
            unmatched.append(relative)
            continue
        groups = match.groupdict()
        parsed.append(
            {
                "patient_id": str(groups["patient_id"]).strip(),
                "tube_id": str(groups["tube_id"]).strip(),
                "path": str(path),
                "file_id": path.stem,
                "regex_label": str(groups.get("label") or "").strip(),
            }
        )
    if strict and unmatched:
        preview = unmatched[:10]
        raise ValueError(
            f"{len(unmatched)} raw files did not match filename_pattern; examples={preview}"
        )
    if not parsed:
        raise ValueError("No raw files matched filename_pattern")
    manifest = pd.DataFrame(parsed)
    if (manifest[["patient_id", "tube_id"]] == "").any().any():
        raise ValueError("The filename pattern produced an empty patient_id or tube_id")
    if labels is not None:
        label_table = (
            labels.copy() if isinstance(labels, pd.DataFrame) else pd.read_csv(labels, dtype=str)
        )
        missing = {labels_id_column, labels_label_column}.difference(label_table.columns)
        if missing:
            raise ValueError(f"Label table is missing columns: {sorted(missing)}")
        if labels_match not in {"patient_id", "file_id"}:
            raise ValueError("labels_match must be 'patient_id' or 'file_id'")
        label_table = label_table[[labels_id_column, labels_label_column]].astype(str).rename(
            columns={labels_id_column: "label_key", labels_label_column: "label"}
        )
        conflicting = label_table.groupby("label_key")["label"].nunique()
        if (conflicting > 1).any():
            keys = conflicting[conflicting > 1].index.tolist()
            raise ValueError(f"Label keys have conflicting labels: {keys[:10]}")
        label_table = label_table.drop_duplicates("label_key")
        manifest = manifest.merge(
            label_table,
            left_on=labels_match,
            right_on="label_key",
            how="left",
            validate="many_to_one",
        ).drop(columns="label_key")
        if manifest["label"].isna().any():
            keys = manifest.loc[manifest["label"].isna(), labels_match].unique().tolist()
            raise ValueError(f"Label table has no label for keys: {keys[:10]}")
    else:
        if "label" not in expression.groupindex:
            raise ValueError("Supply a label table or add a named (?P<label>...) regex group")
        manifest = manifest.rename(columns={"regex_label": "label"})
    if "regex_label" in manifest:
        manifest = manifest.drop(columns="regex_label")
    marker_mapping = markers_by_tube or {}

    def marker_text(tube: str) -> str:
        value = marker_mapping.get(tube, "")
        if isinstance(value, str):
            return value
        return ";".join(map(str, value))

    manifest["markers"] = [marker_text(tube) for tube in manifest["tube_id"]]
    if event_labels_root is not None:
        if not population_columns:
            raise ValueError("population_columns are required with event_labels_root")
        labels_root = Path(event_labels_root).resolve()
        if not labels_root.is_dir():
            raise NotADirectoryError(f"Event-label folder does not exist: {labels_root}")
        event_paths: list[str] = []
        missing_event_labels: list[str] = []
        for row in manifest.itertuples(index=False):
            event_path = labels_root / event_labels_pattern.format(
                stem=row.file_id,
                patient_id=row.patient_id,
                tube_id=row.tube_id,
            )
            if not event_path.is_file():
                missing_event_labels.append(str(event_path))
            event_paths.append(str(event_path.resolve()))
        if strict and missing_event_labels:
            raise FileNotFoundError(
                f"Missing {len(missing_event_labels)} event-label files; "
                f"examples={missing_event_labels[:10]}"
            )
        manifest["event_labels_path"] = event_paths
        manifest["event_id_column"] = event_id_column
        manifest["population_columns"] = ";".join(population_columns)
    duplicates = manifest.duplicated(["patient_id", "tube_id"], keep=False)
    if duplicates.any():
        rows = manifest.loc[duplicates, ["patient_id", "tube_id", "path"]]
        raise ValueError(f"Duplicate patient/tube files:\n{rows.to_string(index=False)}")
    patient_labels = manifest.groupby("patient_id")["label"].nunique()
    if (patient_labels > 1).any():
        patients = patient_labels[patient_labels > 1].index.tolist()
        raise ValueError(f"Patients have conflicting labels across tubes: {patients[:10]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relative_paths:
        manifest["path"] = [
            os.path.relpath(path, start=destination.parent) for path in manifest["path"]
        ]
        if "event_labels_path" in manifest:
            manifest["event_labels_path"] = [
                os.path.relpath(path, start=destination.parent)
                for path in manifest["event_labels_path"]
            ]
    columns = ["patient_id", "tube_id", "path", "label", "markers"]
    columns.extend(
        column
        for column in ("event_labels_path", "event_id_column", "population_columns")
        if column in manifest
    )
    manifest = manifest[columns].sort_values(["patient_id", "tube_id"])
    manifest.to_csv(destination, index=False)
    return manifest.reset_index(drop=True)
