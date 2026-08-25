"""Create explicit ingestion manifests from heterogeneous raw-data folders."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd


SUPPORTED_INPUTS = {".fcs", ".csv", ".tsv", ".txt", ".npy", ".npz"}


def create_manifest_from_folder(
    raw_root: str | Path,
    output: str | Path,
    filename_pattern: str,
    labels: str | Path | pd.DataFrame | None = None,
    markers_by_tube: Mapping[str, str | Sequence[str]] | None = None,
    recursive: bool = True,
    relative_paths: bool = True,
    strict: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Discover raw files and map them to patient/tube/label manifest rows.

    ``filename_pattern`` is searched against each path relative to ``raw_root``
    and must contain named groups ``patient_id`` and ``tube_id``. Labels should
    normally come from a CSV/DataFrame with ``patient_id,label`` columns; when no
    table is supplied the regex must also contain a named ``label`` group.
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
        missing = {"patient_id", "label"}.difference(label_table.columns)
        if missing:
            raise ValueError(f"Label table is missing columns: {sorted(missing)}")
        label_table = label_table[["patient_id", "label"]].astype(str)
        conflicting = label_table.groupby("patient_id")["label"].nunique()
        if (conflicting > 1).any():
            patients = conflicting[conflicting > 1].index.tolist()
            raise ValueError(f"Patients have conflicting labels: {patients[:10]}")
        label_table = label_table.drop_duplicates("patient_id")
        manifest = manifest.merge(label_table, on="patient_id", how="left", validate="many_to_one")
        if manifest["label"].isna().any():
            patients = manifest.loc[manifest["label"].isna(), "patient_id"].unique().tolist()
            raise ValueError(f"Label table has no label for patients: {patients[:10]}")
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
    duplicates = manifest.duplicated(["patient_id", "tube_id"], keep=False)
    if duplicates.any():
        rows = manifest.loc[duplicates, ["patient_id", "tube_id", "path"]]
        raise ValueError(f"Duplicate patient/tube files:\n{rows.to_string(index=False)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relative_paths:
        manifest["path"] = [
            os.path.relpath(path, start=destination.parent) for path in manifest["path"]
        ]
    manifest = manifest[["patient_id", "tube_id", "path", "label", "markers"]].sort_values(
        ["patient_id", "tube_id"]
    )
    manifest.to_csv(destination, index=False)
    return manifest.reset_index(drop=True)
