"""Convert patient-centric Stage 1 files to tube-centric Stage 2 files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


UTF8 = h5py.string_dtype("utf-8")


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _parse_label(value: str) -> int | float | str:
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


class Stage2Organizer:
    def __init__(self, stage1_path: str | Path, stage2_path: str | Path) -> None:
        self.stage1_path = Path(stage1_path)
        self.stage2_path = Path(stage2_path)

    def organize(
        self,
        dataset_name: str,
        subsampled_cell_count: str | int,
        marker_policy: str = "intersection",
        overwrite: bool = True,
    ) -> Path:
        if marker_policy not in {"intersection", "strict"}:
            raise ValueError("marker_policy must be 'intersection' or 'strict'")
        base = f"{dataset_name}/{subsampled_cell_count}"
        tube_records: dict[str, list[dict[str, object]]] = {}
        with h5py.File(self.stage1_path, "r") as source:
            if base not in source:
                raise KeyError(f"Stage 1 group /{base} does not exist")
            for patient_group in source[base].values():
                for tube_id, tube in patient_group.items():
                    markers = _decode(tube["marker_descriptions"][...])
                    tube_records.setdefault(tube_id, []).append(
                        {
                            "patient_id": str(tube.attrs["patient_id"]),
                            "label": str(tube.attrs["label"]),
                            "counts": int(tube.attrs["counts"]),
                            "markers": markers,
                            "source_path": tube.name,
                        }
                    )
        self.stage2_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.stage1_path, "r") as source, h5py.File(
            self.stage2_path, "a"
        ) as target:
            target.attrs.update(schema="flowlot-stage2", schema_version="1.0")
            if base in target and overwrite:
                del target[base]
            elif base in target:
                raise ValueError(f"Stage 2 group /{base} already exists")
            for tube_id, records in tube_records.items():
                tube = target.require_group(f"{base}/{tube_id}")
                first_markers = records[0]["markers"]
                if marker_policy == "strict":
                    if any(record["markers"] != first_markers for record in records):
                        raise ValueError(f"Marker mismatch in tube {tube_id}")
                    common = list(first_markers)
                else:
                    common = [
                        marker
                        for marker in first_markers
                        if all(marker in record["markers"] for record in records)
                    ]
                    if not common:
                        raise ValueError(f"Tube {tube_id} has no common markers")
                metadata = tube.create_group("metadata")
                metadata.create_dataset(
                    "patient_ids", data=np.asarray([r["patient_id"] for r in records], dtype=UTF8)
                )
                labels = [_parse_label(str(record["label"])) for record in records]
                if all(isinstance(value, (int, float)) for value in labels):
                    metadata.create_dataset("labels", data=np.asarray(labels))
                else:
                    metadata.create_dataset("labels", data=np.asarray(labels, dtype=UTF8))
                metadata.create_dataset("cell_counts", data=[record["counts"] for record in records])
                metadata.create_dataset("marker_descriptions", data=np.asarray(common, dtype=UTF8))
                raw = tube.create_group("raw")
                for record in records:
                    indices = [record["markers"].index(marker) for marker in common]
                    patient = raw.create_group(str(record["patient_id"]))
                    cells = source[f"{record['source_path']}/raw_cell_matrix"]
                    patient.create_dataset(
                        "raw_cell_matrix",
                        data=cells[...][:, indices].astype(np.float32),
                        compression="gzip",
                        shuffle=True,
                    )
                    source_tube = source[str(record["source_path"])]
                    for dataset_name in (
                        "sample_event_ids",
                        "sample_source_indices",
                        "population_annotations",
                        "population_counts",
                    ):
                        if dataset_name not in source_tube:
                            continue
                        source_tube.copy(dataset_name, patient, name=dataset_name)
                    if "annotated_event_count" in source_tube.attrs:
                        patient.attrs["annotated_event_count"] = source_tube.attrs[
                            "annotated_event_count"
                        ]
        return self.stage2_path

    def add_preprocess(
        self,
        dataset_name: str,
        subsampled_cell_count: str | int,
        preprocess_id: str,
        marker_subset: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        arcsinh_cofactor: float | None = None,
        skip_tubes: Sequence[str] = (),
        overwrite: bool = True,
    ) -> None:
        """Create marker-aligned processed matrices for every tube and patient."""

        base = f"{dataset_name}/{subsampled_cell_count}"
        with h5py.File(self.stage2_path, "a") as h5:
            if base not in h5:
                raise KeyError(f"Stage 2 group /{base} does not exist")
            for tube_id, tube in h5[base].items():
                if tube_id in set(skip_tubes):
                    continue
                group_name = f"preprocess_{preprocess_id}"
                if group_name in tube:
                    if not overwrite:
                        raise ValueError(f"/{base}/{tube_id}/{group_name} exists")
                    del tube[group_name]
                available = _decode(tube["metadata/marker_descriptions"][...])
                requested = marker_subset.get(tube_id) if isinstance(marker_subset, Mapping) else marker_subset
                selected = list(requested or available)
                missing = set(selected).difference(available)
                if missing:
                    raise ValueError(f"Tube {tube_id} lacks markers: {sorted(missing)}")
                indices = [available.index(marker) for marker in selected]
                preprocess = tube.create_group(group_name)
                preprocess.attrs["arcsinh_cofactor"] = arcsinh_cofactor or 0.0
                preprocess.create_dataset("marker_subset", data=np.asarray(selected, dtype=UTF8))
                for patient_id, patient in tube["raw"].items():
                    # Read first, then reorder in memory; h5py requires fancy indices
                    # to be increasing, while semantic panels may intentionally reorder markers.
                    cells = np.asarray(patient["raw_cell_matrix"][...], dtype=np.float32)[:, indices]
                    if arcsinh_cofactor is not None:
                        if arcsinh_cofactor <= 0:
                            raise ValueError("arcsinh_cofactor must be positive")
                        cells = np.arcsinh(cells / arcsinh_cofactor).astype(np.float32)
                    output = preprocess.create_group(patient_id)
                    output.create_dataset(
                        "processed_matrix", data=cells, compression="gzip", shuffle=True
                    )
