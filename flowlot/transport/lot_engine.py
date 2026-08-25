"""LOT maps and Stage 2 embedding computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from flowlot.reference import ReferenceFactory

from .solvers import TransportResult, solve_transport


UTF8 = h5py.string_dtype("utf-8")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LOTResult:
    reference: FloatArray
    transported: FloatArray
    displacement: FloatArray
    embedding: FloatArray
    transport: TransportResult


def _take_rows(dataset: h5py.Dataset, indices: np.ndarray) -> FloatArray:
    """Read arbitrary/repeated HDF5 rows through one sorted unique selection."""

    unique, inverse = np.unique(indices.astype(np.int64), return_inverse=True)
    return np.asarray(dataset[unique], dtype=np.float64)[inverse]


def _reference_from_h5(
    datasets: list[h5py.Dataset],
    kind: str,
    size: int,
    random_state: int,
    kwargs: dict[str, Any],
) -> FloatArray:
    """Generate references without accumulating every patient matrix in RAM."""

    normalized = kind.lower().replace("_", "")
    rng = np.random.default_rng(random_state)
    dimension = datasets[0].shape[1]
    if normalized in {"patient", "patient0", "patientreference"}:
        index = int(kwargs.get("index", 0))
        dataset = datasets[index]
        if size == len(dataset):
            return np.asarray(dataset[...], dtype=np.float64)
        chosen = rng.choice(len(dataset), size=size, replace=len(dataset) < size)
        return _take_rows(dataset, chosen)
    if normalized in {"uniform", "syntheticuniform"}:
        lower = np.full(dimension, np.inf)
        upper = np.full(dimension, -np.inf)
        for dataset in datasets:
            for start in range(0, len(dataset), 65_536):
                block = dataset[start : start + 65_536]
                lower = np.minimum(lower, block.min(axis=0))
                upper = np.maximum(upper, block.max(axis=0))
        return rng.uniform(lower, upper, size=(size, dimension))
    if normalized in {"gaussian", "syntheticgaussian"}:
        count = 0
        total = np.zeros(dimension)
        second = np.zeros((dimension, dimension))
        for dataset in datasets:
            for start in range(0, len(dataset), 65_536):
                block = np.asarray(dataset[start : start + 65_536], dtype=np.float64)
                count += len(block)
                total += block.sum(axis=0)
                second += block.T @ block
        mean = total / count
        covariance = (second - count * np.outer(mean, mean)) / max(count - 1, 1)
        covariance += float(kwargs.get("jitter", 1e-6)) * np.eye(dimension)
        return rng.multivariate_normal(mean, covariance, size=size)
    if normalized in {"pooled", "pooledsubsample", "subsampled"}:
        lengths = np.asarray([len(dataset) for dataset in datasets])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        selected = rng.choice(int(offsets[-1]), size=size, replace=offsets[-1] < size)
        source_indices = np.searchsorted(offsets[1:], selected, side="right")
        output = np.empty((size, dimension), dtype=np.float64)
        for source_index, dataset in enumerate(datasets):
            positions = np.flatnonzero(source_indices == source_index)
            if len(positions):
                output[positions] = _take_rows(dataset, selected[positions] - offsets[source_index])
        return output
    if normalized in {"barycenter", "wassersteinbarycenter"}:
        maximum_patients = int(kwargs.get("max_patients", 10))
        maximum_cells = int(kwargs.get("max_cells_per_patient", 2048))
        samples = []
        for dataset in datasets[:maximum_patients]:
            chosen = rng.choice(
                len(dataset), size=min(len(dataset), maximum_cells), replace=False
            )
            samples.append(_take_rows(dataset, chosen))
        return ReferenceFactory(random_state).barycenter(
            samples,
            size,
            max_patients=maximum_patients,
            max_cells_per_patient=maximum_cells,
            max_iter=int(kwargs.get("max_iter", 50)),
        )
    raise ValueError(f"Unknown reference type: {kind}")


def compute_lot(
    reference: ArrayLike,
    target: ArrayLike,
    solver: str = "sinkhorn",
    representation: str = "displacement",
    **solver_kwargs: Any,
) -> LOTResult:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    transport = solve_transport(target, reference, solver=solver, **solver_kwargs)
    reference_weights = transport.coupling.sum(axis=0)
    transported = reference.copy()
    active = reference_weights > np.finfo(float).eps
    transported[active] = (
        transport.coupling[:, active].T @ target
    ) / reference_weights[active, None]
    displacement = (transported - reference) * np.sqrt(reference_weights[:, None])
    if representation == "displacement":
        matrix = displacement
    elif representation in {"map", "legacy_map"}:
        matrix = transported
    else:
        raise ValueError("representation must be 'displacement' or 'map'")
    # Fortran order preserves the earlier FlowCode convention: one marker block at a time.
    embedding = matrix.reshape(-1, order="F")
    return LOTResult(reference, transported, displacement, embedding, transport)


def compute_stage2_embeddings(
    stage2_path: str | Path,
    dataset_name: str,
    subsampled_cell_count: str | int,
    preprocess_id: str,
    reference_type: str = "patient0",
    solver: str = "sinkhorn",
    reference_size: int | None = None,
    representation: str = "displacement",
    random_state: int = 0,
    store_transport: bool = True,
    overwrite: bool = True,
    reference_patient_ids: Sequence[str] | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> dict[str, tuple[int, int]]:
    """Compute one common reference and all patient LOT vectors per tube."""

    base = f"{dataset_name}/{subsampled_cell_count}"
    shapes: dict[str, tuple[int, int]] = {}
    with h5py.File(stage2_path, "a") as h5:
        if base not in h5:
            raise KeyError(f"Stage 2 group /{base} does not exist")
        for tube_id, tube in h5[base].items():
            process_path = f"preprocess_{preprocess_id}"
            if process_path not in tube:
                raise KeyError(f"Tube {tube_id} has no {process_path}")
            process = tube[process_path]
            patient_ids = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in tube["metadata/patient_ids"][...]
                if str(value.decode() if isinstance(value, bytes) else value) in process
            ]
            if not patient_ids:
                continue
            size = reference_size or len(process[patient_ids[0]]["processed_matrix"])
            requested_reference_ids = list(reference_patient_ids or patient_ids)
            # Missing tubes are expected: fit each tube reference from the
            # available members of the requested training cohort.
            selected_reference_ids = [
                patient_id for patient_id in requested_reference_ids if patient_id in patient_ids
            ]
            if not selected_reference_ids:
                raise ValueError(f"Tube {tube_id} has no requested reference patients")
            reference_datasets = [
                process[patient_id]["processed_matrix"] for patient_id in selected_reference_ids
            ]
            reference = _reference_from_h5(
                reference_datasets,
                reference_type,
                size,
                random_state,
                reference_kwargs or {},
            )
            embedding_id = f"{reference_type}_{solver}"
            root = tube.require_group(f"lot_embeddings/{preprocess_id}")
            if embedding_id in root:
                if not overwrite:
                    raise ValueError(f"Embedding {embedding_id} already exists in tube {tube_id}")
                del root[embedding_id]
            output = root.create_group(embedding_id)
            output.attrs.update(
                reference_type=reference_type,
                solver=solver,
                representation=representation,
                flatten_order="F",
            )
            output.create_dataset("reference_matrix", data=reference.astype(np.float32))
            output.create_dataset("patient_ids", data=np.asarray(patient_ids, dtype=UTF8))
            output.create_dataset(
                "reference_patient_ids", data=np.asarray(selected_reference_ids, dtype=UTF8)
            )
            sorted_group = output.create_group("sorted_cell_matrices")
            transport_group = output.create_group("transport_matrices") if store_transport else None
            embeddings = []
            costs = []
            convergence = []
            for patient_id in patient_ids:
                sample = process[patient_id]["processed_matrix"][...]
                result = compute_lot(
                    reference,
                    sample,
                    solver=solver,
                    representation=representation,
                    **(solver_kwargs or {}),
                )
                embeddings.append(result.embedding.astype(np.float32))
                costs.append(result.transport.cost)
                convergence.append(result.transport.converged)
                sorted_group.create_dataset(
                    patient_id, data=result.transported.astype(np.float32), compression="gzip"
                )
                if transport_group is not None:
                    transport_group.create_dataset(
                        patient_id,
                        data=result.transport.coupling.astype(np.float32),
                        compression="gzip",
                    )
            matrix = np.stack(embeddings)
            output.create_dataset("embeddings", data=matrix, compression="gzip")
            output.create_dataset("transport_costs", data=np.asarray(costs))
            output.create_dataset("converged", data=np.asarray(convergence, dtype=np.bool_))
            shapes[tube_id] = matrix.shape
    return shapes
