"""LOT maps and Stage 2 embedding computation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from pytranskit.optrans.lot import LinearOptimalTransport
from numpy.typing import ArrayLike, NDArray

from flowlot.reference import ReferenceFactory

from pytranskit.optrans.lot.solvers import TransportResult


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
    transformer = LinearOptimalTransport(
        reference=np.asarray(reference, dtype=np.float64),
        solver=solver,
        representation=representation,
        solver_kwargs=solver_kwargs,
    )
    result = transformer.transform_sample(np.asarray(target, dtype=np.float64))
    return LOTResult(
        reference=np.asarray(result.reference, dtype=np.float64),
        transported=np.asarray(result.transported, dtype=np.float64),
        displacement=np.asarray(result.displacement, dtype=np.float64),
        embedding=np.asarray(result.embedding, dtype=np.float64),
        transport=result.transport,
    )


def _solve_lot_single_patient(
    patient_id: str,
    sample: np.ndarray,
    reference: np.ndarray,
    solver: str,
    solver_kwargs: dict[str, Any],
    store_transport: bool,
) -> tuple[str, np.ndarray, np.ndarray, float, bool, np.ndarray, np.ndarray | None]:
    result = compute_lot(
        reference,
        sample,
        solver=solver,
        representation="displacement",
        **(solver_kwargs or {}),
    )
    coupling = result.transport.coupling.astype(np.float32) if store_transport else None
    return (
        patient_id,
        result.transported.flatten("F").astype(np.float32),
        result.displacement.flatten("F").astype(np.float32),
        float(result.transport.cost),
        bool(result.transport.converged),
        result.transported.astype(np.float32),
        coupling,
    )


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
    embedding_id: str | None = None,
    reference_matrix: ArrayLike | None = None,
    freeze_reference: bool = True,
    n_jobs: int = 1,
) -> dict[str, tuple[int, int]]:
    """Compute one common reference and all patient LOT vectors per tube.
    
    Supports representation='both' to compute OT once and store both '_map'
    and '_disp' representations simultaneously, cutting solver time by 50%.
    Also saves and re-uses persistent references under tube['references'] so
    re-running or clicking does not change or re-sample the reference.
    When n_jobs > 1, patient LOT solves within each tube are computed in parallel.
    """

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

            # Check or load persistent reference
            ref_store_key = f"{reference_type}_size{size}"
            if reference_kwargs and "max_patients" in reference_kwargs:
                ref_store_key += f"_p{reference_kwargs['max_patients']}"
            ref_group = tube.require_group("references")

            if reference_matrix is not None:
                reference = np.asarray(reference_matrix, dtype=np.float64)
                if ref_store_key not in ref_group and freeze_reference:
                    r_out = ref_group.create_group(ref_store_key)
                    r_out.create_dataset("reference_matrix", data=reference.astype(np.float32))
                    r_out.create_dataset(
                        "reference_patient_ids", data=np.asarray(selected_reference_ids, dtype=UTF8)
                    )
                    r_out.attrs.update(
                        reference_type=reference_type, reference_size=size, random_state=random_state
                    )
            elif ref_store_key in ref_group and not overwrite:
                reference = np.asarray(
                    ref_group[ref_store_key]["reference_matrix"][...], dtype=np.float64
                )
            else:
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
                if freeze_reference:
                    if ref_store_key in ref_group:
                        del ref_group[ref_store_key]
                    r_out = ref_group.create_group(ref_store_key)
                    r_out.create_dataset("reference_matrix", data=reference.astype(np.float32))
                    r_out.create_dataset(
                        "reference_patient_ids", data=np.asarray(selected_reference_ids, dtype=UTF8)
                    )
                    r_out.attrs.update(
                        reference_type=reference_type, reference_size=size, random_state=random_state
                    )

            resolved_embedding_id = embedding_id or f"{reference_type}_{solver}"
            if not resolved_embedding_id.strip() or "/" in resolved_embedding_id:
                raise ValueError("embedding_id must be a non-empty HDF5 key without '/'")

            root = tube.require_group(f"lot_embeddings/{preprocess_id}")

            # Determine whether to store both map and displacement
            is_dual = representation.lower() in {"both", "dual"}
            if is_dual:
                base_id = resolved_embedding_id
                for suffix in ["_map", "_disp", "_displacement"]:
                    if base_id.endswith(suffix):
                        base_id = base_id[: -len(suffix)]
                        break
                target_reps = [("map", f"{base_id}_map"), ("displacement", f"{base_id}_disp")]
            else:
                target_reps = [(representation, resolved_embedding_id)]

            # Check existence
            for rep_name, target_id in target_reps:
                if target_id in root:
                    if not overwrite:
                        raise ValueError(f"Embedding {target_id} already exists in tube {tube_id}")
                    del root[target_id]

            # Solve optimal transport
            if n_jobs > 1 and len(patient_ids) > 1:
                from concurrent.futures import ProcessPoolExecutor
                patient_data = [
                    (pid, np.asarray(process[pid]["processed_matrix"][...], dtype=np.float64))
                    for pid in patient_ids
                ]
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    futures = [
                        pool.submit(
                            _solve_lot_single_patient,
                            pid,
                            mat,
                            reference,
                            solver,
                            solver_kwargs or {},
                            store_transport,
                        )
                        for pid, mat in patient_data
                    ]
                    results = [f.result() for f in futures]

                map_embeddings = [r[1] for r in results]
                disp_embeddings = [r[2] for r in results]
                costs = [r[3] for r in results]
                convergence = [r[4] for r in results]
                sorted_cell_dict = {r[0]: r[5] for r in results}
                transport_dict = {r[0]: r[6] for r in results if r[6] is not None}
            else:
                map_embeddings = []
                disp_embeddings = []
                costs = []
                convergence = []
                sorted_cell_dict = {}
                transport_dict = {}

                for patient_id in patient_ids:
                    sample = process[patient_id]["processed_matrix"][...]
                    result = compute_lot(
                        reference,
                        sample,
                        solver=solver,
                        representation="displacement",  # compute_lot computes BOTH transported and displacement
                        **(solver_kwargs or {}),
                    )
                    map_embeddings.append(result.transported.flatten("F").astype(np.float32))
                    disp_embeddings.append(result.displacement.flatten("F").astype(np.float32))
                    costs.append(result.transport.cost)
                    convergence.append(result.transport.converged)
                    sorted_cell_dict[patient_id] = result.transported.astype(np.float32)
                    if store_transport:
                        transport_dict[patient_id] = result.transport.coupling.astype(np.float32)

            # Store the resulting groups
            for rep_name, target_id in target_reps:
                out = root.create_group(target_id)
                out.attrs.update(
                    reference_type=reference_type,
                    solver=solver,
                    representation=rep_name,
                    flatten_order="F",
                    random_state=random_state,
                    reference_size=size,
                    store_transport=store_transport,
                    reference_kwargs_json=json.dumps(
                        reference_kwargs or {}, sort_keys=True, default=str
                    ),
                    solver_kwargs_json=json.dumps(solver_kwargs or {}, sort_keys=True, default=str),
                )
                out.create_dataset("reference_matrix", data=reference.astype(np.float32))
                out.create_dataset("patient_ids", data=np.asarray(patient_ids, dtype=UTF8))
                out.create_dataset(
                    "reference_patient_ids", data=np.asarray(selected_reference_ids, dtype=UTF8)
                )
                out.create_dataset("transport_costs", data=np.asarray(costs))
                out.create_dataset("converged", data=np.asarray(convergence, dtype=np.bool_))

                if (
                    is_dual
                    and rep_name == "displacement"
                    and f"{base_id}_map" in root
                    and "sorted_cell_matrices" in root[f"{base_id}_map"]
                ):
                    out["sorted_cell_matrices"] = root[f"{base_id}_map"]["sorted_cell_matrices"]
                else:
                    s_grp = out.create_group("sorted_cell_matrices")
                    for pid, smat in sorted_cell_dict.items():
                        s_grp.create_dataset(pid, data=smat, compression="lzf")

                if store_transport:
                    if (
                        is_dual
                        and rep_name == "displacement"
                        and f"{base_id}_map" in root
                        and "transport_matrices" in root[f"{base_id}_map"]
                    ):
                        out["transport_matrices"] = root[f"{base_id}_map"]["transport_matrices"]
                    else:
                        t_grp = out.create_group("transport_matrices")
                        for pid, cmat in transport_dict.items():
                            t_grp.create_dataset(pid, data=cmat, compression="lzf")

                emb_matrix = np.stack(map_embeddings if rep_name == "map" else disp_embeddings)
                out.create_dataset("embeddings", data=emb_matrix, compression="gzip")
                shapes[tube_id] = emb_matrix.shape

    return shapes

