import json
from pathlib import Path

import h5py
import nbformat
import numpy as np
import pandas as pd
import pytest

from flowlot.evaluation.repeated_benchmark import (
    aggregate_results,
    audit_registry,
    create_job_table,
    create_split_registry,
    export_legacy_splits_h5,
    load_jobs,
    load_registry,
    patient_bootstrap_confidence_intervals,
    run_job,
)


def _benchmark_stage2(path: Path) -> Path:
    rng = np.random.default_rng(9)
    patient_ids = [f"P{index:03d}" for index in range(40)]
    labels = np.repeat([0, 1], 20)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema="flowlot-stage2", schema_version="1.0")
        for tube_index, tube_name in enumerate(("T1", "T2")):
            tube = handle.create_group(f"demo/8/{tube_name}")
            metadata = tube.create_group("metadata")
            metadata.create_dataset(
                "patient_ids", data=np.asarray(patient_ids, dtype=string_type)
            )
            metadata.create_dataset("labels", data=labels)
            metadata.create_dataset("cell_counts", data=np.full(40, 8))
            metadata.create_dataset(
                "marker_descriptions", data=np.asarray(["A", "B"], dtype=string_type)
            )
            raw = tube.create_group("raw")
            preprocess = tube.create_group("preprocess_ab")
            preprocess.create_dataset(
                "marker_subset", data=np.asarray(["A", "B"], dtype=string_type)
            )
            vectors = []
            for patient_id, label in zip(patient_ids, labels):
                cells = rng.normal(label + tube_index * 0.1, 0.2, size=(8, 2)).astype("f4")
                raw.create_group(patient_id).create_dataset("raw_cell_matrix", data=cells)
                preprocess.create_group(patient_id).create_dataset(
                    "processed_matrix", data=cells
                )
                vectors.append(cells.mean(axis=0))
            embedding = tube.create_group("lot_embeddings/ab/patient0_sinkhorn")
            embedding.create_dataset(
                "patient_ids", data=np.asarray(patient_ids, dtype=string_type)
            )
            embedding.create_dataset("embeddings", data=np.asarray(vectors))
    return path


def test_nested_shared_splits_and_legacy_export(tmp_path):
    stage2 = _benchmark_stage2(tmp_path / "stage2.h5")
    registry = create_split_registry(
        stage2,
        "demo",
        8,
        tmp_path / "splits.json",
        train_per_class=(2, 4, 6, 8),
        repeats=2,
        test_size=0.5,
        seed=17,
        tubes=("T1", "T2"),
    )
    audit = pd.DataFrame(audit_registry(registry))
    assert list(audit.groupby("run")["test_ids_hash"].nunique()) == [1, 1]
    assert set(audit["n_train"]) == {4, 8, 12, 16}
    for split in registry["splits"]:
        cohorts = [set(split["train_ids_by_k"][str(k)]) for k in (2, 4, 6, 8)]
        assert all(left < right for left, right in zip(cohorts, cohorts[1:]))

    legacy = export_legacy_splits_h5(registry, tmp_path / "legacy.h5")
    with h5py.File(legacy) as handle:
        assert handle.attrs["flowlot_registry_hash"] == registry["registry_hash"]
        assert handle["run_0/subsamples/num_sub_per_cls_8/train_patient_ids"].shape == (16,)

    tampered = json.loads((tmp_path / "splits.json").read_text())
    tampered["seed"] += 1
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_registry(tmp_path / "tampered.json")

    fixed_cohort = [f"P{index:03d}" for index in [*range(4, 20), *range(20, 36)]]
    fixed = create_split_registry(
        stage2,
        "demo",
        8,
        tmp_path / "fixed.json",
        train_per_class=(2, 4, 6, 8),
        repeats=1,
        test_size=0.5,
        seed=17,
        tubes=("T1", "T2"),
        cohort_patient_ids=fixed_cohort,
    )
    assert fixed["fixed_cohort"] is True
    assert set(fixed["labels"]) == set(fixed_cohort)


def test_jobs_use_identical_ids_and_aggregate(tmp_path):
    stage2 = _benchmark_stage2(tmp_path / "stage2.h5")
    registry_path = tmp_path / "splits.json"
    registry = create_split_registry(
        stage2,
        "demo",
        8,
        registry_path,
        train_per_class=(2, 4, 6, 8),
        repeats=1,
        test_size=0.5,
        seed=4,
        tubes=("T1", "T2"),
    )
    jobs_path = tmp_path / "jobs.tsv"
    jobs = create_job_table(
        registry_path,
        jobs_path,
        models=("logistic", "nsc"),
        aggregations=("early_mean",),
    )
    assert len(jobs) == 8
    for job in load_jobs(jobs_path):
        shard = run_job(
            stage2,
            registry_path,
            jobs_path,
            tmp_path / "results",
            "ab",
            "patient0_sinkhorn",
            job["index"],
        )
        record = json.loads(shard.read_text())
        split = registry["splits"][0]
        assert record["test_ids"] == split["test_ids"]
        assert record["n_train"] == 2 * job["k"]

    integrity = aggregate_results(
        registry_path,
        jobs_path,
        tmp_path / "results/shards",
        tmp_path / "aggregate",
        bootstrap_iterations=50,
        bootstrap_seed=11,
    )
    assert integrity["completed_jobs"] == len(jobs)
    per_run = pd.read_csv(tmp_path / "aggregate/per_run.csv")
    assert len(per_run) == len(jobs)
    assert (per_run.groupby("k")["train_ids_hash"].nunique() == 1).all()
    assert per_run["test_ids_hash"].nunique() == 1
    paired = pd.read_csv(tmp_path / "aggregate/paired_comparisons.csv")
    assert set(paired["k"]) == {2, 4, 6, 8}
    confidence = pd.read_csv(tmp_path / "aggregate/bootstrap_ci.csv")
    assert len(confidence) == 8
    assert (confidence["confidence_level"] == 0.95).all()
    assert confidence["balanced_accuracy_ci_lower"].le(
        confidence["balanced_accuracy_ci_upper"]
    ).all()
    assert (tmp_path / "aggregate/bootstrap_comparison_k2.tex").exists()
    for suffix in ("pdf", "svg", "png"):
        assert (tmp_path / f"aggregate/aggregation_comparison.{suffix}").exists()


def test_repeated_benchmark_notebook_is_valid():
    target = Path("notebooks/04_split_n_LOT_based_classification.ipynb")
    if not target.exists():
        target = Path("notebooks/legacy_repeated_benchmark.ipynb")
    notebook = nbformat.read(str(target), as_version=4)
    assert len(notebook.cells) >= 10


def test_patient_bootstrap_is_reproducible_and_clusters_repeated_patients():
    frame = pd.DataFrame(
        [
            {
                "dataset": "demo",
                "model": "logistic",
                "aggregation": "early_mean",
                "tube": "ALL",
                "k": 2,
                "test_ids": ["A", "B", "C", "D"],
                "y_true": [0, 0, 1, 1],
                "probabilities": [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]],
            },
            {
                "dataset": "demo",
                "model": "logistic",
                "aggregation": "early_mean",
                "tube": "ALL",
                "k": 2,
                "test_ids": ["A", "B", "C", "D"],
                "y_true": [0, 0, 1, 1],
                "probabilities": [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]],
            },
        ]
    )
    first = patient_bootstrap_confidence_intervals(frame, iterations=100, seed=5)
    second = patient_bootstrap_confidence_intervals(frame, iterations=100, seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "n_unique_test_patients"] == 4
    assert first.loc[0, "accuracy_estimate"] == 1.0
