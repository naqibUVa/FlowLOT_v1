import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import h5py
import nbformat
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from flowlot.evaluation.reporting import (
    plot_clinical_threshold_suite,
    plot_regression_suite,
    plot_transport_geometry,
)
from flowlot.evaluation.table_exporter import export_csv, export_latex, summarize_runs
from flowlot.io import (
    Stage1Builder,
    Stage2Loader,
    Stage2Organizer,
    audit_manifest,
    audit_stage1,
    audit_stage2,
    build_stage1_from_manifest,
    create_manifest_from_folder,
    create_manifest_from_metadata,
    import_legacy_flowcode_hdf5,
    load_cytometry_file,
)
from flowlot.models.fusion import EarlyTubeFusion, LateTubeFusion
from flowlot.models.dual_task_models import LOTMLP
from flowlot.reference import ReferenceFactory
from flowlot.transport import compute_lot, compute_stage2_embeddings, solve_transport


def _make_hdf5(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(4)
    stage1, stage2 = tmp_path / "stage1.h5", tmp_path / "stage2.h5"
    markers = ["A", "B", "C"]
    with Stage1Builder(stage1, "w") as builder:
        builder.add_sample("cohort", 6, "P1", 0, "T1", rng.normal(size=(6, 3)), markers, 10)
        builder.add_sample("cohort", 6, "P2", 1, "T1", rng.normal(1, size=(6, 3)), markers, 11)
        builder.add_sample("cohort", 6, "P1", 0, "T2", rng.normal(size=(6, 3)), markers, 12)
    organizer = Stage2Organizer(stage1, stage2)
    organizer.organize("cohort", 6)
    organizer.add_preprocess("cohort", 6, "ab", ["A", "B"])
    return stage1, stage2


def test_hdf5_pipeline_and_loader(tmp_path):
    _, stage2 = _make_hdf5(tmp_path)
    shapes = compute_stage2_embeddings(
        stage2,
        "cohort",
        6,
        "ab",
        "patient0",
        "hungarian",
        representation="displacement",
        reference_patient_ids=["P1", "P2"],
    )
    assert shapes == {"T1": (2, 12), "T2": (1, 12)}
    with Stage2Loader(stage2) as loader:
        assert loader.tubes("cohort", 6) == ["T1", "T2"]
        ids, embeddings = loader.embeddings("cohort", 6, "T1", "ab", "patient0_hungarian")
        assert ids == ["P1", "P2"]
        assert embeddings.shape == (2, 12)
        np.testing.assert_allclose(embeddings[0], 0, atol=1e-6)
    with h5py.File(stage2) as h5:
        coupling = h5["cohort/6/T1/lot_embeddings/ab/patient0_hungarian/transport_matrices/P1"]
        assert coupling.shape == (6, 6)
        attributes = h5["cohort/6/T1/lot_embeddings/ab/patient0_hungarian"].attrs
        assert attributes["reference_size"] == 6
        assert attributes["solver_kwargs_json"] == "{}"
        t2_reference_ids = h5[
            "cohort/6/T2/lot_embeddings/ab/patient0_hungarian/reference_patient_ids"
        ][...]
        assert list(t2_reference_ids) == [b"P1"]
    compute_stage2_embeddings(
        stage2,
        "cohort",
        6,
        "ab",
        "patient0",
        "hungarian",
        representation="map",
        embedding_id="patient0_hungarian_map",
    )
    with Stage2Loader(stage2) as loader:
        _, map_embeddings = loader.embeddings(
            "cohort", 6, "T1", "ab", "patient0_hungarian_map"
        )
        assert map_embeddings.shape == (2, 12)
    stage1_inventory, stage1_issues = audit_stage1(tmp_path / "stage1.h5")
    assert len(stage1_inventory) == 3
    assert stage1_issues.empty
    stage2_inventories, stage2_issues = audit_stage2(stage2)
    assert len(stage2_inventories["raw"]) == 3
    assert len(stage2_inventories["preprocess"]) == 3
    assert len(stage2_inventories["embeddings"]) == 4
    assert stage2_issues.empty


def test_manifest_audit_reports_missing_and_duplicate_sources(tmp_path):
    source = tmp_path / "cells.npy"
    np.save(source, np.ones((3, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "patient_id,tube_id,path,label,markers\n"
        "P1,T1,cells.npy,AML,A;B\n"
        "P1,T1,missing.npy,AML,A;B\n",
        encoding="utf-8",
    )
    inventory, issues = audit_manifest(manifest)
    assert len(inventory) == 2
    assert set(issues["check"]) == {"source_exists", "unique_patient_tube"}


def test_folder_manifest_generation_and_format_readers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "P001_T1.npy", np.arange(12, dtype=np.float32).reshape(6, 2))
    np.savetxt(
        raw / "P001_T2.csv",
        np.arange(15, dtype=np.float32).reshape(5, 3),
        delimiter=",",
        header="A,B,C",
        comments="",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("patient_id,label\nP001,AML\n", encoding="utf-8")
    manifest = tmp_path / "manifests" / "cohort.csv"
    frame = create_manifest_from_folder(
        raw,
        manifest,
        r"(?P<patient_id>P[0-9]+)_(?P<tube_id>T[0-9]+)\.(?:npy|csv)$",
        labels,
        markers_by_tube={"T1": ["X", "Y"]},
    )
    assert frame[["patient_id", "tube_id", "label"]].values.tolist() == [
        ["P001", "T1", "AML"],
        ["P001", "T2", "AML"],
    ]
    assert not Path(frame.loc[0, "path"]).is_absolute()
    assert frame.loc[0, "markers"] == "X;Y"
    inventory, issues = audit_manifest(manifest)
    assert len(inventory) == 2
    assert issues.empty
    cells, markers = load_cytometry_file(raw / "P001_T2.csv")
    assert cells.shape == (5, 3)
    assert markers == ["A", "B", "C"]


def test_multi_dataset_fcs_reader_selects_first_dataset(tmp_path, monkeypatch):
    class MultipleDataSetsError(Exception):
        pass

    class FlowData:
        def __init__(self, *_args, **_kwargs):
            raise MultipleDataSetsError

    first = SimpleNamespace(
        events=[1.0, 2.0, 3.0, 4.0],
        channel_count=2,
        channels={"1": {"PnN": "event_ID"}, "2": {"PnN": "CD45", "PnS": "CD45"}},
    )
    second = SimpleNamespace(
        events=[9.0, 9.0],
        channel_count=2,
        channels={"1": {"PnN": "X"}, "2": {"PnN": "Y"}},
    )
    flowio = ModuleType("flowio")
    flowio.FlowData = FlowData
    flowio.read_multiple_data_sets = lambda *_args, **_kwargs: [first, second]
    exceptions = ModuleType("flowio.exceptions")
    exceptions.MultipleDataSetsError = MultipleDataSetsError
    monkeypatch.setitem(sys.modules, "flowio", flowio)
    monkeypatch.setitem(sys.modules, "flowio.exceptions", exceptions)
    source = tmp_path / "multi.fcs"
    source.write_bytes(b"test fixture")

    cells, markers = load_cytometry_file(source)

    np.testing.assert_array_equal(cells, [[1.0, 2.0], [3.0, 4.0]])
    assert markers == ["event_ID", "CD45"]


def test_metadata_driven_manifest_for_flowcapii(tmp_path):
    raw = tmp_path / "FlowCAPII" / "FCS"
    raw.mkdir(parents=True)
    np.save(raw / "sample_a.npy", np.ones((4, 2), dtype=np.float32))
    metadata = pd.DataFrame(
        {
            "FCS file": ["sample_a.npy"],
            "Individual": ["AML001"],
            "Tube number": [2],
            "Condition": ["AML"],
        }
    )
    manifest_path = tmp_path / "manifests" / "flowcapii.csv"
    manifest = create_manifest_from_metadata(
        raw,
        metadata,
        manifest_path,
        file_column="FCS file",
        patient_id_column="Individual",
        tube_id_column="Tube number",
        label_column="Condition",
        tube_prefix="P",
        markers_by_tube={"P2": ["A", "B"]},
    )
    assert manifest.loc[0, ["patient_id", "tube_id", "label", "markers"]].tolist() == [
        "AML001",
        "P2",
        "AML",
        "A;B",
    ]
    inventory, issues = audit_manifest(manifest_path)
    assert len(inventory) == 1
    assert issues.empty


def test_multi_count_builds_are_nested_and_preserve_original_count(tmp_path):
    raw = tmp_path / "cells.npy"
    cells = np.column_stack((np.arange(20), np.arange(20) + 100)).astype(np.float32)
    np.save(raw, cells)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "patient_id,tube_id,path,label,markers\nP001,T1,cells.npy,AML,A;B\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage1.h5"
    build_stage1_from_manifest(manifest, output, "cohort", 5, seed=42, mode="w")
    build_stage1_from_manifest(manifest, output, "cohort", 10, seed=42, mode="a")
    with h5py.File(output) as handle:
        small_group = handle["cohort/5/P001_label_AML/T1"]
        large_group = handle["cohort/10/P001_label_AML/T1"]
        small = small_group["raw_cell_matrix"][...]
        large = large_group["raw_cell_matrix"][...]
        assert {tuple(row) for row in small}.issubset({tuple(row) for row in large})
        assert small_group.attrs["counts"] == 20
        assert large_group.attrs["counts"] == 20


def test_event_population_labels_are_aligned_counted_and_copied_to_stage2(tmp_path):
    raw = tmp_path / "raw"
    detailed = tmp_path / "labels"
    raw.mkdir()
    detailed.mkdir()
    event_ids = np.arange(1, 11, dtype=np.float32)
    cells = np.column_stack((event_ids, event_ids * 2, event_ids * 3))
    np.save(raw / "BLAST110_1_P1.npy", cells)
    (detailed / "BLAST110_1_P1.csv").write_text(
        "event_ID,WBC,Blast,LAIP\n"
        "1,1,1,0\n2,1,1,0\n3,1,1,1\n4,1,1,1\n5,1,0,0\n"
        "6,1,0,0\n7,1,0,0\n8,1,0,0\n9,0,0,0\n10,0,0,0\n",
        encoding="utf-8",
    )
    sample_info = tmp_path / "sample_info.csv"
    sample_info.write_text(
        "BLAST110_ID,sample_type\nBLAST110_1_P1,AML_Dx\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.csv"
    create_manifest_from_folder(
        raw,
        manifest,
        r"BLAST110_(?P<patient_id>[0-9]+)_(?P<tube_id>P[0-9]+)\.npy$",
        sample_info,
        markers_by_tube={"P1": ["event_ID", "A", "B"]},
        labels_id_column="BLAST110_ID",
        labels_label_column="sample_type",
        labels_match="file_id",
        event_labels_root=detailed,
        population_columns=["WBC", "Blast", "LAIP"],
    )
    stage1, stage2 = tmp_path / "stage1.h5", tmp_path / "stage2.h5"
    build_stage1_from_manifest(manifest, stage1, "BLAST110", 5, seed=4)
    inventory, issues = audit_stage1(stage1)
    assert len(inventory) == 1
    assert issues.empty
    with h5py.File(stage1) as handle:
        tube = handle["BLAST110/5/1_label_AML_Dx/P1"]
        annotations = tube["population_annotations"][...]
        counts = tube["population_counts"][...]
        np.testing.assert_allclose(counts[:, 0], annotations.sum(axis=0))
        np.testing.assert_allclose(counts[:, 1], [8, 4, 2])
        np.testing.assert_allclose(counts[:, 3], [100, 50, 25])
        selected_ids = [int(value.decode()) for value in tube["sample_event_ids"][...]]
        np.testing.assert_array_equal(
            tube["raw_cell_matrix"][:, 0].astype(int), selected_ids
        )
        assert tube.attrs["counts"] == 10
        assert tube.attrs["annotated_event_count"] == 10
    Stage2Organizer(stage1, stage2).organize("BLAST110", 5)
    with Stage2Loader(stage2) as loader:
        counts = loader.population_counts("BLAST110", 5, "P1", "1")
        assert counts["Blast"]["original_count"] == 4
        assert counts["LAIP"]["original_pct_wbc"] == 25


def test_legacy_flowcode_migration(tmp_path):
    legacy, migrated = tmp_path / "legacy.h5", tmp_path / "migrated.h5"
    with h5py.File(legacy, "w") as h5:
        patient = h5.create_group("Dataset/BLAST110/sample_6/patient_1")
        patient.create_dataset("label", data=np.bytes_("AML"))
        tube = patient.create_group("tube_P1")
        tube.create_dataset("data", data=np.ones((6, 2), dtype=np.float32))
        tube.create_dataset("markers", data=np.asarray([b"A", b"B"]))
        tube.create_dataset("counts", data=[6, 1, 20, 2])
    import_legacy_flowcode_hdf5(legacy, migrated, "BLAST110", "sample_6")
    with h5py.File(migrated) as h5:
        group = h5["BLAST110/6/1_label_AML/P1"]
        assert group["raw_cell_matrix"].shape == (6, 2)
        assert group.attrs["counts"] == 20


def test_transport_solvers_and_references():
    rng = np.random.default_rng(2)
    samples = [rng.normal(i, 1, size=(5, 2)) for i in range(3)]
    factory = ReferenceFactory(3)
    for kind in ["patient0", "uniform", "gaussian", "pooled", "barycenter"]:
        assert factory.create(kind, samples, 4).shape == (4, 2)
    for solver in ["hungarian", "linprog", "emd", "sinkhorn"]:
        result = solve_transport(samples[0], samples[1], solver, reg=0.05)
        np.testing.assert_allclose(result.coupling.sum(), 1, atol=1e-5)
    lot = compute_lot(samples[0], samples[1], "emd")
    assert lot.embedding.shape == (10,)


def test_missing_tube_fusion():
    features = {
        "T1": {"P1": [0.0, 0.0], "P2": [1.0, 1.0], "P3": [0.1, 0.2], "P4": [0.9, 0.8]},
        "T2": {"P1": [0.0], "P2": [1.0], "P4": [0.8]},
    }
    ids = ["P1", "P2", "P3", "P4"]
    early = EarlyTubeFusion().fit_transform(features, ids)
    assert early.shape == (4, 5)
    late = LateTubeFusion(LogisticRegression(), task="classification", method="soft").fit(
        features, dict(zip(ids, [0, 1, 0, 1]))
    )
    probabilities = late.predict_proba(features, ids)
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(1), 1)
    regression = LateTubeFusion(Ridge(), task="regression", method="median").fit(
        features, dict(zip(ids, [0.0, 1.0, 0.1, 0.9]))
    )
    assert regression.predict(features, ids).shape == (4,)


def test_dual_task_heads():
    import torch

    features = torch.randn(3, 6)
    assert LOTMLP(6, "classification", num_classes=4)(features).shape == (3, 4)
    assert LOTMLP(6, "regression")(features).shape == (3,)


def test_reports_schema_and_notebook(tmp_path):
    outputs = plot_regression_suite([0, 1, 2], [0.1, 0.9, 2.1], tmp_path / "regression")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    clinical = plot_clinical_threshold_suite(
        [0.05, 0.2, 0.8, 2.0, 6.0, 10.0],
        [0.08, 0.15, 1.2, 1.8, 7.0, 8.0],
        tmp_path / "clinical",
        groups=["Dx", "FU", "Dx", "FU", "Dx", "FU"],
    )
    assert len(clinical) == 3
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(8, 4))
    geometry = plot_transport_geometry(
        reference,
        rng.normal(size=(10, 4)),
        reference + 0.1,
        tmp_path / "geometry3d",
        dimensions=3,
    )
    assert len(geometry) == 3
    rows = [
        {"model": "A", "fold": 1, "rmse": 0.3, "r2": 0.8},
        {"model": "A", "fold": 2, "rmse": 0.2, "r2": 0.9},
        {"model": "B", "fold": 1, "rmse": 0.4, "r2": 0.7},
    ]
    export_csv(rows, tmp_path / "runs.csv")
    export_latex(summarize_runs(rows), tmp_path / "summary.tex")
    assert "\\toprule" in (tmp_path / "summary.tex").read_text()
    schema = json.loads(Path("docs/stage2_schema.json").read_text())
    assert schema["$id"].endswith("stage2-1.0.json")
    notebook = nbformat.read("notebooks/benchmark_pipeline.ipynb", as_version=4)
    assert len(notebook.cells) >= 10
