"""Console entry points for building, running, and evaluating FlowLOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from flowlot.evaluation.metrics import classification_metrics, regression_metrics
from flowlot.evaluation.reporting import plot_classification_suite, plot_regression_suite
from flowlot.evaluation.table_exporter import export_csv, export_latex, summarize_runs
from flowlot.io import (
    Stage2Loader,
    Stage2Organizer,
    build_stage1_from_manifest,
    import_legacy_flowcode_hdf5,
)
from flowlot.models.fusion import EarlyTubeFusion, LateTubeFusion
from flowlot.transport import compute_stage2_embeddings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowlot-build", description="Construct FlowLOT HDF5 files")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1", help="Build patient-centric raw HDF5")
    stage1.add_argument("--manifest", type=Path, required=True)
    stage1.add_argument("--output", type=Path, required=True)
    stage1.add_argument("--dataset", required=True)
    stage1.add_argument("--cells", default="all")
    stage1.add_argument("--seed", type=int, default=0)
    stage2 = commands.add_parser("stage2", help="Organize Stage 1 by tube")
    stage2.add_argument("--stage1", type=Path, required=True)
    stage2.add_argument("--output", type=Path, required=True)
    stage2.add_argument("--dataset", required=True)
    stage2.add_argument("--cells", default="all")
    stage2.add_argument("--marker-policy", choices=["intersection", "strict"], default="intersection")
    legacy = commands.add_parser("legacy", help="Migrate an earlier FlowCode HDF5 file")
    legacy.add_argument("--input", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument("--dataset")
    legacy.add_argument("--sample-level", help="e.g. sample_1000")
    preprocess = commands.add_parser("preprocess", help="Select/transform markers in Stage 2")
    preprocess.add_argument("--stage1", type=Path, default=Path("unused"))
    preprocess.add_argument("--stage2", type=Path, required=True)
    preprocess.add_argument("--dataset", required=True)
    preprocess.add_argument("--cells", default="all")
    preprocess.add_argument("--id", required=True)
    preprocess.add_argument("--markers", help="Comma-separated common marker subset")
    preprocess.add_argument("--arcsinh-cofactor", type=float)
    return parser


def build_main() -> None:
    args = _build_parser().parse_args()
    if args.command == "stage1":
        output = build_stage1_from_manifest(args.manifest, args.output, args.dataset, args.cells, args.seed)
    elif args.command == "stage2":
        output = Stage2Organizer(args.stage1, args.output).organize(
            args.dataset, args.cells, args.marker_policy
        )
    elif args.command == "legacy":
        output = import_legacy_flowcode_hdf5(
            args.input, args.output, args.dataset, args.sample_level
        )
    else:
        organizer = Stage2Organizer(args.stage1, args.stage2)
        markers = [value.strip() for value in args.markers.split(",")] if args.markers else None
        organizer.add_preprocess(
            args.dataset, args.cells, args.id, markers, args.arcsinh_cofactor
        )
        output = args.stage2
    print(json.dumps({"output": str(output), "command": args.command}, indent=2))


def run_main() -> None:
    parser = argparse.ArgumentParser(prog="flowlot-run", description="Compute Stage 2 LOT embeddings")
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cells", default="all")
    parser.add_argument("--preprocess", required=True)
    parser.add_argument(
        "--reference",
        choices=["patient0", "uniform", "gaussian", "barycenter", "pooled"],
        default="patient0",
    )
    parser.add_argument("--reference-size", type=int)
    parser.add_argument("--solver", choices=["hungarian", "linprog", "emd", "sinkhorn"], default="sinkhorn")
    parser.add_argument("--representation", choices=["displacement", "map"], default="displacement")
    parser.add_argument(
        "--embedding-id", help="Custom Stage 2 group name; defaults to reference_solver"
    )
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-patients",
        help="Comma-separated training-patient IDs used to fit the reference",
    )
    parser.add_argument("--no-store-transport", action="store_true")
    args = parser.parse_args()
    shapes = compute_stage2_embeddings(
        args.stage2,
        args.dataset,
        args.cells,
        args.preprocess,
        args.reference,
        args.solver,
        args.reference_size,
        args.representation,
        args.seed,
        store_transport=not args.no_store_transport,
        reference_patient_ids=(
            [value.strip() for value in args.reference_patients.split(",")]
            if args.reference_patients
            else None
        ),
        solver_kwargs={"reg": args.reg},
        embedding_id=args.embedding_id,
    )
    print(json.dumps({tube: list(shape) for tube, shape in shapes.items()}, indent=2))


def _load_tubes(
    path: Path, dataset: str, cells: str, preprocess: str, embedding: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    features: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, object] = {}
    with Stage2Loader(path) as loader:
        for tube in loader.tubes(dataset, cells):
            try:
                patient_ids, embeddings = loader.embeddings(dataset, cells, tube, preprocess, embedding)
            except KeyError:
                continue
            metadata = loader.metadata(dataset, cells, tube)
            label_map = dict(zip(metadata["patient_ids"], metadata["labels"]))
            features[tube] = dict(zip(patient_ids, embeddings))
            for patient in patient_ids:
                label = label_map[patient]
                if patient in targets and targets[patient] != label:
                    raise ValueError(f"Inconsistent label across tubes for {patient}")
                targets[patient] = label.item() if isinstance(label, np.generic) else label
    if not features:
        raise ValueError(f"No embeddings named {embedding!r} were found")
    return features, targets


def _subset(features: dict[str, dict[str, np.ndarray]], ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    allowed = set(ids)
    subset = {
        tube: {patient: value for patient, value in values.items() if patient in allowed}
        for tube, values in features.items()
    }
    return {tube: values for tube, values in subset.items() if values}


def eval_main() -> None:
    parser = argparse.ArgumentParser(prog="flowlot-eval", description="Cross-validate LOT embeddings")
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cells", default="all")
    parser.add_argument("--preprocess", required=True)
    parser.add_argument("--embedding", required=True, help="e.g. patient0_sinkhorn")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--fusion", choices=["early", "late"], default="early")
    parser.add_argument("--imputation", choices=["mean", "zero"], default="mean")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("flowlot_results"))
    args = parser.parse_args()
    features, raw_targets = _load_tubes(
        args.stage2, args.dataset, args.cells, args.preprocess, args.embedding
    )
    patient_ids = sorted(raw_targets)
    raw = np.asarray([raw_targets[patient] for patient in patient_ids])
    if args.task == "classification":
        encoder = LabelEncoder().fit(raw)
        targets = encoder.transform(raw)
        target_map = dict(zip(patient_ids, targets))
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        probabilities = np.zeros((len(patient_ids), len(encoder.classes_)))
    else:
        targets = raw.astype(float)
        target_map = dict(zip(patient_ids, targets))
        splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
        predictions = np.zeros(len(patient_ids))
    rows = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(patient_ids, targets if args.task == "classification" else None), start=1
    ):
        train_ids = [patient_ids[index] for index in train_index]
        test_ids = [patient_ids[index] for index in test_index]
        estimator = (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            if args.task == "classification"
            else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        )
        if args.fusion == "early":
            fusion = EarlyTubeFusion(args.imputation, add_indicators=True).fit(_subset(features, train_ids))
            train_x = fusion.transform(features, train_ids)
            test_x = fusion.transform(features, test_ids)
            model = estimator.fit(train_x, targets[train_index])
            if args.task == "classification":
                fold_output = model.predict_proba(test_x)
                probabilities[test_index] = fold_output
                metrics = classification_metrics(targets[test_index], fold_output)
            else:
                fold_output = model.predict(test_x)
                predictions[test_index] = fold_output
                metrics = regression_metrics(targets[test_index], fold_output)
        else:
            method = "soft" if args.task == "classification" else "mean"
            fusion = LateTubeFusion(estimator, args.task, method, random_state=args.seed).fit(
                features, {patient: target_map[patient] for patient in train_ids}
            )
            if args.task == "classification":
                fold_output = fusion.predict_proba(features, test_ids)
                probabilities[test_index] = fold_output
                metrics = classification_metrics(targets[test_index], fold_output)
            else:
                fold_output = fusion.predict(features, test_ids)
                predictions[test_index] = fold_output
                metrics = regression_metrics(targets[test_index], fold_output)
        rows.append({"model": f"LOT-{args.fusion}", "fold": fold, **metrics})
    args.output.mkdir(parents=True, exist_ok=True)
    export_csv(rows, args.output / "cross_validation.csv")
    summary = summarize_runs(rows)
    export_csv(summary, args.output / "summary.csv")
    export_latex(summary, args.output / "summary.tex")
    if args.task == "classification":
        overall = classification_metrics(targets, probabilities)
        plot_classification_suite(
            targets,
            probabilities,
            args.output / "classification_diagnostics",
            [str(value) for value in encoder.classes_],
            random_state=args.seed,
        )
    else:
        overall = regression_metrics(targets, predictions)
        plot_regression_suite(targets, predictions, args.output / "regression_diagnostics")
    print(json.dumps({"overall": overall, "output": str(args.output)}, indent=2))


def main() -> None:
    raise SystemExit("Use flowlot-build, flowlot-run, or flowlot-eval")


if __name__ == "__main__":
    main()
