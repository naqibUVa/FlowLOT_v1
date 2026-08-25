"""Train and compare LOT, FlowSOM, and deep cytometry classifiers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset import CytometryDataset
from metrics import classification_metrics
from models import MODEL_REGISTRY
from models.flowsom_baseline import FlowSOMClassifier
from train import TrainConfig, _loader, fit_deep_model, predict


def _materialize(dataset: CytometryDataset) -> tuple[list[np.ndarray], np.ndarray]:
    samples, labels = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        samples.append(item["cells"].numpy())
        labels.append(item["label"])
    return samples, np.asarray(labels)


def _read_vector(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            key = "lot" if "lot" in archive else archive.files[0]
            value = archive[key]
    elif path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("lot", value.get("x"))
        if torch.is_tensor(value):
            value = value.numpy()
    else:
        raise ValueError(f"Unsupported LOT representation: {path}")
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _load_lot_split(
    manifest: Path, split: str, column: str
) -> tuple[np.ndarray, np.ndarray]:
    vectors, labels = [], []
    with manifest.resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Manifest needs a {column!r} column to benchmark LOT")
        for row in reader:
            if row["split"].strip().lower() != split.lower():
                continue
            path = Path(row[column])
            if not path.is_absolute():
                path = manifest.resolve().parent / path
            vectors.append(_read_vector(path))
            labels.append(int(row["label"]))
    if not vectors:
        raise ValueError(f"No {split!r} LOT samples found")
    return np.stack(vectors), np.asarray(labels)


def evaluate_lot(manifest: Path, column: str, seed: int) -> dict[str, float]:
    train_x, train_y = _load_lot_split(manifest, "train", column)
    test_x, test_y = _load_lot_split(manifest, "test", column)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(train_x, train_y)
    return classification_metrics(test_y, classifier.predict_proba(test_x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--models",
        default="flowsom," + ",".join(sorted(MODEL_REGISTRY)),
        help="Comma-separated names; add 'lot' for precomputed LOT vectors",
    )
    parser.add_argument("--lot-column", default="lot_path")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--max-cells", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _write_results(output_dir: Path, results: dict[str, dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *metric_names])
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"model": name, **metrics})


def main() -> None:
    args = parse_args()
    requested = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    valid = {"lot", "flowsom", *MODEL_REGISTRY}
    unknown = set(requested).difference(valid)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    train_data = CytometryDataset.from_manifest(
        args.manifest, "train", max_cells=args.max_cells, random_subsample=True, seed=args.seed
    )
    validation_data = CytometryDataset.from_manifest(
        args.manifest, "validation", max_cells=args.max_cells, seed=args.seed
    )
    test_data = CytometryDataset.from_manifest(
        args.manifest, "test", max_cells=args.max_cells, seed=args.seed
    )
    in_features = train_data[0]["cells"].shape[1]
    train_labels = {record.label for record in train_data.records}
    if train_labels != set(range(len(train_labels))):
        raise ValueError("Training labels must be contiguous integers starting at zero")
    results: dict[str, dict[str, float]] = {}

    if "lot" in requested:
        results["lot"] = evaluate_lot(args.manifest, args.lot_column, args.seed)

    if "flowsom" in requested:
        flowsom_train_data = CytometryDataset.from_manifest(
            args.manifest, "train", max_cells=args.max_cells, seed=args.seed
        )
        train_samples, labels = _materialize(flowsom_train_data)
        test_samples, test_labels = _materialize(test_data)
        baseline = FlowSOMClassifier(random_state=args.seed).fit(train_samples, labels)
        results["flowsom"] = classification_metrics(
            test_labels, baseline.predict_proba(test_samples)
        )

    for name in requested:
        if name not in MODEL_REGISTRY:
            continue
        config = TrainConfig(
            model=name,
            in_features=in_features,
            num_classes=len(train_labels),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            amp=not args.no_amp,
            seed=args.seed,
        )
        model, history, _ = fit_deep_model(
            train_data, validation_data, config, device=args.device
        )
        device = next(model.parameters()).device
        test_labels, probabilities, _ = predict(model, _loader(test_data, config, False), device)
        results[name] = classification_metrics(test_labels, probabilities)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(config), "history": history},
            args.output_dir / f"{name}.pt",
        )
        _write_results(args.output_dir, results)

    _write_results(args.output_dir, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
