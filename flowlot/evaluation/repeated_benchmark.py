"""Shared-split repeated classification benchmarks and HPC result aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import InMemoryCytometryDataset, cytometry_collate
from flowlot.evaluation.metrics import classification_metrics
from flowlot.evaluation.reporting import apply_nature_style, save_figure
from flowlot.evaluation.table_exporter import export_latex
from flowlot.io import Stage2Loader
from flowlot.models.baselines import BASELINE_REGISTRY
from flowlot.models.classical import CELL_MODELS, CLASSICAL_MODELS, make_classical_classifier
from flowlot.models.fusion import EarlyTubeFusion, LateTubeFusion
from models.flowsom_baseline import FlowSOMClassifier


REGISTRY_VERSION = "1.0"
JOB_FIELDS = ["index", "job_id", "model", "aggregation", "tube", "run", "k"]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def _parse_csv(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _cohort_labels(
    stage2: Path,
    dataset: str,
    cell_count: str,
    requested_tubes: Sequence[str] | None,
    patient_policy: str,
) -> tuple[list[str], list[str], dict[str, int], list[str]]:
    with Stage2Loader(stage2) as loader:
        available_tubes = loader.tubes(dataset, cell_count)
        tubes = list(requested_tubes or available_tubes)
        missing = set(tubes).difference(available_tubes)
        if missing:
            raise ValueError(f"Unknown Stage 2 tubes: {sorted(missing)}")
        patients_by_tube: dict[str, set[str]] = {}
        raw_labels: dict[str, object] = {}
        for tube in tubes:
            metadata = loader.metadata(dataset, cell_count, tube)
            patients = set(metadata["patient_ids"])
            patients_by_tube[tube] = patients
            for patient, label in zip(metadata["patient_ids"], metadata["labels"]):
                scalar = label.item() if isinstance(label, np.generic) else label
                if patient in raw_labels and raw_labels[patient] != scalar:
                    raise ValueError(f"Patient {patient} has inconsistent tube labels")
                raw_labels[patient] = scalar
    if patient_policy == "intersection":
        patients = set.intersection(*(patients_by_tube[tube] for tube in tubes))
    elif patient_policy == "union":
        patients = set.union(*(patients_by_tube[tube] for tube in tubes))
    else:
        raise ValueError("patient_policy must be intersection or union")
    patient_ids = sorted(patients)
    text_labels = [str(raw_labels[patient]) for patient in patient_ids]
    encoder = LabelEncoder().fit(text_labels)
    encoded = encoder.transform(text_labels)
    return patient_ids, encoder.classes_.tolist(), dict(zip(patient_ids, map(int, encoded))), tubes


def create_split_registry(
    stage2: str | Path,
    dataset: str,
    cell_count: str | int,
    output: str | Path,
    train_per_class: Sequence[int] = (2, 4, 6, 8),
    repeats: int = 10,
    test_size: float = 0.5,
    seed: int = 42,
    tubes: Sequence[str] | None = None,
    patient_policy: str = "intersection",
) -> dict[str, Any]:
    """Create nested per-class training sets and one fixed test set per repeat."""

    stage2 = Path(stage2).resolve()
    sizes = sorted(set(map(int, train_per_class)))
    if not sizes or sizes[0] < 1:
        raise ValueError("train_per_class must contain positive integers")
    patient_ids, class_names, labels, resolved_tubes = _cohort_labels(
        stage2, dataset, str(cell_count), tubes, patient_policy
    )
    y = np.asarray([labels[patient] for patient in patient_ids])
    counts = np.bincount(y, minlength=len(class_names))
    if np.any(counts < 2):
        raise ValueError(f"Every class needs at least two patients; counts={counts.tolist()}")
    splitter = StratifiedShuffleSplit(
        n_splits=repeats, test_size=test_size, random_state=seed
    )
    split_records = []
    for run, (pool_index, test_index) in enumerate(splitter.split(patient_ids, y)):
        pool_ids = np.asarray(patient_ids)[pool_index]
        pool_labels = y[pool_index]
        test_ids = np.asarray(patient_ids)[test_index].tolist()
        rng = np.random.default_rng(np.random.SeedSequence([seed, run]))
        class_order: dict[int, list[str]] = {}
        for label in range(len(class_names)):
            candidates = pool_ids[pool_labels == label].copy()
            rng.shuffle(candidates)
            if len(candidates) < sizes[-1]:
                raise ValueError(
                    f"Run {run}, class {class_names[label]!r}: training pool has "
                    f"{len(candidates)} patients, fewer than requested k={sizes[-1]}. "
                    "Reduce --test-size or --train-per-class."
                )
            class_order[label] = candidates.tolist()
        subsets: dict[str, list[str]] = {}
        for size in sizes:
            selected = [
                patient
                for label in range(len(class_names))
                for patient in class_order[label][:size]
            ]
            subsets[str(size)] = selected
        split_records.append(
            {
                "run": run,
                "train_pool_ids": pool_ids.tolist(),
                "test_ids": test_ids,
                "train_ids_by_k": subsets,
            }
        )
    registry: dict[str, Any] = {
        "version": REGISTRY_VERSION,
        "stage2": str(stage2),
        "dataset": dataset,
        "cell_count": str(cell_count),
        "tubes": resolved_tubes,
        "patient_policy": patient_policy,
        "class_names": class_names,
        "labels": labels,
        "train_per_class": sizes,
        "repeats": repeats,
        "test_size": test_size,
        "seed": seed,
        "splits": split_records,
    }
    registry["registry_hash"] = _canonical_hash(registry)
    _atomic_json(registry, Path(output))
    return registry


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if "registry_hash" not in registry:
        raise ValueError("Split registry has no checksum")
    expected = registry.pop("registry_hash")
    actual = _canonical_hash(registry)
    registry["registry_hash"] = expected
    if actual != expected:
        raise ValueError(f"Split registry checksum mismatch: expected {expected}, found {actual}")
    return registry


def audit_registry(registry: dict[str, Any]) -> list[dict[str, object]]:
    """Validate disjointness, balance, nesting, and repeated test invariants."""

    labels = registry["labels"]
    sizes = registry["train_per_class"]
    rows = []
    for split in registry["splits"]:
        test = set(split["test_ids"])
        previous: set[str] = set()
        for size in sizes:
            train = set(split["train_ids_by_k"][str(size)])
            if train & test:
                raise ValueError(f"Run {split['run']}, k={size}: train/test overlap")
            if not previous.issubset(train):
                raise ValueError(f"Run {split['run']}, k={size}: subsets are not nested")
            counts = np.bincount([labels[patient] for patient in train], minlength=len(registry["class_names"]))
            if not np.all(counts == size):
                raise ValueError(f"Run {split['run']}, k={size}: unbalanced training set")
            rows.append(
                {
                    "run": split["run"],
                    "k": size,
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_ids_hash": _canonical_hash(sorted(train))[:16],
                    "test_ids_hash": _canonical_hash(sorted(test))[:16],
                }
            )
            previous = train
    return rows


def export_legacy_splits_h5(registry: dict[str, Any], output: str | Path) -> Path:
    """Write the historical ``run_N/subsamples`` layout from the audited registry.

    Unlike the original notebook, the exported k-sized cohorts are nested and
    deterministic. The registry checksum is stored at the root so legacy and new
    results can be traced to the same patient assignments.
    """

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = registry["labels"]
    with h5py.File(path, "w") as handle:
        handle.attrs["flowlot_registry_hash"] = registry["registry_hash"]
        handle.attrs["flowlot_registry_version"] = registry["version"]
        handle.attrs["nested_training_subsets"] = True
        for split in registry["splits"]:
            group = handle.create_group(f"run_{split['run']}")
            train_ids = split["train_pool_ids"]
            test_ids = split["test_ids"]
            group.create_dataset("train_patient_ids", data=np.asarray(train_ids, dtype="S"))
            group.create_dataset("test_patient_ids", data=np.asarray(test_ids, dtype="S"))
            group.create_dataset("train_labels", data=[labels[patient] for patient in train_ids])
            group.create_dataset("test_labels", data=[labels[patient] for patient in test_ids])
            subsamples = group.create_group("subsamples")
            for size in registry["train_per_class"]:
                selected = split["train_ids_by_k"][str(size)]
                subgroup = subsamples.create_group(f"num_sub_per_cls_{size}")
                subgroup.create_dataset(
                    "train_patient_ids", data=np.asarray(selected, dtype="S")
                )
                subgroup.create_dataset(
                    "train_labels", data=[labels[patient] for patient in selected]
                )
    return path


def create_job_table(
    registry_path: str | Path,
    output: str | Path,
    models: Sequence[str],
    aggregations: Sequence[str] = ("single", "early_mean", "late_soft"),
) -> list[dict[str, object]]:
    registry = load_registry(registry_path)
    allowed_models = set(CLASSICAL_MODELS) | set(CELL_MODELS)
    unknown = set(models).difference(allowed_models)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    allowed_aggregations = {"single", "early_mean", "early_zero", "late_soft"}
    invalid_aggregations = set(aggregations).difference(allowed_aggregations)
    if invalid_aggregations:
        raise ValueError(f"Unknown aggregations: {sorted(invalid_aggregations)}")
    if registry["patient_policy"] == "union" and (
        set(models) & set(CELL_MODELS) or "single" in aggregations
    ):
        raise ValueError(
            "The union cohort supports missing-tube-aware early/late fusion only. "
            "Use patient_policy='intersection' for cell or single-tube jobs."
        )
    jobs: list[dict[str, object]] = []
    for model in models:
        configurations: list[tuple[str, str]] = []
        if model in CELL_MODELS:
            configurations.extend(("single", tube) for tube in registry["tubes"])
        else:
            for aggregation in aggregations:
                if aggregation == "single":
                    configurations.extend((aggregation, tube) for tube in registry["tubes"])
                else:
                    configurations.append((aggregation, "ALL"))
        for aggregation, tube in configurations:
            for run in range(registry["repeats"]):
                for size in registry["train_per_class"]:
                    key = f"{model}|{aggregation}|{tube}|r{run}|k{size}"
                    jobs.append(
                        {
                            "index": len(jobs),
                            "job_id": hashlib.sha256(key.encode()).hexdigest()[:20],
                            "model": model,
                            "aggregation": aggregation,
                            "tube": tube,
                            "run": run,
                            "k": size,
                        }
                    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(jobs)
    audit_path = path.with_name(path.stem + "_split_audit.csv")
    pd.DataFrame(audit_registry(registry)).to_csv(audit_path, index=False)
    return jobs


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    for job in jobs:
        job["index"], job["run"], job["k"] = int(job["index"]), int(job["run"]), int(job["k"])
    return jobs


def _load_embedding_maps(
    stage2: Path, registry: dict[str, Any], preprocess: str, embedding: str
) -> dict[str, dict[str, np.ndarray]]:
    features = {}
    with Stage2Loader(stage2) as loader:
        for tube in registry["tubes"]:
            ids, values = loader.embeddings(
                registry["dataset"], registry["cell_count"], tube, preprocess, embedding
            )
            features[tube] = dict(zip(ids, values))
    return features


def _aligned_probabilities(model: Any, features: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(features)
    output = np.zeros((len(features), n_classes))
    for column, label in enumerate(model.classes_):
        output[:, int(label)] = raw[:, column]
    return output


def _run_classical(
    model_name: str,
    aggregation: str,
    tube: str,
    feature_maps: dict[str, dict[str, np.ndarray]],
    train_ids: list[str],
    test_ids: list[str],
    labels: dict[str, int],
    n_classes: int,
    seed: int,
) -> np.ndarray:
    estimator = make_classical_classifier(model_name, seed)
    train_targets = np.asarray([labels[patient] for patient in train_ids])
    if aggregation == "single":
        train = np.stack([feature_maps[tube][patient] for patient in train_ids])
        test = np.stack([feature_maps[tube][patient] for patient in test_ids])
        fitted = estimator.fit(train, train_targets)
        return _aligned_probabilities(fitted, test, n_classes)
    if aggregation in {"early_mean", "early_zero"}:
        train_maps = {
            name: {patient: value for patient, value in values.items() if patient in train_ids}
            for name, values in feature_maps.items()
        }
        fusion = EarlyTubeFusion(
            "mean" if aggregation == "early_mean" else "zero", add_indicators=True
        ).fit(train_maps)
        fitted = estimator.fit(fusion.transform(feature_maps, train_ids), train_targets)
        return _aligned_probabilities(fitted, fusion.transform(feature_maps, test_ids), n_classes)
    fusion = LateTubeFusion(estimator, "classification", "soft", random_state=seed).fit(
        feature_maps, {patient: labels[patient] for patient in train_ids}
    )
    raw = fusion.predict_proba(feature_maps, test_ids)
    output = np.zeros((len(test_ids), n_classes))
    for column, label in enumerate(fusion.classes_):
        output[:, int(label)] = raw[:, column]
    return output


def _load_cell_samples(
    stage2: Path,
    registry: dict[str, Any],
    tube: str,
    preprocess: str,
    patient_ids: Sequence[str],
) -> list[np.ndarray]:
    with Stage2Loader(stage2) as loader:
        return [
            loader.cells(
                registry["dataset"], registry["cell_count"], tube, patient, preprocess
            )
            for patient in patient_ids
        ]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _deep_probabilities(
    model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device
) -> np.ndarray:
    model.eval()
    outputs = []
    for batch in loader:
        logits = model(batch["cells"].to(device), batch["mask"].to(device))
        outputs.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(outputs)


def _run_cell_model(
    model_name: str,
    train_samples: list[np.ndarray],
    test_samples: list[np.ndarray],
    train_labels: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    max_cells: int,
    device_name: str | None,
) -> np.ndarray:
    if model_name == "flowsom":
        model = FlowSOMClassifier(
            random_state=seed,
            som_iterations=max(500, epochs * 100),
            max_training_cells=max_cells * len(train_samples),
            max_cells_per_sample=max_cells,
        ).fit(train_samples, train_labels)
        raw = model.predict_proba(test_samples)
        output = np.zeros((len(test_samples), n_classes))
        for column, label in enumerate(model.classifier.classes_):
            output[:, int(label)] = raw[:, column]
        return output
    _seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    training = InMemoryCytometryDataset(
        train_samples,
        train_labels.tolist(),
        max_cells=max_cells,
        random_subsample=True,
        seed=seed,
    )
    testing = InMemoryCytometryDataset(
        test_samples, [0] * len(test_samples), max_cells=max_cells, seed=seed
    )
    train_loader = DataLoader(
        training, batch_size=min(batch_size, len(training)), shuffle=True, collate_fn=cytometry_collate
    )
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False, collate_fn=cytometry_collate)
    sampled = [training[index]["cells"].numpy() for index in range(len(training))]
    pooled = np.concatenate(sampled)
    mean = torch.as_tensor(pooled.mean(axis=0), dtype=torch.float32, device=device)
    std = torch.as_tensor(pooled.std(axis=0).clip(1e-6), dtype=torch.float32, device=device)
    base = BASELINE_REGISTRY[model_name](train_samples[0].shape[1], n_classes).to(device)

    class Normalized(nn.Module):
        def __init__(
            self, network: nn.Module, channel_mean: torch.Tensor, channel_std: torch.Tensor
        ) -> None:
            super().__init__()
            self.network = network
            self.register_buffer("channel_mean", channel_mean)
            self.register_buffer("channel_std", channel_std)

        def forward(self, cells: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.network(
                (cells - self.channel_mean) / self.channel_std, mask
            )

    model = Normalized(base, mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["cells"].to(device), batch["mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
    return _deep_probabilities(model, test_loader, device)


def run_job(
    stage2: str | Path,
    registry_path: str | Path,
    jobs_path: str | Path,
    output_dir: str | Path,
    preprocess: str,
    embedding: str,
    index: int,
    epochs: int = 50,
    batch_size: int = 8,
    max_cells: int = 2048,
    device: str | None = None,
    resume: bool = True,
) -> Path:
    registry = load_registry(registry_path)
    jobs = load_jobs(jobs_path)
    if not 0 <= index < len(jobs):
        raise IndexError(f"Job index {index} is outside [0, {len(jobs) - 1}]")
    job = jobs[index]
    output = Path(output_dir) / "shards" / f"{job['job_id']}.json"
    if resume and output.exists():
        completed = json.loads(output.read_text(encoding="utf-8"))
        if completed.get("registry_hash") != registry["registry_hash"]:
            raise ValueError(
                f"Existing shard {output} belongs to a different split registry"
            )
        if any(completed.get(field) != job[field] for field in JOB_FIELDS):
            raise ValueError(f"Existing shard {output} does not match job index {index}")
        return output
    split = registry["splits"][job["run"]]
    train_ids = split["train_ids_by_k"][str(job["k"])]
    test_ids = split["test_ids"]
    labels = {patient: int(label) for patient, label in registry["labels"].items()}
    test_labels = np.asarray([labels[patient] for patient in test_ids])
    seed_material = f"{registry['seed']}|{job['job_id']}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    started = time.perf_counter()
    if job["model"] in CLASSICAL_MODELS:
        maps = _load_embedding_maps(Path(stage2), registry, preprocess, embedding)
        probabilities = _run_classical(
            job["model"],
            job["aggregation"],
            job["tube"],
            maps,
            train_ids,
            test_ids,
            labels,
            len(registry["class_names"]),
            seed,
        )
    else:
        train_samples = _load_cell_samples(
            Path(stage2), registry, job["tube"], preprocess, train_ids
        )
        test_samples = _load_cell_samples(
            Path(stage2), registry, job["tube"], preprocess, test_ids
        )
        probabilities = _run_cell_model(
            job["model"],
            train_samples,
            test_samples,
            np.asarray([labels[patient] for patient in train_ids]),
            len(registry["class_names"]),
            seed,
            epochs,
            batch_size,
            max_cells,
            device,
        )
    metrics = classification_metrics(test_labels, probabilities)
    result = {
        **job,
        "dataset": registry["dataset"],
        "cell_count": registry["cell_count"],
        "preprocess": preprocess,
        "embedding": embedding,
        "registry_hash": registry["registry_hash"],
        "seed": seed,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_ids_hash": _canonical_hash(sorted(train_ids))[:16],
        "test_ids_hash": _canonical_hash(sorted(test_ids))[:16],
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "test_ids": test_ids,
        "y_true": test_labels.tolist(),
        "y_pred": probabilities.argmax(axis=1).tolist(),
        "probabilities": probabilities.tolist(),
    }
    _atomic_json(result, output)
    return output


def aggregate_results(
    registry_path: str | Path,
    jobs_path: str | Path,
    shards_dir: str | Path,
    output_dir: str | Path,
    allow_incomplete: bool = False,
) -> dict[str, object]:
    registry = load_registry(registry_path)
    jobs = load_jobs(jobs_path)
    expected = {job["job_id"] for job in jobs}
    records = []
    duplicates: set[str] = set()
    observed: set[str] = set()
    for path in sorted(Path(shards_dir).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["registry_hash"] != registry["registry_hash"]:
            raise ValueError(f"Shard {path} was generated from a different split registry")
        if result["job_id"] in observed:
            duplicates.add(result["job_id"])
        observed.add(result["job_id"])
        records.append({**{key: value for key, value in result.items() if key != "metrics"}, **result["metrics"]})
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if duplicates or unexpected or (missing and not allow_incomplete):
        raise ValueError(
            f"Result integrity failure: missing={len(missing)}, unexpected={unexpected}, "
            f"duplicates={sorted(duplicates)}"
        )
    if not records:
        raise ValueError("No result shards found")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scalar_columns = [
        "dataset", "model", "aggregation", "tube", "run", "k", "n_train", "n_test",
        "train_ids_hash", "test_ids_hash", "elapsed_seconds", "accuracy", "balanced_accuracy",
        "macro_f1", "roc_auc", "pr_auc",
    ]
    frame = pd.DataFrame(records)
    frame[scalar_columns].sort_values(
        ["model", "aggregation", "tube", "run", "k"]
    ).to_csv(output / "per_run.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc", "elapsed_seconds"]
    group_columns = ["dataset", "model", "aggregation", "tube", "k"]
    summary = frame.groupby(group_columns, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    for size, table in summary.groupby("k"):
        latex_rows = []
        for _, row in table.iterrows():
            entry: dict[str, object] = {
                "model": f"{row['model']} / {row['aggregation']} / {row['tube']}"
            }
            for metric in metrics[:-1]:
                entry[f"{metric}_mean"] = row[f"{metric}_mean"]
                entry[f"{metric}_std"] = row[f"{metric}_std"]
            latex_rows.append(entry)
        export_latex(
            latex_rows,
            output / f"comparison_k{size}.tex",
            caption=f"Repeated classification with {size} training patients per class.",
            label=f"tab:flowlot-k{size}",
        )
    paired_rows = []
    frame["method"] = frame[["model", "aggregation", "tube"]].agg(" / ".join, axis=1)
    for size, subset in frame.groupby("k"):
        by_method = {
            method: values.set_index("run") for method, values in subset.groupby("method")
        }
        for left, right in itertools.combinations(sorted(by_method), 2):
            common = by_method[left].index.intersection(by_method[right].index)
            for metric in ("accuracy", "macro_f1"):
                delta = by_method[left].loc[common, metric] - by_method[right].loc[common, metric]
                paired_rows.append(
                    {
                        "k": size,
                        "metric": metric,
                        "method_a": left,
                        "method_b": right,
                        "n_pairs": len(delta),
                        "mean_delta_a_minus_b": delta.mean(),
                        "std_delta": delta.std(ddof=1),
                        "a_win_fraction": (delta > 0).mean(),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(output / "paired_comparisons.csv", index=False)
    apply_nature_style()
    figure_frame = frame.copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for axis, metric, title in zip(
        axes, ["balanced_accuracy", "macro_f1"], ["Balanced accuracy", "Macro F1"]
    ):
        sns.lineplot(
            data=figure_frame,
            x="k",
            y=metric,
            hue="method",
            estimator="mean",
            errorbar="sd",
            marker="o",
            ax=axis,
        )
        axis.set(xlabel="Training patients per class", ylabel=title, ylim=(0, 1.02))
        axis.legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, output / "aggregation_comparison")
    plt.close(fig)
    integrity = {
        "expected_jobs": len(expected),
        "completed_jobs": len(observed & expected),
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "registry_hash": registry["registry_hash"],
    }
    _atomic_json(integrity, output / "integrity.json")
    return integrity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowlot-repeat", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    splits = commands.add_parser("splits", help="Create the immutable shared split registry")
    splits.add_argument("--stage2", type=Path, required=True)
    splits.add_argument("--dataset", required=True)
    splits.add_argument("--cells", default="all")
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--train-per-class", default="2,4,6,8")
    splits.add_argument("--repeats", type=int, default=10)
    splits.add_argument("--test-size", type=float, default=0.5)
    splits.add_argument("--seed", type=int, default=42)
    splits.add_argument("--tubes", help="Comma-separated; default is every tube")
    splits.add_argument("--patient-policy", choices=["intersection", "union"], default="intersection")
    splits.add_argument(
        "--legacy-h5", type=Path, help="Also export the historical HDF5 split layout"
    )
    jobs = commands.add_parser("jobs", help="Expand splits × models into a TSV job array")
    jobs.add_argument("--splits", type=Path, required=True)
    jobs.add_argument("--output", type=Path, required=True)
    jobs.add_argument(
        "--models",
        default="logistic,linear_svm,random_forest,extra_trees,nsc,nsc_energy,flowsom,cellcnn,attention_mil,cytoset,dgcnn,pointnet2",
    )
    jobs.add_argument("--aggregations", default="single,early_mean,late_soft")
    run = commands.add_parser("run", help="Execute one job-table row")
    run.add_argument("--stage2", type=Path, required=True)
    run.add_argument("--splits", type=Path, required=True)
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--preprocess", required=True)
    run.add_argument("--embedding", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--epochs", type=int, default=50)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-cells", type=int, default=2048)
    run.add_argument("--device")
    run.add_argument("--no-resume", action="store_true")
    aggregate = commands.add_parser("aggregate", help="Validate and combine result shards")
    aggregate.add_argument("--splits", type=Path, required=True)
    aggregate.add_argument("--jobs", type=Path, required=True)
    aggregate.add_argument("--shards", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "splits":
        result = create_split_registry(
            args.stage2,
            args.dataset,
            args.cells,
            args.output,
            _parse_csv(args.train_per_class, int),
            args.repeats,
            args.test_size,
            args.seed,
            _parse_csv(args.tubes) if args.tubes else None,
            args.patient_policy,
        )
        if args.legacy_h5:
            export_legacy_splits_h5(result, args.legacy_h5)
        print(json.dumps({"output": str(args.output), "registry_hash": result["registry_hash"]}, indent=2))
    elif args.command == "jobs":
        result = create_job_table(
            args.splits, args.output, _parse_csv(args.models), _parse_csv(args.aggregations)
        )
        print(json.dumps({"output": str(args.output), "jobs": len(result)}, indent=2))
    elif args.command == "run":
        output = run_job(
            args.stage2,
            args.splits,
            args.jobs,
            args.output_dir,
            args.preprocess,
            args.embedding,
            args.index,
            args.epochs,
            args.batch_size,
            args.max_cells,
            args.device,
            not args.no_resume,
        )
        print(json.dumps({"output": str(output), "index": args.index}, indent=2))
    else:
        integrity = aggregate_results(
            args.splits, args.jobs, args.shards, args.output, args.allow_incomplete
        )
        print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
