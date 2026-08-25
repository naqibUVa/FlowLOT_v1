"""Unified AMP-enabled training entry point for cytometry set classifiers."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import CytometryDataset, cytometry_collate
from metrics import classification_metrics
from models import MODEL_REGISTRY


@dataclass
class TrainConfig:
    model: str
    in_features: int
    num_classes: int
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    num_workers: int = 0
    amp: bool = True
    seed: int = 0


class StandardizedModel(nn.Module):
    """Persist training-set marker normalization inside the checkpoint."""

    def __init__(self, model: nn.Module, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std.clamp_min(1e-6))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.model((x - self.input_mean) / self.input_std, mask)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: Dataset[dict[str, Any]], config: TrainConfig, shuffle: bool
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=cytometry_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def estimate_normalization(loader: DataLoader[dict[str, Any]]) -> tuple[Tensor, Tensor]:
    total: Tensor | None = None
    square_total: Tensor | None = None
    count = 0
    for batch in loader:
        values = batch["cells"][batch["mask"]].double()
        total = values.sum(0) if total is None else total + values.sum(0)
        square_total = (values**2).sum(0) if square_total is None else square_total + (values**2).sum(0)
        count += len(values)
    if count == 0 or total is None or square_total is None:
        raise ValueError("Cannot normalize an empty training set")
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(0)
    return mean.float(), variance.sqrt().float().clamp_min(1e-6)


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels, probabilities, sample_ids = [], [], []
    for batch in loader:
        logits = model(
            batch["cells"].to(device, non_blocking=True),
            batch["mask"].to(device, non_blocking=True),
        )
        probabilities.append(logits.softmax(-1).cpu().numpy())
        labels.append(batch["labels"].numpy())
        sample_ids.extend(batch["sample_ids"])
    return np.concatenate(labels), np.concatenate(probabilities), sample_ids


def fit_deep_model(
    train_dataset: Dataset[dict[str, Any]],
    validation_dataset: Dataset[dict[str, Any]],
    config: TrainConfig,
    model_kwargs: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    if config.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown deep model {config.model!r}")
    seed_everything(config.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    mean, std = estimate_normalization(_loader(train_dataset, config, shuffle=False))
    base = MODEL_REGISTRY[config.model](
        config.in_features, config.num_classes, **(model_kwargs or {})
    )
    model = StandardizedModel(base, mean, std).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and resolved_device.type == "cuda"
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            cells = batch["cells"].to(resolved_device, non_blocking=True)
            mask = batch["mask"].to(resolved_device, non_blocking=True)
            labels = batch["labels"].to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved_device.type,
                enabled=config.amp and resolved_device.type == "cuda",
            ):
                loss = criterion(model(cells, mask), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        val_labels, val_probabilities, _ = predict(model, validation_loader, resolved_device)
        metrics = classification_metrics(val_labels, val_probabilities)
        row = {"epoch": float(epoch), "train_loss": loss_sum / seen, **metrics}
        history.append(row)
        score = metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_labels, val_probabilities, _ = predict(model, validation_loader, resolved_device)
    return model, history, classification_metrics(val_labels, val_probabilities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoint.pt"))
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


def main() -> None:
    args = parse_args()
    train_data = CytometryDataset.from_manifest(
        args.manifest, "train", max_cells=args.max_cells, random_subsample=True, seed=args.seed
    )
    validation_data = CytometryDataset.from_manifest(
        args.manifest, "validation", max_cells=args.max_cells, seed=args.seed
    )
    example = train_data[0]
    labels = {record.label for record in train_data.records}
    if labels != set(range(len(labels))):
        raise ValueError("Training labels must be contiguous integers starting at zero")
    config = TrainConfig(
        model=args.model,
        in_features=example["cells"].shape[1],
        num_classes=len(labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        seed=args.seed,
    )
    model, history, metrics = fit_deep_model(
        train_data, validation_data, config, device=args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(config), "history": history},
        args.output,
    )
    print(json.dumps({"checkpoint": str(args.output), "validation": metrics}, indent=2))


if __name__ == "__main__":
    main()
