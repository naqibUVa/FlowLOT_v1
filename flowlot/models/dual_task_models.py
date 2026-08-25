"""Task-switchable neural heads for LOT or early-fusion vectors."""

from __future__ import annotations

from torch import Tensor, nn


class LOTMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        task: str = "classification",
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        layers: list[nn.Module] = []
        source = in_features
        for target in hidden_dims:
            layers.extend(
                [nn.Linear(source, target), nn.LayerNorm(target), nn.GELU(), nn.Dropout(dropout)]
            )
            source = target
        layers.append(nn.Linear(source, num_classes if task == "classification" else 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        output = self.network(features)
        return output.squeeze(-1) if self.task == "regression" else output
