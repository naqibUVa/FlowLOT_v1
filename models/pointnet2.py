"""PointNet++ set abstraction adapted to D-dimensional marker space."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import check_bag, make_mlp, masked_max, masked_mean


def _fps(points: Tensor, count: int) -> Tensor:
    """Deterministic farthest-point sampling on one ``[N, D]`` set."""

    count = min(count, points.shape[0])
    chosen = torch.zeros(count, dtype=torch.long, device=points.device)
    distances = torch.full((points.shape[0],), torch.inf, device=points.device)
    farthest = torch.tensor(0, device=points.device)
    for i in range(count):
        chosen[i] = farthest
        squared = ((points - points[farthest]) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, squared)
        farthest = distances.argmax()
    return chosen


class SetAbstraction(nn.Module):
    def __init__(
        self,
        coordinate_dim: int,
        feature_dim: int,
        out_dim: int,
        npoint: int,
        neighbours: int,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.neighbours = neighbours
        self.out_dim = out_dim
        input_dim = coordinate_dim + feature_dim
        self.local_mlp = make_mlp([input_dim, out_dim // 2, out_dim], final_activation=True)

    def forward(
        self, coordinates: Tensor, features: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, coordinate_dim = coordinates.shape
        new_coordinates = coordinates.new_zeros(batch, self.npoint, coordinate_dim)
        new_features = features.new_zeros(batch, self.npoint, self.out_dim)
        new_mask = torch.zeros(batch, self.npoint, dtype=torch.bool, device=mask.device)
        for b in range(batch):
            valid = mask[b].nonzero(as_tuple=False).squeeze(1)
            xyz = coordinates[b, valid]
            values = features[b, valid]
            centres_idx = _fps(xyz, self.npoint)
            centres = xyz[centres_idx]
            m = centres.shape[0]
            k = min(self.neighbours, xyz.shape[0])
            neighbour_idx = torch.cdist(centres, xyz).topk(k, largest=False).indices
            relative = xyz[neighbour_idx] - centres.unsqueeze(1)
            local = torch.cat([relative, values[neighbour_idx]], dim=-1)
            aggregated = self.local_mlp(local).amax(dim=1)
            new_coordinates[b, :m] = centres
            new_features[b, :m] = aggregated
            new_mask[b, :m] = True
        return new_coordinates, new_features, new_mask


class PointNet2(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        npoint1: int = 128,
        npoint2: int = 32,
        neighbours: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(
            in_features, in_features, hidden_dim, npoint1, neighbours
        )
        self.sa2 = SetAbstraction(
            in_features, hidden_dim, 2 * hidden_dim, npoint2, neighbours
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = check_bag(x, mask)
        coordinates, features, mask = self.sa1(x, x, mask)
        _, features, mask = self.sa2(coordinates, features, mask)
        summary = torch.cat([masked_mean(features, mask), masked_max(features, mask)], dim=-1)
        return self.classifier(summary)
