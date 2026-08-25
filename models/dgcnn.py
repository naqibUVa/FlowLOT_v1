"""Native PyTorch Dynamic Graph CNN for variable-size cell sets."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import check_bag, masked_max, masked_mean


def _knn_gather(x: Tensor, mask: Tensor, k: int, chunk_size: int) -> Tensor:
    """Return ``[B, N, k, C]`` neighbours without materializing B full graphs."""

    batch, n_cells, channels = x.shape
    output = x.new_zeros(batch, n_cells, k, channels)
    for b in range(batch):
        valid_indices = mask[b].nonzero(as_tuple=False).squeeze(1)
        values = x[b, valid_indices]
        actual_k = min(k, values.shape[0])
        parts: list[Tensor] = []
        for start in range(0, values.shape[0], chunk_size):
            distance = torch.cdist(values[start : start + chunk_size], values)
            indices = distance.topk(actual_k, largest=False).indices
            neighbours = values[indices]
            if actual_k < k:
                neighbours = torch.cat(
                    [neighbours, neighbours[:, -1:].expand(-1, k - actual_k, -1)], dim=1
                )
            parts.append(neighbours)
        output[b, valid_indices] = torch.cat(parts, dim=0)
    return output


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, chunk_size: int) -> None:
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size
        self.network = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        neighbours = _knn_gather(x, mask, self.k, self.chunk_size)
        centre = x.unsqueeze(2).expand_as(neighbours)
        edges = torch.cat([centre, neighbours - centre], dim=-1)
        return self.network(edges).amax(dim=2) * mask.unsqueeze(-1)


class DGCNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        k: int = 20,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        knn_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.edge1 = EdgeConv(in_features, hidden_dim, k, knn_chunk_size)
        self.edge2 = EdgeConv(hidden_dim, 2 * hidden_dim, k, knn_chunk_size)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = check_bag(x, mask)
        features = self.edge2(self.edge1(x, mask), mask)
        global_features = torch.cat(
            [masked_mean(features, mask), masked_max(features, mask)], dim=-1
        )
        return self.classifier(global_features)
