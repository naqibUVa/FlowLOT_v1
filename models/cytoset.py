"""CytoSet-inspired permutation-invariant cytometry classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import check_bag, make_mlp, masked_max, masked_mean


class CytoSet(nn.Module):
    """Cell encoder followed by mean/max set statistics and a set decoder.

    Cell sub-sampling is deliberately handled by the dataset so the same policy is
    shared by every architecture and can be varied each training epoch.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        set_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cell_encoder = make_mlp([in_features, hidden_dim, set_dim], dropout, True)
        self.classifier = make_mlp([2 * set_dim, set_dim, num_classes], dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = check_bag(x, mask)
        encoded = self.cell_encoder(x)
        summary = torch.cat([masked_mean(encoded, mask), masked_max(encoded, mask)], dim=-1)
        return self.classifier(summary)
