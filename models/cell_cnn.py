"""Modern PyTorch implementation of CellCnn-style cell filters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import check_bag


class CellCNN(nn.Module):
    """Learn phenotype filters and aggregate their strongest cell responses.

    A kernel-size-one convolution is the direct analogue of applying each learned
    CellCnn filter independently to every cell. ''topk_fraction'' reproduces the
    original multi-instance top-k pooling; set it to 1 for mean pooling.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_filters: int = 128,
        topk_fraction: float = 0.05,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0 < topk_fraction <= 1:
            raise ValueError("topk_fraction must lie in (0, 1]")
        self.topk_fraction = topk_fraction
        self.filters = nn.Conv1d(in_features, num_filters, kernel_size=1)
        self.activation = nn.LeakyReLU(0.1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_filters, num_classes))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = check_bag(x, mask)
        responses = self.activation(self.filters(x.transpose(1, 2))).transpose(1, 2)
        pooled: list[Tensor] = []
        for values, valid in zip(responses, mask):
            values = values[valid]
            k = max(1, int(round(values.shape[0] * self.topk_fraction)))
            pooled.append(values.topk(k, dim=0).values.mean(dim=0))
        return self.classifier(torch.stack(pooled))
