"""Gated attention multiple-instance learning for cytometry bags."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import check_bag, make_mlp


class AttentionMIL(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = make_mlp([in_features, hidden_dim, hidden_dim], dropout, True)
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        mask = check_bag(x, mask)
        embeddings = self.encoder(x)
        scores = self.attention_w(
            torch.tanh(self.attention_v(embeddings)) * torch.sigmoid(self.attention_u(embeddings))
        ).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=1)
        bag = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        logits = self.classifier(bag)
        return (logits, weights) if return_attention else logits
