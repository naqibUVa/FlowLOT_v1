"""Mask-aware tensor operations shared by set models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def check_bag(x: Tensor, mask: Tensor | None) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, cells, features], got {tuple(x.shape)}")
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.shape != x.shape[:2]:
        raise ValueError("mask must have shape [batch, cells]")
    if (~mask.any(dim=1)).any():
        raise ValueError("Every bag must contain at least one valid cell")
    return mask


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_max(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    return x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=dim)


def make_mlp(
    sizes: list[int], dropout: float = 0.0, final_activation: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (source, target) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(source, target))
        is_last = i == len(sizes) - 2
        if not is_last or final_activation:
            layers.extend([nn.LayerNorm(target), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
