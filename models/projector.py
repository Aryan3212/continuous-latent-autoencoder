from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils.schema import ProjectorCfg


class Projector(nn.Module):
    """Per-frame MLP projector with BatchNorm1d.

    Used to decouple loss-space (JEPA + SIGReg) from representation-space
    (decoder + downstream probes). Mirrors LeJEPA's
    `MLP(in, [hidden, hidden, out], norm_layer=BatchNorm1d)` recipe and
    LeWM's projector. BatchNorm is required because LayerNorm-style encoder
    outputs (LayerNorm-normalised) resist being reshaped to N(0, I).

    Input:  (B, D, T)   encoder output
    Output: (B, P, T)   projected output
    """

    def __init__(self, dim: int, cfg: ProjectorCfg):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(max(0, cfg.n_hidden_layers)):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.BatchNorm1d(cfg.hidden_dim))
            layers.append(nn.GELU())
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = cfg.output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected (B, D, T), got {tuple(z.shape)}")
        B, D, T = z.shape
        x = z.transpose(1, 2).reshape(B * T, D)
        x = self.net(x)
        return x.view(B, T, -1).transpose(1, 2).contiguous()
