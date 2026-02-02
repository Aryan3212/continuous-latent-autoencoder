from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PredictorConfig:
    kind: str = "mlp"  # mlp | transformer
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1


class MLPPredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = dim
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,d,T')
        x = z.transpose(1, 2)  # (B,T',d)
        x = self.net(x)
        return x.transpose(1, 2)


class Predictor(nn.Module):
    def __init__(self, dim: int, cfg: PredictorConfig):
        super().__init__()
        if cfg.kind == "mlp":
            self.model = MLPPredictor(dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        else:
            raise ValueError(f"Unsupported predictor kind (Exp0): {cfg.kind}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

