from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils.schema import FrontendCfg


class ConvFrontend(nn.Module):
    """
    Strided Conv1D stack:
      x: (B, 1, T) -> h0: (B, C, T')
    """

    def __init__(self, cfg: FrontendCfg):
        super().__init__()
        assert len(cfg.channels) == len(cfg.kernels) == len(cfg.strides)
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in zip(cfg.channels, cfg.kernels, cfg.strides):
            pad = k // 2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad))
            gn_groups = min(cfg.groups, out_ch)
            layers.append(nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = cfg.channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected x as (B,1,T), got {tuple(x.shape)}")
        return self.net(x)

