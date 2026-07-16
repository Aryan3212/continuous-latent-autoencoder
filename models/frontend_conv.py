from __future__ import annotations

import math

import torch
import torch.nn as nn

from schema import FrontendCfg


class ConvFrontend(nn.Module):
    """Strided convolutional waveform frontend."""

    def __init__(self, cfg: FrontendCfg):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in zip(cfg.channels, cfg.kernels, cfg.strides):
            pad = k // 2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad))
            gn_groups = math.gcd(cfg.groups, out_ch)
            layers.append(nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = cfg.channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
