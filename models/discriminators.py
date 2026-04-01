from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, channels: int = 32):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(channels, channels * 2, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(channels * 2, channels * 4, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(channels * 4, channels * 8, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(channels * 8, channels * 8, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.out_conv = weight_norm(nn.Conv2d(channels * 8, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.out_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, channels, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(channels, channels * 4, 41, 4, padding=20, groups=4)),
                weight_norm(nn.Conv1d(channels * 4, channels * 16, 41, 4, padding=20, groups=16)),
                weight_norm(nn.Conv1d(channels * 16, channels * 16, 41, 4, padding=20, groups=16)),
                weight_norm(nn.Conv1d(channels * 16, channels * 16, 5, 1, padding=2)),
            ]
        )
        self.out_conv = weight_norm(nn.Conv1d(channels * 16, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.out_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int], channels: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p, channels=channels) for p in periods])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits = []
        fmaps = []
        for d in self.discriminators:
            logit, fmap = d(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 3, channels: int = 16):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(channels=channels) for _ in range(scales)])
        self.pool = nn.AvgPool1d(4, 2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits = []
        fmaps = []
        cur = x
        for i, d in enumerate(self.discriminators):
            logit, fmap = d(cur)
            logits.append(logit)
            fmaps.append(fmap)
            if i < len(self.discriminators) - 1:
                cur = self.pool(cur)
        return logits, fmaps


def discriminator_loss(real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + (F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean())
    n = max(len(real_logits), 1)
    return loss / n


def generator_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for f in fake_logits:
        loss = loss + (-f.float().mean())
    n = max(len(fake_logits), 1)
    return loss / n


def feature_matching_loss(real_fmaps: List[List[torch.Tensor]], fake_fmaps: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    total_layers = 0
    for r_layers, f_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(r_layers, f_layers):
            loss = loss + (r.float().detach() - f.float()).abs().mean()
            total_layers += 1
    if total_layers > 0:
        loss = loss / total_layers
    return loss
