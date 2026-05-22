from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    channels: int = 256
    up_strides: List[int] = None
    up_kernels: List[int] = None
    res_blocks_per_up: int = 2
    res_dilations: List[int] = None
    film_hidden: int = 128


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, target_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cond_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * target_channels, kernel_size=1),
        )

    def forward(self, cond: torch.Tensor, length: int) -> torch.Tensor:
        # cond: (B, d, Tcond) -> (B, 2C, length)
        if cond.size(-1) != length:
            cond = F.interpolate(cond, size=length, mode="linear", align_corners=False)
        return self.net(cond)


class ResBlockFiLM(nn.Module):
    def __init__(self, channels: int, dilations: List[int], cond_dim: int, film_hidden: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d)
            )
        self.film = FiLM(cond_dim, channels, hidden=film_hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        g_b = self.film(cond, x.size(-1))
        gamma, beta = g_b.chunk(2, dim=1)
        h = x
        for conv in self.convs:
            h = conv(h)
            h = (1.0 + gamma) * h + beta
            h = F.gelu(h)
        return x + h


class WaveformDecoder(nn.Module):
    """
    Simple neural vocoder-like generator (no GAN in Exp0).
      z: (B,d,T') -> x_hat: (B,1,T)
    """

    def __init__(self, latent_dim: int, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        up_strides = cfg.up_strides or [4, 4, 4, 4, 5]
        up_kernels = cfg.up_kernels or [8, 8, 8, 8, 10]
        res_dilations = cfg.res_dilations or [1, 3, 9]
        assert len(up_strides) == len(up_kernels)

        self.in_conv = nn.Conv1d(latent_dim, cfg.channels, kernel_size=3, padding=1)

        ups: List[nn.Module] = []
        resblocks: List[nn.Module] = []
        in_ch = cfg.channels
        for s, k in zip(up_strides, up_kernels):
            ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=float(s), mode="linear", align_corners=False),
                    nn.Conv1d(in_ch, in_ch // 2, kernel_size=k, padding=k // 2),
                )
            )
            out_ch = in_ch // 2
            for _ in range(cfg.res_blocks_per_up):
                resblocks.append(
                    ResBlockFiLM(
                        channels=out_ch,
                        dilations=res_dilations,
                        cond_dim=latent_dim,
                        film_hidden=cfg.film_hidden,
                    )
                )
            in_ch = out_ch

        self.ups = nn.ModuleList(ups)
        self.resblocks = nn.ModuleList(resblocks)
        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)
        self.up_strides = up_strides
        self.res_blocks_per_up = cfg.res_blocks_per_up

    def forward(self, z: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        x = self.in_conv(z)
        rb_i = 0
        for up in self.ups:
            x = F.gelu(up(x))
            for _ in range(self.res_blocks_per_up):
                x = self.resblocks[rb_i](x, z)
                rb_i += 1
        x = torch.tanh(self.out_conv(x))
        if target_len is not None and x.size(-1) != target_len:
            if x.size(-1) > target_len:
                x = x[..., :target_len]
            else:
                x = F.pad(x, (0, target_len - x.size(-1)))
        return x
