from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema import DecoderCfg


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, target_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cond_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * target_channels, kernel_size=1),
        )

    def forward(self, cond: torch.Tensor, length: int) -> torch.Tensor:
        if cond.size(-1) != length:
            cond = F.interpolate(cond, size=length, mode="linear", align_corners=False)
        return self.net(cond)


class ResBlockFiLM(nn.Module):
    def __init__(self, channels: int, dilations: list[int], cond_dim: int, film_hidden: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d)
            )
        self.film = FiLM(cond_dim, channels, hidden=film_hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g_b = self.film(cond, x.size(-1))
        gamma, beta = g_b.chunk(2, dim=1)
        h = x
        for conv in self.convs:
            h = conv(h)
            h = (1.0 + gamma) * h + beta
            h = F.gelu(h)
        return x + h


class _SameLengthConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.kernel_size[0] - 1
        left = self.kernel_size[0] // 2
        return super().forward(F.pad(x, (left, total_pad - left)))


class WaveformDecoder(nn.Module):
    """Upsample continuous latents into a waveform."""

    def __init__(self, latent_dim: int, cfg: DecoderCfg):
        super().__init__()
        up_strides = cfg.up_strides
        up_kernels = cfg.up_kernels
        res_dilations = cfg.res_dilations
        self.in_conv = nn.Conv1d(latent_dim, cfg.channels, kernel_size=3, padding=1)

        ups: list[nn.Module] = []
        resblocks: list[nn.Module] = []
        in_ch = cfg.channels
        for s, k in zip(up_strides, up_kernels):
            ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=float(s), mode="linear", align_corners=False),
                    _SameLengthConv1d(in_ch, in_ch // 2, kernel_size=k),
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
        if target_len is not None:
            x = x[..., :target_len]
        return x
