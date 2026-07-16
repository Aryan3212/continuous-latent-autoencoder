"""HiFi-GAN multi-period discriminator over reconstruction spectrograms."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1


def _same_pad(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


class DiscriminatorP(nn.Module):
    """Single-period discriminator over ``(B, F, T)`` spectrograms."""

    def __init__(
        self,
        period: int,
        channels: Sequence[int] = (32, 128, 512, 1024),
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period
        chans = [in_channels, *channels]
        self.convs = nn.ModuleList(
            weight_norm(
                nn.Conv2d(
                    chans[i],
                    chans[i + 1],
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(_same_pad(kernel_size), 0),
                )
            )
            for i in range(len(chans) - 1)
        )
        last = chans[-1]
        self.convs.append(
            weight_norm(
                nn.Conv2d(last, last, (kernel_size, 1), 1, padding=(_same_pad(kernel_size), 0))
            )
        )
        self.conv_post = weight_norm(nn.Conv2d(last, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        logits = torch.flatten(x, 1, -1)
        return logits, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Bank of per-period sub-discriminators (HiFi-GAN MPD)."""

    def __init__(
        self,
        periods: Sequence[int],
        channels: Sequence[int] = (32, 128, 512, 1024),
        in_channels: int = 1,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            DiscriminatorP(p, channels=channels, in_channels=in_channels) for p in periods
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        d_real, d_fake, fmap_real, fmap_fake = [], [], [], []
        for d in self.discriminators:
            r, fr = d(y)
            g, fg = d(y_hat)
            d_real.append(r)
            d_fake.append(g)
            fmap_real.append(fr)
            fmap_fake.append(fg)
        return d_real, d_fake, fmap_real, fmap_fake
