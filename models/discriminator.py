"""HiFi-GAN Multi-Period Discriminator (MPD).

Operates on the decoder's raw waveform output. Each sub-discriminator reshapes
the 1-D waveform to a 2-D grid at its period and runs a small 2-D conv stack,
which is much lighter on VRAM than full-length 1-D multi-scale discriminators —
the reason MPD-only was chosen for the 6 GB card. Returns per-sub-discriminator
logits and the intermediate feature maps used by the feature-matching loss.

Reference: Kong et al., "HiFi-GAN" (2020); see
reference-implementations / standard MPD formulation.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1


def _same_pad(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


class DiscriminatorP(nn.Module):
    """Single-period sub-discriminator: (B,1,T) -> 2-D grid -> conv stack.

    `channels` are the hidden widths. The HiFi-GAN default (32,128,512,1024)
    yields a ~41M-param MPD — far too heavy for a 6 GB card next to a ~3M
    generator — so a slimmer width is used in practice (see AdvCfg.disc_channels).
    """

    def __init__(
        self,
        period: int,
        channels: List[int] = (32, 128, 512, 1024),
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period
        chans = [1, *channels]
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
        # final stride-1 block, then 1-channel projection
        self.convs.append(
            weight_norm(
                nn.Conv2d(last, last, (kernel_size, 1), 1, padding=(_same_pad(kernel_size), 0))
            )
        )
        self.conv_post = weight_norm(nn.Conv2d(last, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
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

    def __init__(self, periods: List[int], channels: List[int] = (32, 128, 512, 1024)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            DiscriminatorP(p, channels=channels) for p in periods
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """y / y_hat: (B,1,T) real / generated waveforms.

        Returns (d_real, d_fake, fmap_real, fmap_fake) — lists over
        sub-discriminators (HiFi-GAN signature).
        """
        d_real, d_fake, fmap_real, fmap_fake = [], [], [], []
        for d in self.discriminators:
            r, fr = d(y)
            g, fg = d(y_hat)
            d_real.append(r)
            d_fake.append(g)
            fmap_real.append(fr)
            fmap_fake.append(fg)
        return d_real, d_fake, fmap_real, fmap_fake
