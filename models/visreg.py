from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema import VISRegCfg


class VISReg(nn.Module):
    """VISReg (Vector-ISotropic Gaussianisation) from
    https://haiyuwu.github.io/visreg/. Paper-faithful implementation.

    Enforces each batch of points to be an isotropic N(0, I) distribution via
    three terms (computed over the ``B`` dimension of the input):

      - center_loss: pulls the per-sample mean to 0 (mu.pow(2).mean())
      - scale_loss:  pulls per-point std to 1 ((std - 1).pow(2).mean())
      - shape_loss:  matches the sorted random projections to the quantiles of
                     a standard normal (computed once per B, cached).

    Input: ``z`` of shape ``(N, B, D)`` — ``N`` independent samples, each a
    population of ``B`` points in ``D`` dims. Output: a single scalar loss.

    There are no learnable parameters; the projection matrix ``W`` is resampled
    (unseeded, as in the paper) on every forward pass, so each call is an
    independent Monte-Carlo estimate of the Gaussianisation objective. This is
    fine under DDP (each rank gets its own projection, and the averaged
    gradient remains an unbiased estimate).
    """

    def __init__(self, cfg: VISRegCfg):
        super().__init__()
        self.K = cfg.num_projections
        self._cached_B = -1
        self._cached_target = None

    def _get_target(self, B: int, device, dtype) -> torch.Tensor:
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)) + 1e-6
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device, z.dtype).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return scale_loss + shape_loss + center_loss
