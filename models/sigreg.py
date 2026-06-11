from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import distributed as dist


def _all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class EppsPulley(nn.Module):
    """Algorithm 1 univariate test, paper-faithful (symmetric grid, DDP-aware N)."""

    def __init__(self, t_max: float = 5.0, n_points: int = 17):
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd")
        t = torch.linspace(-t_max, t_max, n_points, dtype=torch.float32)
        dt = (2 * t_max) / (n_points - 1)
        weights = torch.full((n_points,), dt, dtype=torch.float32)
        weights[0] = dt / 2
        weights[-1] = dt / 2
        phi = torch.exp(-t.square() * 0.5)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N, M)
        n_local = x.size(-2)
        n_global = n_local * _world_size()
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = _all_reduce_avg(torch.cos(x_t).mean(-3))
        sin_mean = _all_reduce_avg(torch.sin(x_t).mean(-3))
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n_global


class SlicingUnivariateTest(nn.Module):
    """Algorithm 1 slicing wrapper. Seeded by the training step for DDP sync
    and resume-reproducibility (no hidden internal counter)."""

    def __init__(self, univariate_test: nn.Module, num_slices: int = 256):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        # x: (N, D)
        with torch.no_grad():
            gen = self._get_generator(x.device, int(step))
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=gen)
            A /= A.norm(p=2, dim=0)

        sliced = x @ A  # (N, M)
        stats = self.univariate_test(sliced)  # (M,)
        return stats.mean()


@dataclass
class SIGRegConfig:
    num_slices: int = 256
    t_max: float = 5.0
    n_points: int = 17


class SIGReg(nn.Module):
    """Paper-faithful SIGReg (Algorithm 1). Input: (N, D). Output: scalar + stats."""

    def __init__(self, dim: int, cfg: SIGRegConfig):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        univariate = EppsPulley(t_max=cfg.t_max, n_points=cfg.n_points)
        self.test = SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=cfg.num_slices,
        )

    def forward(self, z: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z.dim() != 2:
            raise ValueError(f"Expected z as (N, D), got {tuple(z.shape)}")

        loss = self.test(z, step=step)

        with torch.no_grad():
            var = z.var(dim=0, unbiased=False)  # (D,)
            stats = {
                "sigreg_loss": loss.detach(),
                "z_var_min": var.min(),
                "z_var_med": var.median(),
                "z_var_max": var.max(),
            }
        return loss, stats
