from __future__ import annotations

import torch
import torch.nn as nn

from utils.schema import SIGRegCfg


class EppsPulley(nn.Module):
    """Algorithm 1 univariate test, paper-faithful (symmetric grid)."""

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
        n_global = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(-3)
        sin_mean = torch.sin(x_t).mean(-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n_global


class SlicingUnivariateTest(nn.Module):
    """Algorithm 1 slicing wrapper. Seeded by the training step for
    resume-reproducibility (no hidden internal counter)."""

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


class SIGReg(nn.Module):
    """Paper-faithful SIGReg (Algorithm 1). Input: (N, D). Output: scalar."""

    def __init__(self, dim: int, cfg: SIGRegCfg):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        univariate = EppsPulley(t_max=cfg.t_max, n_points=cfg.n_points)
        self.test = SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=cfg.num_slices,
        )

    def forward(self, z: torch.Tensor, step: int) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Expected z as (N, D), got {tuple(z.shape)}")

        return self.test(z, step=step)
