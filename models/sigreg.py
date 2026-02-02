from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import distributed as dist


def _all_reduce(x: torch.Tensor, op: str = "AVG") -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        if op == "MAX":
            return torch.distributed.nn.all_reduce(x, dist.ReduceOp.MAX)
        return torch.distributed.nn.all_reduce(x, dist.ReduceOp.AVG)
    return x


class EppsPulley(nn.Module):
    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd")
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)
        cos_mean = _all_reduce(cos_vals.mean(-3))
        sin_mean = _all_reduce(sin_vals.mean(-3))
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n


class SlicingUnivariateTest(nn.Module):
    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: str = "mean",
        clip_value: Optional[float] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            global_step_sync = _all_reduce(self.global_step.clone(), op="MAX")
            seed = int(global_step_sync.item())
            g = self._get_generator(x.device, seed)
            a = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            a /= a.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ a)
        if self.clip_value is not None:
            stats = stats.clone()
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        if self.reduction is None:
            return stats
        raise ValueError(f"Unknown reduction: {self.reduction}")


@dataclass
class SIGRegConfig:
    num_slices: int = 256
    t_max: float = 3.0
    n_points: int = 17
    reduction: str = "mean"
    clip_value: Optional[float] = None


class SIGReg(nn.Module):
    """
    LeJEPA SIGReg: CF/ECF matching with random projections (Algorithm 1).
    """

    def __init__(self, dim: int, cfg: SIGRegConfig):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        univariate = EppsPulley(t_max=cfg.t_max, n_points=cfg.n_points)
        self.test = SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=cfg.num_slices,
            reduction=cfg.reduction,
            clip_value=cfg.clip_value,
        )

    def forward(self, z: torch.Tensor, step: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z.dim() != 3:
            raise ValueError(f"Expected z as (B,d,T'), got {tuple(z.shape)}")
        b, d, t = z.shape
        x = z.permute(0, 2, 1).reshape(b * t, d).unsqueeze(0)  # (1,N,D)
        loss = self.test(x)

        var = x.var(dim=0, unbiased=False)
        stats = {
            "sigreg_loss": loss.detach(),
            "z_var_min": var.min().detach(),
            "z_var_med": var.median().detach(),
            "z_var_max": var.max().detach(),
        }
        return loss, stats
