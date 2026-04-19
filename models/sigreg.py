from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import distributed as dist


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, dist.ReduceOp.AVG)
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
        self._step_int: int = 0
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
            seed = self._step_int + 137
            gen = self._get_generator(x.device, seed)
            a = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=gen)
            a /= a.norm(p=2, dim=0)
            self._step_int += 1
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
    var_weight: float = 1.0
    residual_weight: float = 0.2


class SIGReg(nn.Module):
    """
    LeJEPA SIGReg: CF/ECF matching with random projections (Algorithm 1).
    """

    def __init__(self, dim: int, cfg: SIGRegConfig):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.var_weight = cfg.var_weight
        self.residual_weight = cfg.residual_weight
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

        # Decomposed SIGReg: split z into utterance mean and per-frame residuals.
        m = z.mean(dim=-1)  # (B, D)
        r = z - m.unsqueeze(-1)  # (B, D, T)

        # Utterance-level Gaussianity test on pooled means.
        ecf_loss_utt = self.test(m.unsqueeze(0))  # (1, B, D)

        # Residual-level Gaussianity test on flattened residuals.
        r_flat = r.permute(0, 2, 1).reshape(b * t, d).unsqueeze(0)  # (1, B*T, D)
        ecf_loss_res = self.test(r_flat)

        # Variance penalty on utterance means only.
        var = m.var(dim=0, unbiased=False)  # (D,)
        l_var = (var - 1.0).pow(2).mean()

        loss = ecf_loss_utt + self.residual_weight * ecf_loss_res + self.var_weight * l_var

        var_res = r_flat.var(dim=1, unbiased=False).squeeze(0)  # (D,)
        stats = {
            "sigreg_loss": loss.detach(),
            "z_var_min": var.min().detach(),
            "z_var_med": var.median().detach(),
            "z_var_max": var.max().detach(),
            "z_var_penalty": l_var.detach(),
            "z_var_min_res": var_res.min().detach(),
            "z_var_med_res": var_res.median().detach(),
            "z_var_max_res": var_res.max().detach(),
        }
        return loss, stats
