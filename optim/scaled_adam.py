from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class ScaledAdam(Optimizer):
    """
    Zipformer-style ScaledAdam port (icefall/zipformer/optim.py).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3.0e-2,
        clipping_scale: Optional[float] = None,
        betas=(0.9, 0.98),
        scalar_lr_scale: float = 0.1,
        eps: float = 1.0e-8,
        param_min_rms: float = 1.0e-5,
        param_max_rms: float = 3.0,
        scalar_max: float = 10.0,
        size_update_period: int = 4,
        clipping_update_period: int = 100,
    ):
        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            betas=betas,
            scalar_lr_scale=scalar_lr_scale,
            eps=eps,
            param_min_rms=param_min_rms,
            param_max_rms=param_max_rms,
            scalar_max=scalar_max,
            size_update_period=size_update_period,
            clipping_update_period=clipping_update_period,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            clipping_scale = self._get_clipping_scale(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ScaledAdam does not support sparse gradients")
                if clipping_scale != 1.0:
                    grad = grad.mul(clipping_scale)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                delta = self._momentum_step(group, p, state, grad)
                p.add_(delta)
                if p.numel() == 1:
                    scalar_max = group["scalar_max"]
                    p.clamp_(min=-scalar_max, max=scalar_max)
                state["step"] += 1

        return loss

    def _basic_step(self, group, p, state, grad):
        lr = group["lr"]
        if p.numel() == 1:
            lr = lr * group["scalar_lr_scale"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(p, dtype=torch.float)
            state["exp_avg_sq"] = exp_avg_sq

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction2 = 1.0 - beta2 ** (state["step"] + 1)
        if bias_correction2 < 0.99:
            exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
        denom = exp_avg_sq.sqrt().add_(eps)
        return -lr * grad / denom

    def _scaling_step(self, group, p, state, grad):
        delta = self._basic_step(group, p, state, grad)
        if p.numel() == 1:
            return delta

        step = state["step"]
        size_update_period = group["size_update_period"]

        param_rms = state.get("param_rms")
        scale_grads = state.get("scale_grads")
        scale_exp_avg_sq = state.get("scale_exp_avg_sq")

        if param_rms is None:
            param_rms = (p**2).mean(dim=list(range(p.ndim)), keepdim=True).sqrt()
            param_rms = param_rms.to(torch.float)
            scale_exp_avg_sq = torch.zeros_like(param_rms)
            scale_grads = torch.zeros(
                size_update_period, *param_rms.shape, dtype=torch.float, device=p.device
            )
            state["param_rms"] = param_rms
            state["scale_grads"] = scale_grads
            state["scale_exp_avg_sq"] = scale_exp_avg_sq

        scale_grads[step % size_update_period] = (p * grad).sum(
            dim=list(range(p.ndim)), keepdim=True
        )

        if step % size_update_period == size_update_period - 1:
            param_rms.copy_((p**2).mean(dim=list(range(p.ndim)), keepdim=True).sqrt())

        param_min_rms = group["param_min_rms"]
        delta = delta * param_rms.clamp(min=param_min_rms)

        if step % size_update_period == size_update_period - 1 and step > 0:
            beta2 = group["betas"][1]
            size_lr = group["lr"] * group["scalar_lr_scale"]
            param_max_rms = group["param_max_rms"]
            eps = group["eps"]
            beta2_corr = beta2**size_update_period
            scale_exp_avg_sq.mul_(beta2_corr).add_(
                (scale_grads**2).mean(dim=0),
                alpha=1 - beta2_corr,
            )
            size_step = (step + 1) // size_update_period
            bias_correction2 = 1 - beta2_corr**size_step
            denom = scale_exp_avg_sq.sqrt() + eps
            scale_step = (
                -size_lr * (bias_correction2**0.5) * scale_grads.sum(dim=0) / denom
            )
            is_too_small = param_rms < param_min_rms
            scale_step.masked_fill_(is_too_small, 0.0)
            scale_step.clamp_(min=-0.1, max=0.1)
            scale_step = torch.minimum(scale_step, (param_max_rms - param_rms) / param_rms)
            delta.add_(p * scale_step)

        return delta

    def _momentum_step(self, group, p, state, grad):
        delta = self._scaling_step(group, p, state, grad)
        beta1 = group["betas"][0]
        stored_delta = state.get("delta")
        if stored_delta is None:
            stored_delta = torch.zeros_like(p, dtype=torch.float)
            state["delta"] = stored_delta
        stored_delta.mul_(beta1).add_(delta, alpha=1.0 - beta1)
        return stored_delta

    def _get_clipping_scale(self, group) -> float:
        clipping_scale = group["clipping_scale"]
        if clipping_scale is None:
            return 1.0
        first_param = None
        for p in group["params"]:
            if p.grad is not None:
                first_param = p
                break
        if first_param is None:
            return 1.0
        first_state = self.state[first_param]
        step = first_state.get("step", 0)
        if step == 0:
            return 1.0

        clipping_update_period = group["clipping_update_period"]
        scalar_lr_scale = group["scalar_lr_scale"]
        tot_sumsq = torch.tensor(0.0, device=first_param.device)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if p.numel() == 1:
                tot_sumsq += (grad**2).sum() * (scalar_lr_scale**2)
            else:
                param_rms = self.state[p].get("param_rms")
                if param_rms is None:
                    param_rms = (p**2).mean(dim=list(range(p.ndim)), keepdim=True).sqrt().to(torch.float)
                    self.state[p]["param_rms"] = param_rms
                tot_sumsq += ((grad * param_rms) ** 2).sum()

        tot_norm = tot_sumsq.sqrt()
        model_norms = first_state.get("model_norms")
        if model_norms is None:
            model_norms = torch.zeros(clipping_update_period, device=first_param.device)
            first_state["model_norms"] = model_norms
        model_norms[step % clipping_update_period] = tot_norm

        irregular_estimate_steps = [i for i in (10, 20, 40) if i < clipping_update_period]
        if step % clipping_update_period == 0 or step in irregular_estimate_steps:
            sorted_norms = model_norms.sort()[0].to("cpu")
            if step in irregular_estimate_steps:
                sorted_norms = sorted_norms[-step:]
            num_norms = sorted_norms.numel()
            median = sorted_norms[min(num_norms - 1, (num_norms // 2))]
            threshold = clipping_scale * median
            if step in irregular_estimate_steps:
                threshold = threshold * 2.0
            first_state["model_norm_threshold"] = threshold
            first_state["num_clipped"] = 0

        model_norm_threshold = first_state.get("model_norm_threshold")
        if model_norm_threshold is None:
            return 1.0
        ans = min(1.0, (model_norm_threshold / (tot_norm + 1.0e-20)).item())
        if ans != ans:
            ans = 0.0
        if ans < 1.0:
            first_state["num_clipped"] = first_state.get("num_clipped", 0) + 1
        if ans == 0.0:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()
        return ans
