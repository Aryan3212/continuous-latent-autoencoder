import torch

from optim.scaled_adam import ScaledAdam


class ReferenceScaledAdam:
    def __init__(
        self,
        params,
        lr=3.0e-2,
        betas=(0.9, 0.98),
        scalar_lr_scale=0.1,
        eps=1.0e-8,
        param_min_rms=1.0e-5,
        param_max_rms=3.0,
        scalar_max=10.0,
        size_update_period=4,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.scalar_lr_scale = scalar_lr_scale
        self.eps = eps
        self.param_min_rms = param_min_rms
        self.param_max_rms = param_max_rms
        self.scalar_max = scalar_max
        self.size_update_period = size_update_period
        self.state = {}

    def _basic_step(self, p, state, grad):
        lr = self.lr
        if p.numel() == 1:
            lr = lr * self.scalar_lr_scale
        beta2 = self.betas[1]
        eps = self.eps
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

    def _scaling_step(self, p, state, grad):
        delta = self._basic_step(p, state, grad)
        if p.numel() == 1:
            return delta

        step = state["step"]
        size_update_period = self.size_update_period

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

        delta = delta * param_rms.clamp(min=self.param_min_rms)

        if step % size_update_period == size_update_period - 1 and step > 0:
            beta2 = self.betas[1]
            size_lr = self.lr * self.scalar_lr_scale
            beta2_corr = beta2**size_update_period
            scale_exp_avg_sq.mul_(beta2_corr).add_(
                (scale_grads**2).mean(dim=0),
                alpha=1 - beta2_corr,
            )
            size_step = (step + 1) // size_update_period
            bias_correction2 = 1 - beta2_corr**size_step
            denom = scale_exp_avg_sq.sqrt() + self.eps
            scale_step = (
                -size_lr * (bias_correction2**0.5) * scale_grads.sum(dim=0) / denom
            )
            is_too_small = param_rms < self.param_min_rms
            scale_step.masked_fill_(is_too_small, 0.0)
            scale_step.clamp_(min=-0.1, max=0.1)
            scale_step = torch.minimum(scale_step, (self.param_max_rms - param_rms) / param_rms)
            delta.add_(p * scale_step)

        return delta

    def _momentum_step(self, p, state, grad):
        delta = self._scaling_step(p, state, grad)
        beta1 = self.betas[0]
        stored_delta = state.get("delta")
        if stored_delta is None:
            stored_delta = torch.zeros_like(p, dtype=torch.float)
            state["delta"] = stored_delta
        stored_delta.mul_(beta1).add_(delta, alpha=1.0 - beta1)
        return stored_delta

    def step(self):
        for p in self.params:
            state = self.state.setdefault(p, {"step": 0})
            delta = self._momentum_step(p, state, p.grad)
            p.add_(delta)
            if p.numel() == 1:
                p.clamp_(min=-self.scalar_max, max=self.scalar_max)
            state["step"] += 1


def test_scaled_adam_matches_reference_single_step():
    torch.manual_seed(0)
    p1 = torch.randn(3, 4, requires_grad=True)
    p2 = torch.randn(1, requires_grad=True)
    g1 = torch.randn_like(p1)
    g2 = torch.randn_like(p2)
    p1.grad = g1.clone()
    p2.grad = g2.clone()

    ref_p1 = p1.detach().clone()
    ref_p2 = p2.detach().clone()
    ref_p1.grad = g1.clone()
    ref_p2.grad = g2.clone()

    ref = ReferenceScaledAdam([ref_p1, ref_p2])
    ref.step()

    opt = ScaledAdam([p1, p2])
    opt.step()

    assert torch.allclose(p1, ref_p1, atol=1e-6), (p1 - ref_p1).abs().max().item()
    assert torch.allclose(p2, ref_p2, atol=1e-6), (p2 - ref_p2).abs().max().item()


if __name__ == "__main__":
    test_scaled_adam_matches_reference_single_step()
    print("scaled_adam parity test passed")
