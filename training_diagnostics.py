from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from models.mhc import MHCWrapper, sinkhorn_log


COLLAPSED_DIM_STD_THRESHOLD = 1.0e-3
ISOTROPY_EPS = 1.0e-12


def representation_distribution_stats(
    population: torch.Tensor,
    *,
    prefix: str,
    collapsed_dim_std_threshold: float = COLLAPSED_DIM_STD_THRESHOLD,
    isotropy_eps: float = ISOTROPY_EPS,
) -> dict[str, float]:
    """Return detached FP32 covariance diagnostics for an ``(N, D)`` population."""
    if population.ndim != 2:
        raise ValueError(f"expected an (N, D) population, got {tuple(population.shape)}")
    if population.size(0) < 2:
        raise ValueError("representation diagnostics require at least two population rows")

    with torch.no_grad():
        x = population.detach().float()
        mean = x.mean(dim=0)
        centered = x - mean
        covariance = centered.T @ centered / (x.size(0) - 1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        trace = eigenvalues.sum()
        eig_sq_sum = eigenvalues.square().sum()
        largest = eigenvalues[-1]
        smallest = eigenvalues[0]
        dim_std = covariance.diagonal().clamp_min(0.0).sqrt()
        quantiles = torch.quantile(
            dim_std,
            torch.tensor([0.05, 0.5, 0.95], device=x.device, dtype=x.dtype),
        )
        offdiag = covariance - torch.diag_embed(covariance.diagonal())
        offdiag_count = max(covariance.numel() - covariance.size(0), 1)

        return {
            f"{prefix}/dim_std_mean": dim_std.mean().item(),
            f"{prefix}/dim_std_min": dim_std.min().item(),
            f"{prefix}/dim_std_p05": quantiles[0].item(),
            f"{prefix}/dim_std_median": quantiles[1].item(),
            f"{prefix}/dim_std_p95": quantiles[2].item(),
            f"{prefix}/dim_std_max": dim_std.max().item(),
            f"{prefix}/collapsed_dim_frac": (dim_std < collapsed_dim_std_threshold).float().mean().item(),
            f"{prefix}/mean_abs": mean.abs().mean().item(),
            f"{prefix}/cov_diag_mean": covariance.diagonal().mean().item(),
            f"{prefix}/cov_offdiag_abs_mean": (offdiag.abs().sum() / offdiag_count).item(),
            f"{prefix}/cov_offdiag_rms": (offdiag.square().sum() / offdiag_count).sqrt().item(),
            f"{prefix}/effective_rank": (trace.square() / eig_sq_sum.clamp_min(isotropy_eps)).item(),
            f"{prefix}/top_eigenvalue_fraction": (largest / trace.clamp_min(isotropy_eps)).item(),
            f"{prefix}/isotropy_ratio": (smallest / largest.clamp_min(isotropy_eps)).item(),
        }


def module_grad_norm(module: nn.Module) -> torch.Tensor:
    """FP32 L2 norm over a module's unscaled gradients."""
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("cannot measure gradient norm for a module without trainable parameters")
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total.add_(parameter.grad.detach().float().square().sum())
    return total.sqrt()


def parameters_have_finite_grads(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("cannot inspect an empty parameter collection")
    finite = torch.ones((), device=parameters[0].device, dtype=torch.bool)
    for parameter in parameters:
        if parameter.grad is not None:
            finite.logical_and_(torch.isfinite(parameter.grad.detach()).all())
    return finite


def snapshot_trainable_parameters(module: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def module_update_norm(module: nn.Module, before: list[torch.Tensor]) -> torch.Tensor:
    """FP32 L2 norm of the actual optimizer update, including weight decay."""
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if len(parameters) != len(before):
        raise ValueError("parameter snapshot does not match module")
    if not parameters:
        raise ValueError("cannot measure update norm for a module without trainable parameters")
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter, old_value in zip(parameters, before):
        total.add_((parameter.detach().float() - old_value.float()).square().sum())
    return total.sqrt()


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1).mean()


def mhc_diagnostics(encoder: nn.Module) -> dict[str, float]:
    """Return layer-qualified parameter, matrix, and gradient diagnostics."""
    wrappers = [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, MHCWrapper)
    ]
    stats: dict[str, float] = {
        "mhc/enabled": float(bool(wrappers)),
        "mhc/wrapper_count": float(len(wrappers)),
    }
    with torch.no_grad():
        for name, wrapper in wrappers:
            # Encoder names are ``mhc_wrappers.<layer>`` and therefore stable
            # across DDP and checkpoint loading.
            layer_name = name.replace("mhc_wrappers.", "layer_").replace(".", "_")
            prefix = f"mhc/{layer_name}"
            projected = sinkhorn_log(
                wrapper.H_res_logits.detach().float(),
                num_iters=wrapper.mhc_num_iters,
                tau=wrapper.mhc_tau,
            )
            if wrapper.identity_mix:
                alpha = torch.sigmoid(wrapper.H_res_alpha_logit.detach().float())
                identity = torch.eye(wrapper.num_streams, device=projected.device)
                effective = (1.0 - alpha) * identity + alpha * projected
                stats[f"{prefix}/residual_alpha"] = alpha.item()
            else:
                effective = projected

            diagonal = effective.diagonal()
            offdiag = effective - torch.diag_embed(diagonal)
            offdiag_count = max(effective.numel() - effective.size(0), 1)
            identity = torch.eye(wrapper.num_streams, device=effective.device)
            h_pre = wrapper.H_pre_logits.detach().float().softmax(dim=-1)
            stats[f"{prefix}/h_pre_entropy"] = _entropy(h_pre).item()
            stats[f"{prefix}/effective_diag_mean"] = diagonal.mean().item()
            stats[f"{prefix}/effective_offdiag_abs_mean"] = (offdiag.abs().sum() / offdiag_count).item()
            stats[f"{prefix}/effective_identity_frobenius"] = (effective - identity).norm().item()
            stats[f"{prefix}/sinkhorn_max_row_sum_error"] = (projected.sum(dim=-1) - 1.0).abs().max().item()
            stats[f"{prefix}/sinkhorn_max_col_sum_error"] = (projected.sum(dim=-2) - 1.0).abs().max().item()

            if wrapper.add_branch_out_to_residual:
                h_post = wrapper.H_post_logits.detach().float().softmax(dim=-1)
                stats[f"{prefix}/h_post_entropy"] = _entropy(h_post).item()
                stats[f"{prefix}/branch_scale"] = torch.tanh(wrapper.branch_scale.detach().float()).item()

            for parameter_name in (
                "H_res_logits",
                "H_pre_logits",
                "H_post_logits",
                "H_res_alpha_logit",
                "branch_scale",
            ):
                parameter = getattr(wrapper, parameter_name, None)
                if parameter is not None:
                    grad = parameter.grad
                    value = 0.0 if grad is None else grad.detach().float().norm().item()
                    stats[f"{prefix}/grad_norm/{parameter_name}"] = value
    return stats
