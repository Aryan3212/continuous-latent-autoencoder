from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 10, tau: float = 0.05) -> torch.Tensor:
    n = logits.shape[-1]
    z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=z.device, dtype=z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(z + u.unsqueeze(-1), dim=-2)

    return torch.exp(z + u.unsqueeze(-1) + v.unsqueeze(-2))


class MHCWrapper(nn.Module):
    def __init__(
        self,
        num_streams: int,
        layer_index: int,
        sinkhorn_iters: int,
        tau: float,
        dropout: float = 0.0,
        add_branch_out_to_residual: bool = True,
        identity_mix: bool = True,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        self.num_streams = num_streams
        self.mhc_num_iters = int(sinkhorn_iters)
        self.mhc_tau = float(tau)
        self.dropout = nn.Dropout(dropout)
        self.add_branch_out_to_residual = add_branch_out_to_residual
        self.identity_mix = identity_mix

        init_residual_index = layer_index % num_streams
        init_h_res = torch.full((num_streams, num_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        init_h_pre = torch.full((1, num_streams), -8.0)
        init_h_pre[:, init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        if add_branch_out_to_residual:
            self.H_post_logits = nn.Parameter(torch.zeros(1, num_streams))
            self.branch_scale = nn.Parameter(torch.zeros(1))

        if identity_mix:
            if alpha_init <= 0 or alpha_init >= 1:
                raise ValueError("alpha_init must be in (0, 1)")
            logit_alpha = math.log(alpha_init / (1 - alpha_init))
            self.H_res_alpha_logit = nn.Parameter(torch.tensor(logit_alpha))

    def forward(self, residuals: torch.Tensor, branch: nn.Module) -> torch.Tensor:
        projected = sinkhorn_log(
            self.H_res_logits,
            num_iters=self.mhc_num_iters,
            tau=self.mhc_tau,
        )

        if self.identity_mix:
            alpha = torch.sigmoid(self.H_res_alpha_logit)
            identity = torch.eye(
                self.num_streams,
                device=residuals.device,
                dtype=residuals.dtype,
            )
            h_res = (1 - alpha) * identity + alpha * projected
        else:
            h_res = projected

        residuals_out = torch.einsum("sr, s b t d -> r b t d", h_res, residuals)

        h_pre = self.H_pre_logits.softmax(dim=-1)
        branch_input = torch.einsum("vs, s b t d -> v b t d", h_pre, residuals).squeeze(0)
        branch_out = self.dropout(branch(branch_input))

        if self.add_branch_out_to_residual:
            h_post = self.H_post_logits.softmax(dim=-1)
            branch_out_scaled = branch_out * torch.tanh(self.branch_scale)
            branch_to_residuals = torch.einsum("vs, b t d -> s b t d", h_post, branch_out_scaled)
            residuals_out = residuals_out + branch_to_residuals

        return residuals_out
