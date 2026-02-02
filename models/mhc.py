from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass
class MHCConfig:
    enabled: bool = True
    num_streams: int = 2
    start_layer: int = 2
    period: int = 3
    sinkhorn_iters: int = 10
    tau: float = 0.05
    dropout: float = 0.0


class MHCWrapper(nn.Module):
    def __init__(
        self,
        branch: nn.Module,
        dim: int,
        num_streams: int,
        layer_index: int,
        sinkhorn_iters: int,
        tau: float,
        dropout: float = 0.0,
        add_branch_out_to_residual: bool = True,
    ) -> None:
        super().__init__()
        if num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        self.branch = branch
        self.num_streams = num_streams
        self.mhc_num_iters = int(sinkhorn_iters)
        self.mhc_tau = float(tau)
        self.dropout = nn.Dropout(dropout)
        self.add_branch_out_to_residual = add_branch_out_to_residual

        init_residual_index = layer_index % num_streams
        init_h_res = torch.full((num_streams, num_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        init_h_pre = torch.full((1, num_streams), -8.0)
        init_h_pre[:, init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        if add_branch_out_to_residual:
            self.H_post_logits = nn.Parameter(torch.zeros(1, num_streams))

    def forward(
        self,
        residuals: torch.Tensor,
        *,
        pos_emb: torch.Tensor,
        chunk_size: int,
        attn_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if residuals.dim() != 4:
            raise ValueError(f"Expected residuals as (S,T,B,D), got {tuple(residuals.shape)}")

        h_res = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)
        residuals_out = torch.einsum("sr, s t b d -> r t b d", h_res, residuals)

        h_pre = self.H_pre_logits.softmax(dim=-1)
        branch_input = torch.einsum("vs, s t b d -> v t b d", h_pre, residuals).squeeze(0)
        branch_out = self.branch(
            branch_input,
            pos_emb,
            chunk_size=chunk_size,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        branch_out = self.dropout(branch_out)

        if self.add_branch_out_to_residual:
            h_post = self.H_post_logits.softmax(dim=-1)
            branch_to_residuals = torch.einsum("vs, t b d -> s t b d", h_post, branch_out)
            residuals_out = residuals_out + branch_to_residuals

        return residuals_out
