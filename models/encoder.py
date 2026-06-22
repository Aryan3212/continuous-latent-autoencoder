from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.conformer import ConformerLayer
from models.mhc import MHCWrapper
from schema import EncoderCfg


class Encoder(nn.Module):
    """Conformer-based encoder over low-rate tokens.

    Input:  h0 (B, C, T')  -> Output: hE (B, D, T').
    """

    def __init__(self, in_channels: int, cfg: EncoderCfg):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Conv1d(in_channels, cfg.d_model, kernel_size=1)

        layer = ConformerLayer(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            feedforward_dim=cfg.feedforward_dim,
            cnn_module_kernel=cfg.cnn_module_kernel,
            dropout=cfg.dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(cfg.n_layers)])

        self.mhc_cfg = cfg.mhc
        self._use_mhc = bool(cfg.mhc.enabled and cfg.mhc.num_streams > 1)
        self._mhc_layers: list[bool] = []
        self.mhc_wrappers = nn.ModuleList()
        for i in range(cfg.n_layers):
            use_mhc = (
                self._use_mhc
                and i >= cfg.mhc.start_layer
                and ((i - cfg.mhc.start_layer) % cfg.mhc.period == 0)
            )
            self._mhc_layers.append(use_mhc)
            if use_mhc:
                self.mhc_wrappers.append(
                    MHCWrapper(
                        branch=self.layers[i],
                        dim=cfg.d_model,
                        num_streams=cfg.mhc.num_streams,
                        layer_index=i,
                        sinkhorn_iters=cfg.mhc.sinkhorn_iters,
                        tau=cfg.mhc.tau,
                        dropout=cfg.mhc.dropout,
                        identity_mix=cfg.mhc.identity_mix,
                        alpha_init=cfg.mhc.alpha_init,
                    )
                )
            else:
                self.mhc_wrappers.append(nn.Identity())

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        if h0.dim() != 3:
            raise ValueError(f"Expected h0 as (B,C,T'), got {tuple(h0.shape)}")
        x = self.in_proj(h0).transpose(1, 2)                 # (B, T', D)

        residuals: torch.Tensor = x

        mhc_active = self._use_mhc
        streams = int(self.mhc_cfg.num_streams)

        for i, layer in enumerate(self.layers):
            use_mhc = self._mhc_layers[i]
            if use_mhc:
                if residuals.dim() == 3:
                    residuals = residuals.unsqueeze(0).expand(streams, -1, -1, -1)
                residuals = self.mhc_wrappers[i](residuals)
            else:
                if mhc_active and residuals.dim() == 4:
                    residuals = self._apply_per_stream(residuals, layer)
                else:
                    residuals = layer(residuals)

        if residuals.dim() == 4:
            residuals = residuals.sum(dim=0)

        x = residuals.transpose(1, 2)                         # (B, D, T')
        return x

    def _apply_per_stream(self, residuals: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        streams, batch, seq_len, dim = residuals.shape
        out = layer(residuals.reshape(streams * batch, seq_len, dim))
        return out.reshape(streams, batch, seq_len, dim)
