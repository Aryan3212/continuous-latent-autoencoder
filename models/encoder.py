from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.conformer import ConformerLayer
from models.fastconformer import FastConformerLayer
from models.mhc import MHCWrapper
from schema import EncoderCfg


def _build_encoder_layer(cfg: EncoderCfg) -> nn.Module:
    common = dict(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        feedforward_dim=cfg.feedforward_dim,
        cnn_module_kernel=cfg.cnn_module_kernel,
        dropout=cfg.dropout,
    )
    if cfg.encoder_type == "conformer":
        return ConformerLayer(**common)
    if cfg.encoder_type == "fastconformer":
        return FastConformerLayer(**common, use_se=cfg.use_se)
    raise ValueError(f"unknown encoder_type={cfg.encoder_type!r} (expected 'conformer'|'fastconformer')")


class Encoder(nn.Module):
    """Map frontend features ``(B, C, T)`` to latents ``(B, D, T)``."""

    def __init__(self, in_channels: int, cfg: EncoderCfg):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, cfg.d_model, kernel_size=1)

        self.xscaling = cfg.xscaling
        if cfg.xscaling:
            self.register_buffer("xscale", torch.tensor(math.sqrt(cfg.d_model)), persistent=False)

        self.layers = nn.ModuleList(_build_encoder_layer(cfg) for _ in range(cfg.n_layers))

        self._use_mhc = bool(cfg.mhc.enabled and cfg.mhc.num_streams > 1)
        self.num_streams = cfg.mhc.num_streams
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
        x = self.in_proj(h0).transpose(1, 2)                 # (B, T', D)
        if self.xscaling:
            x = x * self.xscale

        residuals: torch.Tensor = x

        for i, layer in enumerate(self.layers):
            use_mhc = self._mhc_layers[i]
            if use_mhc:
                if residuals.dim() == 3:
                    residuals = residuals.unsqueeze(0).expand(self.num_streams, -1, -1, -1)
                residuals = self.mhc_wrappers[i](residuals, layer)
            else:
                if residuals.dim() == 4:
                    residuals = self._apply_per_stream(residuals, layer)
                else:
                    residuals = layer(residuals)

        if residuals.dim() == 4:
            residuals = residuals.mean(dim=0)

        x = residuals.transpose(1, 2)                         # (B, D, T')
        return x

    def _apply_per_stream(self, residuals: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        streams, batch, seq_len, dim = residuals.shape
        out = layer(residuals.reshape(streams * batch, seq_len, dim))
        return out.reshape(streams, batch, seq_len, dim)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for i, use_mhc in enumerate(self._mhc_layers):
            if not use_mhc:
                continue
            old_prefix = f"{prefix}mhc_wrappers.{i}.branch."
            for key in [key for key in state_dict if key.startswith(old_prefix)]:
                state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
