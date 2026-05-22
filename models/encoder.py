from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import copy
import torch
import torch.nn as nn

from models.mhc import MHCConfig, MHCWrapper
from models.zipformer import CompactRelPositionalEncoding, Zipformer2EncoderLayer


@dataclass
class EncoderConfig:
    d_model: int = 256
    n_layers: int = 6
    num_heads: int = 4
    query_head_dim: int = 32
    pos_head_dim: int = 4
    value_head_dim: int = 16
    feedforward_dim: int = 512
    dropout: float = 0.1
    cnn_module_kernel: int = 31
    pos_dim: int = 192
    warmup_batches: float = 4000.0
    mhc: MHCConfig = field(default_factory=MHCConfig)

    def __post_init__(self) -> None:
        if isinstance(self.mhc, dict):
            self.mhc = MHCConfig(**self.mhc)


class Encoder(nn.Module):
    """
    Encoder over low-rate tokens.

    Input:
      h0: (B, C, T')  -> output hE: (B, D, T')
    """

    def __init__(self, in_channels: int, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Conv1d(in_channels, cfg.d_model, kernel_size=1)
        layer = Zipformer2EncoderLayer(
            embed_dim=cfg.d_model,
            pos_dim=cfg.pos_dim,
            num_heads=cfg.num_heads,
            query_head_dim=cfg.query_head_dim,
            pos_head_dim=cfg.pos_head_dim,
            value_head_dim=cfg.value_head_dim,
            feedforward_dim=cfg.feedforward_dim,
            dropout=cfg.dropout,
            cnn_module_kernel=cfg.cnn_module_kernel,
            causal=False,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(cfg.n_layers)])
        self.pos_enc = CompactRelPositionalEncoding(cfg.pos_dim, dropout_rate=0.15, length_factor=1.0)


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

    def forward(self, h0: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h0.dim() != 3:
            raise ValueError(f"Expected h0 as (B,C,T'), got {tuple(h0.shape)}")
        x = self.in_proj(h0).transpose(1, 2)  # (B,T',D)
        x = x.transpose(0, 1)  # (T',B,D)
        pos_emb = self.pos_enc(x)

        residuals: torch.Tensor = x
        if key_padding_mask is not None and key_padding_mask.dim() != 2:
            raise ValueError(f"Expected key_padding_mask as (B,T'), got {tuple(key_padding_mask.shape)}")

        mhc_active = self._use_mhc
        streams = int(self.mhc_cfg.num_streams)

        for i, layer in enumerate(self.layers):
            use_mhc = self._mhc_layers[i]
            if use_mhc:
                if residuals.dim() == 3:
                    residuals = residuals.unsqueeze(0).expand(streams, -1, -1, -1)
                residuals = self.mhc_wrappers[i](
                    residuals,
                    pos_emb=pos_emb,
                    chunk_size=-1,
                    attn_mask=None,
                    src_key_padding_mask=key_padding_mask,
                )
            else:
                if mhc_active and residuals.dim() == 4:
                    residuals = self._apply_per_stream(
                        residuals, layer, pos_emb=pos_emb, key_padding_mask=key_padding_mask
                    )
                else:
                    residuals = layer(
                        residuals,
                        pos_emb,
                        chunk_size=-1,
                        attn_mask=None,
                        src_key_padding_mask=key_padding_mask,
                    )

        if residuals.dim() == 4:
            residuals = residuals.sum(dim=0)

        x = residuals.transpose(0, 1).transpose(1, 2)  # (B,D,T')
        return x

    def _apply_per_stream(
        self,
        residuals: torch.Tensor,
        layer: nn.Module,
        *,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        streams, seq_len, batch, dim = residuals.shape
        flat = residuals.permute(1, 2, 0, 3).reshape(seq_len, batch * streams, dim)
        if key_padding_mask is not None:
            flat_mask = key_padding_mask.repeat_interleave(streams, dim=0)
        else:
            flat_mask = None
        out = layer(
            flat,
            pos_emb,
            chunk_size=-1,
            attn_mask=None,
            src_key_padding_mask=flat_mask,
        )
        out = out.reshape(seq_len, batch, streams, dim).permute(2, 0, 1, 3)
        return out
