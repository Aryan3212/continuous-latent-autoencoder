from __future__ import annotations

import torch
import torch.nn as nn

from models.conformer import (
    ConvModule,
    FeedForward,
    MultiHeadSelfAttentionRotary,
)


class SqueezeExcitation(nn.Module):
    """Channel-attention gating (NeMo FastConformer).

    Computes a per-channel scale from a global temporal average and multiplies
    the conv-module output by it: ``x = x * sigmoid(MLP(pool(x)))``.
    """

    def __init__(self, d_model: int, se_ratio: int = 8):
        super().__init__()
        inner = max(1, d_model // se_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, inner),
            nn.SiLU(),
            nn.Linear(inner, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, 1, D) gate
        s = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        g = torch.sigmoid(self.fc(s)).unsqueeze(1)  # (B, 1, D)
        return x * g


class FastConformerLayer(nn.Module):
    """Macaron Conformer block + Squeeze-and-Excitation (FastConformer).

    Drop-in replacement for ``ConformerLayer``: identical ``(B, T, D)`` in/out and
    the same constructor shape (plus ``use_se``). Differs from the standard
    Conformer only by the added SE gate after the convolution module; the 9-tap
    conv default (vs 31) is the main speed win and is selected via the config's
    ``cnn_module_kernel``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_module_kernel: int = 9,
        dropout: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, feedforward_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttentionRotary(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size=cnn_module_kernel, dropout=dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, feedforward_dim, dropout)
        self.norm_final = nn.LayerNorm(d_model)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.attn_dropout(self.attn(self.norm_attn(x)))
        x = x + self.conv(x)
        if self.use_se:
            x = x * self.se(x)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_final(x)
