from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Standard rotary positional embedding, head-dim half-rotation pattern.

    Builds cos/sin tables on first use and caches them up to the max seen T.
    Apply via `apply_rotary(q, k, cos, sin)`.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotary, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_T = 0
        self.register_buffer("cos_cache", torch.zeros(0), persistent=False)
        self.register_buffer("sin_cache", torch.zeros(0), persistent=False)

    def _maybe_build(self, T: int, device: torch.device, dtype: torch.dtype) -> None:
        if T <= self._cached_T and self.cos_cache.device == device:
            return
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq.to(device))   # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                         # (T, head_dim)
        self.cos_cache = emb.cos().to(dtype)
        self.sin_cache = emb.sin().to(dtype)
        self._cached_T = T

    def forward(self, T: int, device: torch.device, dtype: torch.dtype):
        self._maybe_build(T, device, dtype)
        return self.cos_cache[:T], self.sin_cache[:T]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, head_dim); cos/sin: (T, head_dim) -> broadcast to (1, 1, T, head_dim).
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class MultiHeadSelfAttentionRotary(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv_proj(x)                                   # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                              # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                                    # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary(T, x.device, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, T) -> (B, 1, 1, T) of additive -inf at padded positions for SDPA
            mask = key_padding_mask.to(torch.bool)
            attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )                                                          # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module: LN -> PW(2D) -> GLU -> DW(k) -> BN -> SiLU -> PW(D) -> Dropout."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.ln(x).transpose(1, 2)              # (B, D, T)
        y = self.pw1(y)                              # (B, 2D, T)
        y = F.glu(y, dim=1)                          # (B, D, T)
        y = self.dw(y)                               # (B, D, T)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y)                              # (B, D, T)
        y = self.dropout(y)
        return y.transpose(1, 2)                     # (B, T, D)


class ConformerLayer(nn.Module):
    """Pre-norm macaron Conformer block.

    Compatible call signature with the prior Zipformer layer so MHCWrapper plugs in
    unchanged. `pos_emb`, `chunk_size`, `attn_mask` arguments are accepted and ignored.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_module_kernel: int = 31,
        dropout: float = 0.0,
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

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        chunk_size: int = -1,
        attn_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (T, B, D) — convert to (B, T, D) internally for ops, return (T, B, D).
        del pos_emb, chunk_size, attn_mask  # accepted for signature compat; ignored
        x = x.transpose(0, 1)                            # (B, T, D)

        # Macaron FFN #1
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        # MHSA
        x = x + self.attn_dropout(self.attn(self.norm_attn(x), key_padding_mask=src_key_padding_mask))
        # Convolution module
        x = x + self.conv(x)
        # Macaron FFN #2
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        # Final norm
        x = self.norm_final(x)

        return x.transpose(0, 1)                         # (T, B, D)
