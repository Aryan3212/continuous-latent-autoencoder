from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class FeatureMaskConfig:
    enabled: bool = True
    time_mask_prob: float = 0.7
    time_mask_len: int = 8
    channel_mask_prob: float = 0.3
    channel_mask_len: int = 32


def apply_feature_mask(h0: torch.Tensor, cfg: FeatureMaskConfig) -> torch.Tensor:
    """
    SpecAug-style masking on learned features.
      h0: (B,C,T')
    """
    if not cfg.enabled:
        return h0
    b, c, t = h0.shape
    out = h0.clone()

    if random.random() < cfg.time_mask_prob:
        m = min(cfg.time_mask_len, t)
        if m > 0:
            start = random.randint(0, max(0, t - m))
            out[:, :, start : start + m] = 0.0

    if random.random() < cfg.channel_mask_prob:
        m = min(cfg.channel_mask_len, c)
        if m > 0:
            start = random.randint(0, max(0, c - m))
            out[:, start : start + m, :] = 0.0

    return out


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


def mix_by_snr(a: torch.Tensor, b: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mix waveforms a (primary) and b (interferer) such that:
      20log10(rms(a)/rms(b_scaled)) == snr_db
    Expects a,b shaped (1,T) or (T,).
    """
    if a.dim() == 1:
        a1 = a.unsqueeze(0)
    else:
        a1 = a
    if b.dim() == 1:
        b1 = b.unsqueeze(0)
    else:
        b1 = b
    ra = rms(a1)
    rb = rms(b1)
    target_ratio = 10 ** (snr_db / 20.0)
    scale = (ra / (rb * target_ratio)).clamp(min=0.0)
    b_scaled = b1 * scale
    y = a1 + b_scaled
    return y.squeeze(0) if a.dim() == 1 else y


@dataclass
class MixConfig:
    enabled: bool = False
    snr_db_min: float = 3.0
    snr_db_max: float = 15.0
    prob: float = 0.0
    swap_prob: float = 0.0  # when mixing, probability to make B the primary


def maybe_mix_pair(a: torch.Tensor, b: torch.Tensor, cfg: MixConfig) -> Tuple[torch.Tensor, bool, float, int]:
    if (not cfg.enabled) or random.random() > cfg.prob:
        return a, False, 0.0, 0
    snr_db = random.uniform(cfg.snr_db_min, cfg.snr_db_max)
    primary_idx = 0
    if cfg.swap_prob > 0.0 and random.random() < cfg.swap_prob:
        # Make B primary by swapping roles.
        primary_idx = 1
        y = mix_by_snr(b, a, snr_db)
    else:
        y = mix_by_snr(a, b, snr_db)
    return y, True, snr_db, primary_idx
