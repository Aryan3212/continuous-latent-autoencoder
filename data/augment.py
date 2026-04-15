from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import torch
import torch.nn.functional as F


@dataclass
class FeatureMaskConfig:
    enabled: bool = True
    time_mask_prob: float = 0.7
    time_mask_len: int = 8
    channel_mask_prob: float = 0.3
    channel_mask_len: int = 32


def apply_feature_mask(h0: torch.Tensor, cfg: FeatureMaskConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SpecAug-style masking on learned features.
      h0: (B,C,T')
    Returns (masked_h0, mask) where mask is 1 at masked positions.
    """
    b, c, t = h0.shape
    mask = torch.zeros((b, 1, t), device=h0.device, dtype=h0.dtype)
    if not cfg.enabled:
        return h0, mask
    
    out = h0.clone()

    if random.random() < cfg.time_mask_prob:
        m = min(cfg.time_mask_len, t)
        if m > 0:
            start = random.randint(0, max(0, t - m))
            out[:, :, start : start + m] = 0.0
            mask[:, :, start : start + m] = 1.0

    if random.random() < cfg.channel_mask_prob:
        m = min(cfg.channel_mask_len, c)
        if m > 0:
            start = random.randint(0, max(0, c - m))
            out[:, start : start + m, :] = 0.0
            # Channel masking doesn't affect time-domain reconstruction loss targeting as easily
            # but we could mark it. For now, focus on time masking for MAR.

    return out, mask


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


# --- New Waveform Augmentations (Exp3) ---

@dataclass
class WaveAugConfig:
    enabled: bool = False
    
    # Noise
    noise_prob: float = 0.0
    noise_snr_min: float = 3.0
    noise_snr_max: float = 20.0
    
    # RIR / Reverb
    reverb_prob: float = 0.0
    # For now, simplistic reverb via decay if no RIR files provided
    
    # Low Pass
    lowpass_prob: float = 0.0
    lowpass_min_freq: float = 2000.0
    lowpass_max_freq: float = 8000.0
    
    # Volume
    gain_prob: float = 0.0
    gain_min: float = 0.5
    gain_max: float = 1.5
    
    # Clipping
    clip_prob: float = 0.0
    clip_min: float = 0.5 # Threshold relative to peak


def apply_waveform_augment(wav: torch.Tensor, sample_rate: int, cfg: WaveAugConfig) -> torch.Tensor:
    """
    Apply augmentations to a batch of waveforms (B, 1, T).
    Currently implemented as per-sample loop for correctness with randomization.
    """
    if not cfg.enabled:
        return wav
    
    out_list = []
    device = wav.device
    for i in range(wav.shape[0]):
        x = wav[i] # (1, T)
        
        # 1. Gain
        if cfg.gain_prob > 0 and random.random() < cfg.gain_prob:
            g = random.uniform(cfg.gain_min, cfg.gain_max)
            x = x * g
            
        # 2. Add Gaussian Noise (approx background)
        if cfg.noise_prob > 0 and random.random() < cfg.noise_prob:
            snr = random.uniform(cfg.noise_snr_min, cfg.noise_snr_max)
            noise = torch.randn_like(x)
            x = mix_by_snr(x, noise, snr)
            
        # 3. Low Pass (approx via simple moving average or torchaudio if available)
        # We use a simple approximation: temporal pooling or conv
        if cfg.lowpass_prob > 0 and random.random() < cfg.lowpass_prob:
            # Random cutoff freq proxy: kernel size for avg pool
            # High freq -> small kernel, Low freq -> large kernel
            # 16k SR. Kernel 2 -> 8k, Kernel 4 -> 4k, Kernel 8 -> 2k
            k_float = sample_rate / random.uniform(cfg.lowpass_min_freq, cfg.lowpass_max_freq)
            k = max(2, int(k_float))
            if k % 2 == 0: k += 1
            if k > 1:
                # Pad to keep size
                pad = k // 2
                x_padded = F.pad(x.unsqueeze(0), (pad, pad), mode='reflect')
                x = F.avg_pool1d(x_padded, kernel_size=k, stride=1).squeeze(0)
                
        # 4. Clipping
        if cfg.clip_prob > 0 and random.random() < cfg.clip_prob:
            thresh = random.uniform(cfg.clip_min, 0.99)
            x = x.clamp(-thresh, thresh)
            
        out_list.append(x)
        
    return torch.stack(out_list, dim=0)
