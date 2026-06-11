from __future__ import annotations

import random
import torch
import torch.nn.functional as F

from utils.schema import WaveAugCfg, WaveChunkMaskCfg


def make_frame_chunk_masks(
    batch_size: int,
    num_frames: int,
    cfg: WaveChunkMaskCfg,
) -> torch.Tensor:
    """Returns (B, num_frames), 1 where masked. CPU tensor (float32).

    Overlays random spans of length [min_span_frames, max_span_frames] per
    sample until target_ratio coverage is reached (or attempts cap hit).
    Independent draw per sample.
    """
    masks = torch.zeros((batch_size, num_frames), dtype=torch.float32)
    if not cfg.enabled or num_frames <= 0 or cfg.target_ratio <= 0.0:
        return masks
    target = max(1, int(round(num_frames * cfg.target_ratio)))
    min_span = max(1, int(cfg.min_span_frames))
    max_span = max(min_span, int(cfg.max_span_frames))
    max_span = min(max_span, num_frames)
    for b in range(batch_size):
        covered = 0
        attempts = 0
        max_attempts = max(50, target * 4)
        while covered < target and attempts < max_attempts:
            span = random.randint(min_span, max_span)
            span = min(span, num_frames)
            start = random.randint(0, num_frames - span)
            before = int(masks[b, start:start + span].sum().item())
            masks[b, start:start + span] = 1.0
            covered += span - before
            attempts += 1
    return masks


def apply_waveform_chunk_mask(
    wav: torch.Tensor,
    frame_masks: torch.Tensor,
    samples_per_frame: int,
) -> torch.Tensor:
    """Zero out audio samples corresponding to masked frames.

    wav:           (B, 1, T_wav)
    frame_masks:   (B, num_frames), 1 = masked
    samples_per_frame: frontend total stride (e.g. product of conv strides)

    Returns (B, 1, T_wav) with samples in masked frames replaced by zero.
    Sample mask is built via repeat_interleave so a masked frame zeros exactly
    `samples_per_frame` contiguous audio samples, then trimmed/padded to T_wav.
    """
    _, _, T_wav = wav.shape
    sample_mask = frame_masks.repeat_interleave(int(samples_per_frame), dim=-1)
    if sample_mask.size(1) > T_wav:
        sample_mask = sample_mask[:, :T_wav]
    elif sample_mask.size(1) < T_wav:
        pad = T_wav - sample_mask.size(1)
        sample_mask = F.pad(sample_mask, (0, pad), value=0.0)
    sample_mask = sample_mask.to(device=wav.device, dtype=wav.dtype)
    return wav * (1.0 - sample_mask.unsqueeze(1))


def apply_waveform_augment(wav: torch.Tensor, sample_rate: int, cfg: WaveAugCfg) -> torch.Tensor:
    """
    Apply augmentations to a batch of waveforms (B, 1, T). Fully vectorised.
    Each sample gets independent per-augmentation random decisions.
    Lowpass uses bucketed batching by kernel size (≤ ~5 distinct kernels typically).
    """
    if not cfg.enabled:
        return wav

    B = wav.shape[0]
    device = wav.device
    dtype = wav.dtype

    # 1. Gain — per-sample factor, per-sample apply mask
    if cfg.gain_prob > 0:
        gain_mask = (torch.rand(B, 1, 1, device=device) < cfg.gain_prob).to(dtype)
        gain = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(cfg.gain_min, cfg.gain_max)
        wav = wav * (gain_mask * gain + (1.0 - gain_mask))

    # 2. Noise — per-sample SNR, per-sample apply mask
    if cfg.noise_prob > 0:
        noise_mask = (torch.rand(B, 1, 1, device=device) < cfg.noise_prob).to(dtype)
        snr_db = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(cfg.noise_snr_min, cfg.noise_snr_max)
        noise = torch.randn_like(wav)
        ra = wav.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()       # (B, 1, 1)
        rb = noise.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()     # (B, 1, 1)
        target_ratio = 10.0 ** (snr_db / 20.0)
        scale = (ra / (rb * target_ratio)).clamp_min(0.0)
        wav = wav + noise_mask * (noise * scale)

    # 3. Lowpass — per-sample windowed-sinc FIR, one grouped conv over the batch.
    # (The previous moving-average filter's -3dB point was ~0.44x the configured
    # cutoff and couldn't represent gentle cutoffs at all.)
    if cfg.lowpass_prob > 0:
        taps = 63  # odd; transition width ~ 4*fs/taps ≈ 1 kHz at 16 kHz
        cutoff = torch.empty(B, 1, device=device).uniform_(cfg.lowpass_min_freq, cfg.lowpass_max_freq)
        fc = (cutoff / float(sample_rate)).clamp(max=0.5)                     # (B, 1) normalized
        t = torch.arange(taps, device=device, dtype=torch.float32) - (taps - 1) / 2
        h = 2.0 * fc * torch.sinc(2.0 * fc * t)                               # (B, taps)
        h = h * torch.hann_window(taps, periodic=False, device=device)
        h = (h / h.sum(dim=-1, keepdim=True)).to(dtype)
        apply_mask = torch.rand(B, 1, 1, device=device) < cfg.lowpass_prob
        xp = F.pad(wav, (taps // 2, taps // 2), mode="reflect")               # (B, 1, T+taps-1)
        # .to(dtype): under autocast conv1d emits fp16 while wav stays fp32.
        filtered = F.conv1d(xp.transpose(0, 1), h.unsqueeze(1), groups=B).transpose(0, 1).to(dtype)
        wav = torch.where(apply_mask, filtered, wav)

    # 4. Clipping — per-sample threshold relative to that sample's peak, so the
    # aug actually clips regardless of recording level.
    if cfg.clip_prob > 0:
        clip_mask = (torch.rand(B, 1, 1, device=device) < cfg.clip_prob)
        peak = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)           # (B, 1, 1)
        thresh = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(cfg.clip_min, 0.99) * peak
        clipped = torch.maximum(torch.minimum(wav, thresh), -thresh)
        wav = torch.where(clip_mask, clipped, wav)

    return wav
