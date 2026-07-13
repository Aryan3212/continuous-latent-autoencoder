"""Dataset loading + waveform augmentation (one module, no package).

Consolidates the former ``data/dataset.py`` (JSONL manifest dataset, collate,
manifest-root resolution) and ``data/augment.py`` (waveform augment + JEPA
chunk masking). Imported as ``from data_loading import ...`` by ``train.py``,
the eval probes, and a couple of scripts. Actual datasets live under
``$DATA_ROOT`` (default: ``<repo>/datasets``), not in the repo source.
"""
from __future__ import annotations

import json
import torchaudio
torchaudio.set_audio_backend("ffmpeg")
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from schema import WaveAugCfg, WaveChunkMaskCfg


# =========================================================================== #
# Dataset + manifest loading
# =========================================================================== #


@dataclass
class DatasetConfig:
    manifest: str | List[str]
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    random_crop: bool = True


def _read_manifest(paths: str | List[str]) -> List[Dict[str, Any]]:
    if isinstance(paths, str):
        paths = [paths]
    items: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    return items


def _random_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return F.pad(wav, (0, num_samples - wav.numel()))
    start = random.randint(0, wav.numel() - num_samples)
    return wav[start : start + num_samples]


def _start_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return F.pad(wav, (0, num_samples - wav.numel()))
    return wav[:num_samples]


def resolve_manifest_root(
    manifest_path: str, items: List[Dict[str, Any]]
) -> pathlib.Path:
    """Directory against which relative ``audio_filepath`` rows resolve.

    Two layouts exist in the wild: manifests sitting next to the audio
    (root = the manifest's own directory), and the packed (`scripts/housekeeping.py`) layout
    ``<root>/manifests/*.jsonl`` whose rows are relative to ``<root>``
    (e.g. ``audio/openslr53/utt1.flac`` — root = one level up). Probe the
    first relative row against both candidates so a wrong guess fails here,
    at construction, rather than mid-epoch inside a dataloader worker.
    """
    parent = pathlib.Path(manifest_path).resolve().parent
    for it in items:
        p = it.get("audio_filepath")
        if not p or os.path.isabs(p):
            continue
        for cand in (parent, parent.parent):
            if (cand / p).exists():
                return cand
        raise FileNotFoundError(
            f"relative audio_filepath {p!r} from {manifest_path} not found "
            f"under {parent} or {parent.parent}"
        )
    return parent  # no relative rows: root is never used


class AudioDataset(torch.utils.data.Dataset):
    """Map-style dataset over a JSONL manifest.

    Each manifest line is ``{"audio_filepath": "...", ...}`` (other keys like
    ``duration``, ``text``, ``dataset`` are passed through as metadata).

    Path resolution: a relative ``audio_filepath`` is resolved against the
    root returned by :func:`resolve_manifest_root` — the manifest's own
    directory, or one level up for the packed (`scripts/housekeeping.py`) layout where
    ``<root>/manifests/*.jsonl`` rows store paths relative to ``<root>``
    (e.g. ``audio/openslr53/utt00001.flac``).

    Multi-manifest note: when ``cfg.manifest`` is a list, all entries must
    contain absolute ``audio_filepath`` values. Mixing relative paths with a
    list-of-manifests is not currently supported because there is no
    per-row provenance to know which manifest a row came from.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.items = _read_manifest(cfg.manifest)
        self.num_samples = int(math.ceil(cfg.segment_seconds * cfg.sample_rate))
        self._resamplers: Dict[int, Any] = {}  # cached torchaudio Resample per source rate

        self._manifest_root: Optional[pathlib.Path]
        if isinstance(cfg.manifest, str):
            self._manifest_root = resolve_manifest_root(cfg.manifest, self.items)
        else:
            # List-of-manifests: relative paths can't be unambiguously resolved
            # without per-row provenance, so require absolute. Validate eagerly
            # to fail fast rather than mid-epoch.
            self._manifest_root = None
            for i, it in enumerate(self.items):
                p = it.get("audio_filepath")
                if p is None:
                    raise KeyError(
                        f"manifest row {i} missing required key 'audio_filepath'"
                    )
                if not os.path.isabs(p):
                    raise NotImplementedError(
                        "Relative audio_filepath with a list-of-manifests is not "
                        "supported (no per-row provenance to resolve against). "
                        "Pass a single manifest path or absolutize the entries."
                    )

    def __len__(self) -> int:
        return len(self.items)

    def _resample(self, wav: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.cfg.sample_rate:
            return wav
        import torchaudio
        key = src_sr
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(src_sr, self.cfg.sample_rate)
        return self._resamplers[key](wav)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torchaudio
        item = self.items[idx]
        if "audio_filepath" not in item:
            raise KeyError(
                f"manifest row {idx} missing required key 'audio_filepath'. "
                "The 'path' and 'audio' fallback keys are no longer supported; "
                "standardize on 'audio_filepath'."
            )
        path = item["audio_filepath"]
        if not os.path.isabs(path) and self._manifest_root is not None:
            path = str(self._manifest_root / path)
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.flatten()
        wav = self._resample(wav, int(sr))
        if self.cfg.random_crop:
            wav = _random_crop(wav, self.num_samples)
        else:
            wav = _start_crop(wav, self.num_samples)
        return {"wav": wav, "meta": item}


def collate_fixed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    wav = torch.stack([b["wav"] for b in batch], dim=0)  # (B, T)
    wav = wav.unsqueeze(1)                                # (B, 1, T)
    meta = [b["meta"] for b in batch]
    return {"wav": wav, "meta": meta}


# =========================================================================== #
# Waveform augmentation + JEPA chunk masking
# =========================================================================== #


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
