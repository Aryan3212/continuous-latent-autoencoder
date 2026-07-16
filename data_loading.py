"""Dataset loading and waveform augmentation."""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any

import torchaudio
import torch
import torch.nn.functional as F

from schema import SpanMaskCfg, WaveAugCfg


@dataclass
class DatasetConfig:
    manifest: str | list[str]
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    random_crop: bool = True


def _read_manifest(paths: str | list[str]) -> list[tuple[dict[str, Any], pathlib.Path]]:
    if isinstance(paths, str):
        paths = [paths]
    records: list[tuple[dict[str, Any], pathlib.Path]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            items = [json.loads(line) for line in file if line.strip()]
        root = resolve_manifest_root(path, items)
        records.extend((item, root) for item in items)
    return records


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
    manifest_path: str, items: list[dict[str, Any]]
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
    return parent


class AudioDataset(torch.utils.data.Dataset):
    """Map-style dataset over one or more JSONL manifests."""

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        records = _read_manifest(cfg.manifest)
        self.items = [item for item, _ in records]
        self._manifest_roots = [root for _, root in records]
        self.num_samples = int(math.ceil(cfg.segment_seconds * cfg.sample_rate))
        self._resamplers: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _resample(self, wav: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.cfg.sample_rate:
            return wav
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = torchaudio.transforms.Resample(
                src_sr, self.cfg.sample_rate
            )
        return self._resamplers[src_sr](wav)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        path = item["audio_filepath"]
        if not os.path.isabs(path):
            path = str(self._manifest_roots[idx] / path)
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


def collate_fixed(batch: list[dict[str, Any]]) -> dict[str, Any]:
    wav = torch.stack([item["wav"] for item in batch], dim=0).unsqueeze(1)
    meta = [item["meta"] for item in batch]
    return {"wav": wav, "meta": meta}


def make_span_masks(
    batch_size: int,
    num_frames: int,
    cfg: SpanMaskCfg,
) -> torch.Tensor:
    """Build CPU masks of contiguous frame spans."""
    masks = torch.zeros((batch_size, num_frames), dtype=torch.float32)
    if not cfg.enabled or num_frames <= 0 or cfg.ratio == 0.0:
        return masks
    target = max(1, int(round(num_frames * cfg.ratio)))
    min_span = min(cfg.min_span_frames, num_frames)
    max_span = min(cfg.max_span_frames, num_frames)
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
    """Expand frontend-frame masks and apply them to waveform samples."""
    waveform_length = wav.shape[-1]
    sample_mask = frame_masks.repeat_interleave(int(samples_per_frame), dim=-1)
    if sample_mask.size(1) > waveform_length:
        sample_mask = sample_mask[:, :waveform_length]
    elif sample_mask.size(1) < waveform_length:
        sample_mask = F.pad(sample_mask, (0, waveform_length - sample_mask.size(1)))
    sample_mask = sample_mask.to(device=wav.device, dtype=wav.dtype)
    return wav * (1.0 - sample_mask.unsqueeze(1))


def apply_frame_mask(x: torch.Tensor, frame_masks: torch.Tensor) -> torch.Tensor:
    """Zero complete frames in a channels-first feature tensor."""
    mask = frame_masks.to(device=x.device, dtype=x.dtype).unsqueeze(1)
    return x * (1.0 - mask)


def apply_waveform_augment(wav: torch.Tensor, sample_rate: int, cfg: WaveAugCfg) -> torch.Tensor:
    """Apply independent waveform augmentations to each batch item."""
    if not cfg.enabled:
        return wav

    batch_size = wav.shape[0]
    device = wav.device
    dtype = wav.dtype

    if cfg.gain_prob > 0:
        gain_mask = (torch.rand(batch_size, 1, 1, device=device) < cfg.gain_prob).to(dtype)
        gain = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.gain_min, cfg.gain_max
        )
        wav = wav * (gain_mask * gain + (1.0 - gain_mask))

    if cfg.noise_prob > 0:
        noise_mask = (torch.rand(batch_size, 1, 1, device=device) < cfg.noise_prob).to(dtype)
        snr_db = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.noise_snr_min, cfg.noise_snr_max
        )
        noise = torch.randn_like(wav)
        ra = wav.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        rb = noise.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        target_ratio = 10.0 ** (snr_db / 20.0)
        scale = (ra / (rb * target_ratio)).clamp_min(0.0)
        wav = wav + noise_mask * (noise * scale)

    if cfg.lowpass_prob > 0:
        taps = 63
        cutoff = torch.empty(batch_size, 1, device=device).uniform_(
            cfg.lowpass_min_freq, cfg.lowpass_max_freq
        )
        fc = (cutoff / float(sample_rate)).clamp(max=0.5)
        t = torch.arange(taps, device=device, dtype=torch.float32) - (taps - 1) / 2
        h = 2.0 * fc * torch.sinc(2.0 * fc * t)
        h = h * torch.hann_window(taps, periodic=False, device=device)
        h = (h / h.sum(dim=-1, keepdim=True)).to(dtype)
        apply_mask = torch.rand(batch_size, 1, 1, device=device) < cfg.lowpass_prob
        xp = F.pad(wav, (taps // 2, taps // 2), mode="reflect")
        filtered = F.conv1d(
            xp.transpose(0, 1), h.unsqueeze(1), groups=batch_size
        ).transpose(0, 1).to(dtype)
        wav = torch.where(apply_mask, filtered, wav)

    if cfg.clip_prob > 0:
        clip_mask = torch.rand(batch_size, 1, 1, device=device) < cfg.clip_prob
        peak = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        thresh = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.clip_min, 0.99
        ) * peak
        clipped = torch.maximum(torch.minimum(wav, thresh), -thresh)
        wav = torch.where(clip_mask, clipped, wav)

    return wav
