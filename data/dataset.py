from __future__ import annotations

import json
import math
import pathlib
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_audio(
    path: str,
    target_sr: int,
    *,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> torch.Tensor:
    """
    Returns mono float tensor shaped (T,).
    Uses soundfile if available, else torchaudio.
    """
    p = str(path)
    # Fallback to ffmpeg-python if possible, or standard torchaudio load
    import torchaudio
    
    try:
        # Try standard loading first (handles most cases if backend is correct)
        if start_sec is not None and duration_sec is not None:
            try:
                info = torchaudio.info(p)
                sr = info.sample_rate
                frame_offset = int(round(start_sec * sr))
                num_frames = int(round(duration_sec * sr))
                wav, sr = torchaudio.load(p, frame_offset=frame_offset, num_frames=num_frames)
            except AttributeError:
                # Fallback for environments where torchaudio.info is missing
                wav, sr = torchaudio.load(p)
                frame_offset = int(round(start_sec * sr))
                num_frames = int(round(duration_sec * sr))
                wav = wav[:, frame_offset : frame_offset + num_frames]
        else:
            wav, sr = torchaudio.load(p)
    except Exception as e:
        # Fallback to direct ffmpeg via torchaudio's specialized reader if simple load fails? 
        # Or simple re-raise. For robustness, we assume caller handles the error (retries).
        raise RuntimeError(f"Failed to load {p}: {e}")

    wav = wav.mean(dim=0) # Convert to mono
    if sr != target_sr:
        try:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception as e:
            raise RuntimeError(f"Resampling failed for {p}: {e}")
    return wav


def _random_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return torch.nn.functional.pad(wav, (0, num_samples - wav.numel()))
    start = random.randint(0, wav.numel() - num_samples)
    return wav[start : start + num_samples]


def _start_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return torch.nn.functional.pad(wav, (0, num_samples - wav.numel()))
    return wav[:num_samples]


@dataclass
class ManifestConfig:
    manifest_path: str
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    random_crop: bool = True


class AudioManifestDataset(Dataset):
    def __init__(self, cfg: ManifestConfig):
        super().__init__()
        self.cfg = cfg
        self.items: List[Dict[str, Any]] = []
        for line in pathlib.Path(cfg.manifest_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            self.items.append(json.loads(line))
        if len(self.items) == 0:
            raise ValueError(f"Empty manifest: {cfg.manifest_path}")
        self.num_samples = int(math.ceil(cfg.segment_seconds * cfg.sample_rate))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Robust loading loop: try up to 10 times to find a valid sample
        attempts = 0
        while attempts < 10:
            try:
                it = self.items[idx]
                start = it.get("start")
                dur = it.get("duration")
                
                wav = _load_audio(
                    it["audio_filepath"],
                    self.cfg.sample_rate,
                    start_sec=float(start) if start is not None else None,
                    duration_sec=float(dur) if dur is not None else None,
                )
                
                wav = _random_crop(wav, self.num_samples) if self.cfg.random_crop else _start_crop(wav, self.num_samples)
                out = {"wav": wav, "meta": it}
                return out
            
            except Exception as e:
                # If loading fails (missing file, bad format, etc.), log and retry with a new random index
                attempts += 1
                logging.warning(f"Error loading index {idx} ({self.items[idx].get('audio_filepath', 'unknown')}): {e}. Retrying with random sample.")
                idx = random.randint(0, len(self.items) - 1)
        
        # If we failed 10 times, raise the error to avoid infinite loops or silent failures
        raise RuntimeError(f"Failed to load any valid sample after {attempts} attempts.")


def collate_fixed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    wav = torch.stack([b["wav"] for b in batch], dim=0)  # (B,T)
    wav = wav.unsqueeze(1)  # (B,1,T)
    meta = [b["meta"] for b in batch]
    return {"wav": wav, "meta": meta}
