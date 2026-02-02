from __future__ import annotations

import json
import math
import pathlib
import random
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
    try:
        import soundfile as sf  # type: ignore

        if start_sec is not None and duration_sec is not None:
            info = sf.info(p)
            start = int(round(start_sec * info.samplerate))
            frames = int(round(duration_sec * info.samplerate))
            wav, sr = sf.read(p, start=start, frames=frames, dtype="float32", always_2d=False)
        else:
            wav, sr = sf.read(p, dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if wav.dim() == 2:
            wav = wav.mean(dim=1)
        if sr != target_sr:
            try:
                import torchaudio  # type: ignore

                wav = torchaudio.functional.resample(wav, sr, target_sr)
            except Exception as e:
                raise RuntimeError(f"Need torchaudio for resampling {sr}->{target_sr}: {e}")
        return wav
    except Exception:
        import torchaudio  # type: ignore

        if start_sec is not None and duration_sec is not None:
            # frame_offset/num_frames are supported by many backends.
            info = torchaudio.info(p)
            frame_offset = int(round(start_sec * info.sample_rate))
            num_frames = int(round(duration_sec * info.sample_rate))
            wav, sr = torchaudio.load(p, frame_offset=frame_offset, num_frames=num_frames)  # (C,T)
        else:
            wav, sr = torchaudio.load(p)  # (C,T)
        wav = wav.mean(dim=0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
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


def collate_fixed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    wav = torch.stack([b["wav"] for b in batch], dim=0)  # (B,T)
    wav = wav.unsqueeze(1)  # (B,1,T)
    meta = [b["meta"] for b in batch]
    return {"wav": wav, "meta": meta}
