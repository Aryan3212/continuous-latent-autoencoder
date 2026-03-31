from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import webdataset as wds


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
class WebDatasetConfig:
    urls: str | List[str]
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    random_crop: bool = True
    shuffle_size: int = 1000
    resampled: bool = True


class PreprocessSample:
    def __init__(self, cfg: WebDatasetConfig, num_samples: int):
        self.cfg = cfg
        self.num_samples = num_samples

    def __call__(self, sample):
        import torchaudio
        import io
        
        audio = None
        for k, v in sample.items():
            if k.endswith("flac") or k.endswith("wav"):
                audio = v
                break
                
        meta = None
        for k, v in sample.items():
            if k.endswith("json"):
                meta = v
                break

        if isinstance(audio, bytes):
            try:
                audio, _ = torchaudio.load(io.BytesIO(audio))
            except Exception as e:
                print(f"Error loading audio: {e}")
                return None
        elif audio is None:
            return None

        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        elif audio.ndim == 1:
            pass # already mono
        else:
            audio = audio.flatten()

        if self.cfg.random_crop:
            audio = _random_crop(audio, self.num_samples)
        else:
            audio = _start_crop(audio, self.num_samples)

        return {"wav": audio, "meta": meta}

def is_valid_sample(x):
    return x is not None

def get_audio_wds(cfg: WebDatasetConfig) -> wds.WebDataset:
    num_samples = int(math.ceil(cfg.segment_seconds * cfg.sample_rate))
    preprocess_fn = PreprocessSample(cfg, num_samples)

    # Using .decode() with "torch" then a custom map
    dataset = wds.WebDataset(cfg.urls, resampled=cfg.resampled, shardshuffle=False)
    if cfg.resampled and cfg.shuffle_size > 0:
        dataset = dataset.shuffle(cfg.shuffle_size)

    dataset = (
        dataset
        .decode("torch")
        .map(preprocess_fn)
        .select(is_valid_sample)
    )
    return dataset

def collate_fixed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    wav = torch.stack([b["wav"] for b in batch], dim=0)  # (B,T)
    wav = wav.unsqueeze(1)  # (B,1,T)
    meta = [b["meta"] for b in batch]
    return {"wav": wav, "meta": meta}
