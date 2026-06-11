from __future__ import annotations

import json
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


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


class AudioDataset(torch.utils.data.Dataset):
    """Map-style dataset over a JSONL manifest.

    Each manifest line is ``{"audio_filepath": "...", ...}`` (other keys like
    ``duration``, ``text``, ``dataset`` are passed through as metadata).

    Path resolution: a relative ``audio_filepath`` is resolved against the
    *parent directory of the manifest file*. This matches the Tier-2 layout
    produced by ``clae_data.pack`` where manifest rows store paths relative
    to the dataset repo root (e.g. ``audio/openslr53/utt00001.flac``).

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
            self._manifest_root = pathlib.Path(cfg.manifest).resolve().parent
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
