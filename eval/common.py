from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch

BLANK_IDX = 0


def build_charset(texts: List[str]) -> List[str]:
    chars = sorted({c for t in texts for c in t.lower() if c != "\n"})
    return ["<blank>"] + chars


def greedy_decode_ctc(log_probs: torch.Tensor, id2ch: List[str]) -> List[str]:
    pred = log_probs.argmax(dim=-1).cpu().tolist()  # (B, T)
    outs: List[str] = []
    for seq in pred:
        last = None
        chars: List[str] = []
        for i in seq:
            if i == BLANK_IDX:
                last = i
                continue
            if last != i:
                chars.append(id2ch[i])
            last = i
        outs.append("".join(chars))
    return outs

from data.dataset import AudioDataset, DatasetConfig, collate_fixed
from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from utils.config import apply_overrides, load_config
from utils.schema import Config


@dataclass
class LoadedModel:
    cfg: Config
    device: torch.device
    frontend: ConvFrontend
    encoder: Encoder


def load_frozen_encoder(config_path: str, ckpt_path: str, overrides: List[str]) -> LoadedModel:
    cfg = apply_overrides(load_config(config_path), overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frontend = ConvFrontend(FrontendConfig(**cfg.model.frontend.model_dump()))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**cfg.model.encoder.model_dump()))
    model = torch.nn.ModuleDict({"frontend": frontend, "encoder": encoder}).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    # Using strict=False because older checkpoints might have 'bottleneck' in state_dict.
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return LoadedModel(cfg=cfg, device=device, frontend=frontend.to(device), encoder=encoder.to(device))


def _log_progress(name: str, n_samples: int, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    rate = n_samples / elapsed if elapsed > 0 else 0.0
    print(f"  [{name}] {n_samples} samples extracted ({rate:.0f} samples/s)", flush=True)


@torch.no_grad()
def iter_embeddings(
    lm: LoadedModel,
    manifest_path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int = 0,
    log_name: str = "",
) -> Iterable[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    ds = AudioDataset(
        DatasetConfig(
            manifest=manifest_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fixed, drop_last=False,
    )
    use_amp = lm.device.type == "cuda"
    start_t = time.perf_counter()
    n_samples = 0
    for i, batch in enumerate(dl):
        wav = batch["wav"].to(lm.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            h0 = lm.frontend(wav)
            hE = lm.encoder(h0)
        z = hE.float()  # (B,d,T')
        e = torch.cat([z.mean(dim=-1), z.std(dim=-1, unbiased=False)], dim=1)  # (B,2d)
        n_samples += e.size(0)
        if log_name and (i + 1) % 50 == 0:
            _log_progress(log_name, n_samples, start_t)
        yield e.cpu(), batch["meta"]
    if log_name:
        _log_progress(log_name, n_samples, start_t)


@torch.no_grad()
def iter_frame_features(
    lm: LoadedModel,
    manifest_path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int = 0,
    use_latent: bool = False, # deprecated
    log_name: str = "",
) -> Iterable[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    ds = AudioDataset(
        DatasetConfig(
            manifest=manifest_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fixed, drop_last=False,
    )
    use_amp = lm.device.type == "cuda"
    start_t = time.perf_counter()
    n_samples = 0
    for i, batch in enumerate(dl):
        wav = batch["wav"].to(lm.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            h0 = lm.frontend(wav)
            hE = lm.encoder(h0)  # (B,D,T')
        feats = hE.float()
        n_samples += feats.size(0)
        if log_name and (i + 1) % 50 == 0:
            _log_progress(log_name, n_samples, start_t)
        yield feats.transpose(1, 2).cpu(), batch["meta"]  # (B,T',D)
    if log_name:
        _log_progress(log_name, n_samples, start_t)
