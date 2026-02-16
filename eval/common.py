from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from data.dataset import AudioManifestDataset, ManifestConfig, collate_fixed
from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from utils.config import apply_overrides, load_config


@dataclass
class LoadedModel:
    cfg: Dict[str, Any]
    device: torch.device
    frontend: ConvFrontend
    encoder: Encoder


def load_frozen_encoder(config_path: str, ckpt_path: str, overrides: List[str]) -> LoadedModel:
    cfg = apply_overrides(load_config(config_path), overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcfg = cfg["model"]
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    model = torch.nn.ModuleDict({"frontend": frontend, "encoder": encoder}).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    # Using strict=False because older checkpoints might have 'bottleneck' in state_dict.
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return LoadedModel(cfg=cfg, device=device, frontend=frontend.to(device), encoder=encoder.to(device))


@torch.no_grad()
def iter_embeddings(
    lm: LoadedModel,
    manifest_path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int = 0,
) -> Iterable[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    ds = AudioManifestDataset(
        ManifestConfig(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fixed)
    for batch in dl:
        wav = batch["wav"].to(lm.device)
        h0 = lm.frontend(wav)
        hE = lm.encoder(h0)
        z = hE # (B,d,T')
        e = torch.cat([z.mean(dim=-1), z.std(dim=-1, unbiased=False)], dim=1)  # (B,2d)
        yield e.cpu(), batch["meta"]


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
) -> Iterable[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    ds = AudioManifestDataset(
        ManifestConfig(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fixed)
    for batch in dl:
        wav = batch["wav"].to(lm.device)
        h0 = lm.frontend(wav)
        hE = lm.encoder(h0)  # (B,D,T')
        feats = hE
        yield feats.transpose(1, 2).cpu(), batch["meta"]  # (B,T',D)
