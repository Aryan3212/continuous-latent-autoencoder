from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch

BLANK_IDX = 0


def build_charset(texts: List[str]) -> List[str]:
    chars = sorted({c for t in texts for c in t.lower() if c != "\n"})
    return ["<blank>"] + chars


def greedy_decode_ctc(
    log_probs: torch.Tensor, id2ch: List[str], lens: List[int] | None = None
) -> List[str]:
    pred = log_probs.argmax(dim=-1).cpu().tolist()  # (B, T)
    outs: List[str] = []
    for bi, seq in enumerate(pred):
        if lens is not None:
            seq = seq[: int(lens[bi])]
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
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
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

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    model = torch.nn.ModuleDict({"frontend": frontend, "encoder": encoder}).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    filtered = {k: v for k, v in state["model"].items() if k.split(".", 1)[0] in {"frontend", "encoder"}}
    model.load_state_dict(filtered, strict=True)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return LoadedModel(cfg=cfg, device=device, frontend=frontend.to(device), encoder=encoder.to(device))


def _log_progress(name: str, n_samples: int, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    rate = n_samples / elapsed if elapsed > 0 else 0.0
    print(f"  [{name}] {n_samples} samples extracted ({rate:.0f} samples/s)", flush=True)


@torch.no_grad()
def iter_frame_features(
    lm: LoadedModel,
    manifest_path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int = 0,
    log_name: str = "",
    chunk_seconds: float | None = None,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
    """Yield (feats (B,T',D), valid_lens (B,), meta) per batch.

    chunk_seconds: when set and shorter than segment_seconds, the waveform is
    encoded in independent windows of that length and the frame features are
    concatenated along time. Pass the PRETRAINING segment length: the encoder
    has unmasked global attention + BatchNorm and only ever saw segment-length
    inputs, so a single pass over a longer padded waveform is
    out-of-distribution and degrades every frame.

    valid_lens counts the frames covered by real audio, from the manifest's
    `duration`; rows without a usable duration get the full frame count.
    """
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
        wav = batch["wav"].to(lm.device)  # (B, 1, S)
        B, _, S = wav.shape
        total_samples = S
        with torch.amp.autocast("cuda", enabled=use_amp):
            if chunk_seconds is not None and int(round(chunk_seconds * sample_rate)) < S:
                cs = int(round(chunk_seconds * sample_rate))
                n_chunks = math.ceil(S / cs)
                total_samples = n_chunks * cs
                if total_samples > S:
                    wav = torch.nn.functional.pad(wav, (0, total_samples - S))
                hE = lm.encoder(lm.frontend(wav.view(B * n_chunks, 1, cs)))  # (B*n, D, Tc)
                D, Tc = hE.size(1), hE.size(2)
                hE = hE.view(B, n_chunks, D, Tc).permute(0, 2, 1, 3).reshape(B, D, n_chunks * Tc)
            else:
                hE = lm.encoder(lm.frontend(wav))  # (B,D,T')
        feats = hE.float()
        n_frames = feats.size(-1)
        samples_per_frame = total_samples / n_frames
        lens: List[int] = []
        for m in batch["meta"]:
            dur = m.get("duration")
            t_valid = n_frames
            if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > 0:
                t_valid = min(n_frames, max(1, math.ceil(float(dur) * sample_rate / samples_per_frame)))
            lens.append(t_valid)
        n_samples += feats.size(0)
        if log_name and (i + 1) % 50 == 0:
            _log_progress(log_name, n_samples, start_t)
        yield feats.transpose(1, 2).cpu(), torch.tensor(lens, dtype=torch.long), batch["meta"]
    if log_name:
        _log_progress(log_name, n_samples, start_t)


@torch.no_grad()
def iter_embeddings_masked(
    lm: LoadedModel,
    manifest_path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int = 0,
    log_name: str = "",
) -> Iterable[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    """Pooled mean+std utterance embeddings, excluding zero-padding frames from pooling.

    Utterances shorter than segment_seconds are right-padded with zeros by the
    dataset; mean+std pooling over those frames dilutes the utterance
    statistics. When a manifest row carries a 'duration' field (seconds), only
    the frames covered by real audio enter the pooled mean/std. Rows without
    'duration' fall back to full-segment pooling. Yields (B, 2d) embeddings
    plus the batch metadata list.
    """
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
        n_frames = z.size(-1)
        samples_per_frame = wav.size(-1) / n_frames
        embs: List[torch.Tensor] = []
        for b, m in enumerate(batch["meta"]):
            dur = m.get("duration")
            t_valid = n_frames
            if dur is not None:
                t_valid = min(n_frames, max(1, math.ceil(float(dur) * sample_rate / samples_per_frame)))
            zb = z[b, :, :t_valid]
            embs.append(torch.cat([zb.mean(dim=-1), zb.std(dim=-1, unbiased=False)], dim=0))
        e = torch.stack(embs, dim=0)  # (B,2d)
        n_samples += e.size(0)
        if log_name and (i + 1) % 50 == 0:
            _log_progress(log_name, n_samples, start_t)
        yield e.cpu(), batch["meta"]
    if log_name:
        _log_progress(log_name, n_samples, start_t)


def embedding_stats(x: torch.Tensor) -> Dict[str, Any]:
    """Collapse gauge for pooled utterance embeddings.

    Computes the participation-ratio effective rank (sum(lambda))^2 /
    sum(lambda^2) of the embedding covariance over samples, with eigenvalues
    clamped to >= 0. A healthy embedding space has effective rank well above
    1; utterance-level collapse shows up as a value near 0-1 even when probe
    accuracy has not moved yet.
    """
    x = x.detach().float()
    n, d = int(x.size(0)), int(x.size(1))
    stats: Dict[str, Any] = {"embed_dim": d, "embed_num_samples": n}
    if n < 2:
        stats["embed_effective_rank"] = 0.0
        return stats
    xc = (x - x.mean(dim=0, keepdim=True)).double()
    cov = (xc.t() @ xc) / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if total.item() <= 0.0:
        stats["embed_effective_rank"] = 0.0
        return stats
    stats["embed_effective_rank"] = float((total * total / eigvals.square().sum()).item())
    return stats


def checkpoint_step(ckpt_path: str) -> int | None:
    """Best-effort read of the training step stored in a checkpoint payload."""
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        step = state.get("step") if isinstance(state, dict) else None
        return int(step) if step is not None else None
    except Exception:
        return None
