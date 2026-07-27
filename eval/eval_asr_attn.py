from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from jiwer import cer, wer

from data_loading import resolve_manifest_root
from eval.common import amp_enabled, load_frozen_encoder
from eval.eval_asr import _filter_manifest_by_duration, _load_feats_and_text
from eval.repr_bench import TARGET_SR, _resample, build_embedder, model_spec

# Special-token indices — these must not collide with the CTC probe's vocab
# (CTC uses index 0 for <blank>; here index 0 is <pad>).
PAD_IDX: int = 0
BOS_IDX: int = 1
EOS_IDX: int = 2


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_attn_vocab(texts: List[str]) -> List[str]:
    """Return the full vocabulary list: special tokens then sorted characters.

    Index mapping: 0=<pad>, 1=<bos>, 2=<eos>, 3…=sorted chars.
    ``\\n`` is excluded from the character set (matches build_charset policy).
    """
    chars = sorted({c for t in texts for c in t.lower() if c != "\n"})
    return ["<pad>", "<bos>", "<eos>"] + chars


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Add fixed sinusoidal positional encoding to ``(B, L, d_model)`` inputs."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as a non-trainable buffer so it moves with the module.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Attention decoder head
# ---------------------------------------------------------------------------

class AttnDecoderHead(nn.Module):
    """Small Transformer decoder over frozen frame features.

    The encoder memory is projected from ``feat_dim`` to ``d_model`` and then
    consumed by a stack of cross-attention layers.  An autoregressive causal
    mask is applied to the target side.

    Unlike the CTC head there is **no** ``T >= L`` constraint: the decoder can
    emit any number of tokens regardless of the number of input frames, which
    is the key diagnostic property this probe exploits.
    """

    def __init__(
        self,
        feat_dim: int,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos = SinusoidalPE(d_model)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_ff, dropout, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def encode_memory(self, feats: torch.Tensor) -> torch.Tensor:
        """Project and add PE to encoder frame features.

        Args:
            feats: ``(B, T, feat_dim)`` — frozen encoder outputs on GPU.

        Returns:
            ``(B, T, d_model)`` memory tensor.
        """
        return self.pos(self.in_proj(feats))

    def forward(
        self,
        memory: torch.Tensor,
        mem_kpm: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_kpm: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass of the decoder.

        Args:
            memory:  ``(B, T, d_model)`` — projected encoder frames.
            mem_kpm: ``(B, T)`` bool — True where the memory position is
                     padding and should be ignored (PyTorch convention).
            tgt_in:  ``(B, Lt)`` long — BOS-prefixed token ids.
            tgt_kpm: ``(B, Lt)`` bool — True where tgt_in is PAD_IDX.

        Returns:
            logits ``(B, Lt, vocab_size)``.
        """
        tgt = self.pos(self.tok_emb(tgt_in))  # (B, Lt, d_model)
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1)
        ).to(tgt.device)
        h = self.dec(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
        )
        return self.out(h)  # (B, Lt, vocab_size)


# ---------------------------------------------------------------------------
# Greedy autoregressive decoding
# ---------------------------------------------------------------------------

def _decode(
    head: AttnDecoderHead,
    feats: torch.Tensor,
    lens: torch.Tensor,
    texts: List[str],
    id2tok: List[str],
    device: torch.device,
    max_decode_len: int = 200,
    eval_batch: int = 128,
) -> Dict[str, Any]:
    """Greedy autoregressive decode over a full feature tensor.

    Features are stored on CPU and moved to ``device`` in slices of
    ``eval_batch`` to avoid OOM.

    Args:
        head:           The trained ``AttnDecoderHead`` (set to eval mode before
                        calling).
        feats:          ``(N, T, D)`` CPU tensor — full feature set.
        lens:           ``(N,)`` CPU long tensor — valid frame counts.
        texts:          Reference transcripts for metric computation.
        id2tok:         Vocabulary list (index → token string).
        device:         GPU/CPU device for inference.
        max_decode_len: Maximum number of autoregressive steps per sample.
        eval_batch:     Samples per inference slice.

    Returns:
        Dict with keys ``wer``, ``cer``, ``num_samples``, ``examples``.
    """
    head.eval()
    all_hyps: List[str] = []

    with torch.no_grad():
        for start in range(0, feats.size(0), eval_batch):
            xb = feats[start : start + eval_batch].to(device)   # (B, T, D)
            vl = lens[start : start + eval_batch].to(device)    # (B,)
            B, T, _ = xb.shape

            # Build memory and its padding mask once per slice.
            memory = head.encode_memory(xb)  # (B, T, d_model)
            mem_kpm = (
                torch.arange(T, device=device)[None, :] >= vl[:, None]
            )  # (B, T) bool — True where frame is padding

            # Start all sequences with BOS.
            ys = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_decode_len):
                tgt_kpm = (ys == PAD_IDX)  # (B, Lt)
                logits = head(memory, mem_kpm, ys, tgt_kpm)  # (B, Lt, V)
                next_tok = logits[:, -1].argmax(dim=-1)       # (B,)
                # Force already-finished rows to emit PAD (neutral).
                next_tok = next_tok.masked_fill(finished, PAD_IDX)
                ys = torch.cat([ys, next_tok[:, None]], dim=1)
                finished = finished | (next_tok == EOS_IDX)
                if finished.all():
                    break

            # Convert token id sequences to strings.
            for row in ys.cpu().tolist():
                # Drop leading BOS, cut at first EOS.
                row = row[1:]  # remove BOS
                hyp_ids: List[int] = []
                for tok_id in row:
                    if tok_id == EOS_IDX:
                        break
                    hyp_ids.append(tok_id)
                # Skip special tokens when mapping to characters.
                skip = {PAD_IDX, BOS_IDX, EOS_IDX}
                all_hyps.append("".join(id2tok[i] for i in hyp_ids if i not in skip))

    w = wer(texts, all_hyps)
    c = cer(texts, all_hyps)
    examples: List[Dict[str, str]] = []
    for i in range(min(5, len(texts))):
        examples.append({"ref": texts[i], "hyp": all_hyps[i]})
    return {"wer": float(w), "cer": float(c), "num_samples": len(texts), "examples": examples}


def _decode_cached_loader(
    head: AttnDecoderHead,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor, List[str]]],
    id2tok: List[str],
    device: torch.device,
    max_decode_len: int = 200,
) -> Dict[str, Any]:
    """Greedily decode a disk-backed feature ``DataLoader`` batch by batch."""
    head.eval()
    all_texts: List[str] = []
    all_hyps: List[str] = []
    skip = {PAD_IDX, BOS_IDX, EOS_IDX}
    with torch.no_grad():
        for xb_cpu, vl_cpu, texts in loader:
            xb = xb_cpu.to(device)
            vl = vl_cpu.to(device)
            B, T, _ = xb.shape
            memory = head.encode_memory(xb)
            mem_kpm = torch.arange(T, device=device)[None, :] >= vl[:, None]
            ys = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_decode_len):
                logits = head(memory, mem_kpm, ys, ys == PAD_IDX)
                next_tok = logits[:, -1].argmax(dim=-1).masked_fill(finished, PAD_IDX)
                ys = torch.cat([ys, next_tok[:, None]], dim=1)
                finished = finished | (next_tok == EOS_IDX)
                if finished.all():
                    break
            for row in ys.cpu().tolist():
                hyp_ids: List[int] = []
                for tok_id in row[1:]:
                    if tok_id == EOS_IDX:
                        break
                    hyp_ids.append(tok_id)
                all_hyps.append("".join(id2tok[i] for i in hyp_ids if i not in skip))
            all_texts.extend(texts)
    examples = [
        {"ref": ref, "hyp": hyp}
        for ref, hyp in zip(all_texts[:5], all_hyps[:5])
    ]
    return {
        "wer": float(wer(all_texts, all_hyps)),
        "cer": float(cer(all_texts, all_hyps)),
        "num_samples": len(all_texts),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Batch collation helpers for seq2seq training
# ---------------------------------------------------------------------------

def _make_batch_targets(
    target_ids: List[List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shifted input/output target tensors for a training batch.

    For each sample:
        tgt_in  = [BOS] + ids          (length Lt = len(ids) + 1)
        tgt_out = ids   + [EOS]        (length Lt)

    Both are right-padded with PAD_IDX to the batch maximum length.

    Returns:
        tgt_in:  ``(B, Lt_max)`` long
        tgt_out: ``(B, Lt_max)`` long
        tgt_kpm: ``(B, Lt_max)`` bool — True where tgt_in is PAD_IDX
    """
    max_len = max(len(ids) + 1 for ids in target_ids)
    B = len(target_ids)
    tgt_in = torch.full((B, max_len), PAD_IDX, dtype=torch.long, device=device)
    tgt_out = torch.full((B, max_len), PAD_IDX, dtype=torch.long, device=device)
    for i, ids in enumerate(target_ids):
        L = len(ids)
        tgt_in[i, 0] = BOS_IDX
        if L > 0:
            tgt_in[i, 1 : L + 1] = torch.tensor(ids, dtype=torch.long, device=device)
            tgt_out[i, 0:L] = torch.tensor(ids, dtype=torch.long, device=device)
        tgt_out[i, L] = EOS_IDX
    tgt_kpm = tgt_in == PAD_IDX  # (B, Lt_max) — True where padding
    return tgt_in, tgt_out, tgt_kpm


@dataclass
class _ExternalFeatureCache:
    """Small index for a feature cache whose frame tensors remain on disk."""

    root: pathlib.Path
    records: List[Dict[str, Any]]
    feature_dim: int

    @property
    def texts(self) -> List[str]:
        return [str(record["text"]) for record in self.records]


class _CachedFeatureDataset(Dataset[Tuple[torch.Tensor, str]]):
    """Map-style view that loads one cached, variable-length feature tensor."""

    def __init__(self, cache: _ExternalFeatureCache) -> None:
        self.cache = cache

    def __len__(self) -> int:
        return len(self.cache.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        record = self.cache.records[index]
        # Cache files are created locally by this evaluator, not supplied as
        # arbitrary checkpoints; weights_only still avoids pickle execution.
        frames = torch.load(
            self.cache.root / str(record["feature"]), map_location="cpu", weights_only=True
        )
        if not isinstance(frames, torch.Tensor) or frames.ndim != 2:
            raise RuntimeError(f"Invalid cached ASR feature file: {record['feature']}")
        return frames.float(), str(record["text"])


def _collate_cached_features(
    batch: List[Tuple[torch.Tensor, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    frames, texts = zip(*batch)
    lens = torch.tensor([frame.size(0) for frame in frames], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(list(frames), batch_first=True), lens, list(texts)


def _cache_config(
    model_name: str,
    manifest: str,
    *,
    text_key: str,
    max_samples: int,
    segment_seconds: float,
    max_utt_seconds: float,
) -> Dict[str, Any]:
    stat = pathlib.Path(manifest).resolve().stat()
    return {
        "format": 1,
        "model": model_name,
        "manifest": str(pathlib.Path(manifest).resolve()),
        "manifest_size": stat.st_size,
        "manifest_mtime_ns": stat.st_mtime_ns,
        "text_key": text_key,
        "max_samples": max_samples,
        "segment_seconds": segment_seconds,
        "max_utt_seconds": max_utt_seconds,
    }


def _read_cache_records(index_path: pathlib.Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        return []
    with index_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_cache_metadata(path: pathlib.Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _cache_external_features(
    model_name: str,
    manifest: str,
    *,
    text_key: str,
    max_samples: int,
    segment_seconds: float,
    max_utt_seconds: float,
    embedder: Any,
    cache_dir: pathlib.Path,
    split_name: str,
) -> Tuple[_ExternalFeatureCache, int, int]:
    """Stream a manifest into a resumable disk cache without retaining frames.

    ``DataLoader`` later reads these tensors individually and pads only the
    current batch.  The JSONL index intentionally contains only lightweight
    paths/transcripts (at most ``max_samples`` entries), never frame arrays or
    the full source manifest.
    """
    import torchaudio

    spec = model_spec(model_name)
    if not spec.supports_asr_probe:
        raise ValueError(f"{model_name} exposes utterance-only features and cannot be used for ASR")

    config = _cache_config(
        model_name, manifest, text_key=text_key, max_samples=max_samples,
        segment_seconds=segment_seconds, max_utt_seconds=max_utt_seconds,
    )
    meta_path = cache_dir / "metadata.json"
    index_path = cache_dir / "index.jsonl"
    features_dir = cache_dir / "features"
    if cache_dir.exists():
        if not meta_path.exists():
            raise RuntimeError(
                f"Feature cache {cache_dir} has no metadata; remove it or choose --feature_cache_dir"
            )
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        previous_config = {key: previous.get(key) for key in config}
        if previous_config != config:
            raise RuntimeError(
                f"Feature cache {cache_dir} was made for different inputs; remove it or choose "
                "--feature_cache_dir"
            )
    else:
        cache_dir.mkdir(parents=True)
        features_dir.mkdir()
        _write_cache_metadata(meta_path, {**config, "complete": False, "feature_dim": None})

    records = _read_cache_records(index_path)
    existing = len(records)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if bool(metadata.get("complete")):
        if not records:
            raise RuntimeError(f"Completed feature cache {cache_dir} has no records")
        feature_dim = int(metadata["feature_dim"])
        print(f"  [ASR-ATTN] Reusing {split_name} feature cache: {existing} samples at {cache_dir}", flush=True)
        return (
            _ExternalFeatureCache(cache_dir, records, feature_dim),
            int(metadata.get("n_filtered", 0)),
            int(metadata.get("n_unknown_duration", 0)),
        )

    features_dir.mkdir(exist_ok=True)
    root: pathlib.Path | None = None
    kept = dropped = unknown_duration = truncated = accepted = 0
    feature_dim: int | None = None
    if records:
        cached = torch.load(cache_dir / str(records[0]["feature"]), map_location="cpu", weights_only=True)
        if not isinstance(cached, torch.Tensor) or cached.ndim != 2:
            raise RuntimeError(f"Invalid cached ASR feature file: {records[0]['feature']}")
        feature_dim = int(cached.size(1))
    # Resume interrupted extraction by scanning rows again but retaining no
    # waveform/features for the already-written prefix.
    with open(manifest, encoding="utf-8") as f, index_path.open("a", encoding="utf-8") as index_file:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            duration = row.get("duration")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
                unknown_duration += 1
            elif duration > max_utt_seconds:
                dropped += 1
                continue
            text = row.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            raw_path = row.get("audio_filepath")
            if not isinstance(raw_path, str):
                raise ValueError(f"Manifest row has no string audio_filepath: {row}")
            if os.path.isabs(raw_path):
                path = raw_path
            else:
                # Do not lock the root to the manifest parent merely because
                # an earlier row happened to use an absolute path.
                if root is None:
                    root = resolve_manifest_root(manifest, [row])
                path = str(root / raw_path)
            kept += 1
            if accepted < existing:
                accepted += 1
                continue
            if max_samples and accepted >= max_samples:
                break
            info_fn = getattr(torchaudio, "info", None)
            if info_fn is not None:
                info = info_fn(path)
                max_frames = math.ceil(segment_seconds * info.sample_rate)
                wav, sr = torchaudio.load(path, num_frames=max_frames)
                truncated += int(info.num_frames > max_frames)
            else:
                # Some torchaudio builds expose ``load`` but not ``info``.
                # Crop before resampling and WavLM inference in that case too.
                wav, sr = torchaudio.load(path)
                max_frames = math.ceil(segment_seconds * sr)
                truncated += int(wav.size(-1) > max_frames)
                wav = wav[..., :max_frames]
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav16k = _resample(wav.squeeze(0), int(sr), TARGET_SR)
            frames = torch.from_numpy(embedder.fn(wav16k)).float().contiguous()
            if frames.ndim != 2 or frames.size(0) < 1:
                raise RuntimeError(f"{model_name} returned invalid ASR feature shape {tuple(frames.shape)}")
            if feature_dim is None:
                feature_dim = frames.size(1)
            elif frames.size(1) != feature_dim:
                raise RuntimeError(f"{model_name} changed feature dimension within one run")
            feature_rel = pathlib.Path("features") / f"{accepted:08d}.pt"
            feature_path = cache_dir / feature_rel
            temp_path = feature_path.with_suffix(".tmp")
            torch.save(frames, temp_path)
            temp_path.replace(feature_path)
            record = {"feature": str(feature_rel), "text": text, "source": path}
            index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            index_file.flush()
            records.append(record)
            accepted += 1
            if accepted % 50 == 0:
                print(
                    f"  [ASR-ATTN] {model_name}: cached {accepted} {split_name} features "
                    f"({truncated} clipped to {segment_seconds:g}s)",
                    flush=True,
                )

    if not records:
        raise ValueError(f"No usable transcript/audio rows in {manifest}")
    if feature_dim is None:
        cached = torch.load(cache_dir / str(records[0]["feature"]), map_location="cpu", weights_only=True)
        if not isinstance(cached, torch.Tensor) or cached.ndim != 2:
            raise RuntimeError(f"Invalid cached ASR feature file: {records[0]['feature']}")
        feature_dim = int(cached.size(1))
    _write_cache_metadata(
        meta_path,
        {
            **config,
            "complete": True,
            "feature_dim": feature_dim,
            "n_filtered": dropped,
            "n_unknown_duration": unknown_duration,
        },
    )
    print(
        f"  [ASR-ATTN] {model_name}: cached {len(records)} {split_name} features total "
        f"({truncated} clipped to {segment_seconds:g}s; {dropped} known-long rows skipped; "
        f"{unknown_duration} unknown durations scanned)",
        flush=True,
    )
    return _ExternalFeatureCache(cache_dir, records, feature_dim), dropped, unknown_duration


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attention seq2seq ASR probe — diagnostic counterpart to eval_asr.py"
    )
    # Core / shared with eval_asr.py
    ap.add_argument("--model", default="ours", help="repr_bench adapter name (default: ours)")
    ap.add_argument("--config", default=None, help="Required for --model ours")
    ap.add_argument("--ckpt", default=None, help="CLAE checkpoint; ignored by external adapters")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--segment_seconds", type=float, default=None,
        help=(
            "Maximum audio duration passed to the feature extractor "
            "(also defaults --max_utt_seconds). External adapters hard-crop "
            "audio even when manifest durations are missing."
        ),
    )
    ap.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap train/dev samples (0=unlimited)"
    )
    ap.add_argument(
        "--feature_cache_dir", default=None,
        help=(
            "Directory for external-adapter frame tensors (default: derived from --out). "
            "The cache is resumable and lets training/evaluation stream batches from disk."
        ),
    )
    ap.add_argument(
        "--num_workers", type=int, default=0,
        help="Feature-cache DataLoader workers (default: 0; safer for WSL and HDD-backed audio).",
    )
    ap.add_argument(
        "--max_utt_seconds", type=float, default=None,
        help="Drop manifest rows longer than this (default: effective segment_seconds)"
    )
    ap.add_argument(
        "--chunk_seconds", type=float, default=None,
        help=(
            "Encode audio in independent windows of this length and concatenate "
            "features (default: pretraining data.segment_seconds; <=0 disables)."
        ),
    )
    ap.add_argument(
        "--features", choices=["encoder", "frontend", "mel"], default="encoder",
        help=(
            "encoder: the model under test; frontend: conv frontend only; "
            "mel: log-mel fbank control bypassing the model"
        ),
    )
    ap.add_argument(
        "--mel_hop", type=int, default=320,
        help="mel hop in samples (with --features mel): 320=50 Hz; 1280=12.5 Hz",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=True)
    # Decoder hyperparameters (new, not in eval_asr.py)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument(
        "--max_decode_len", type=int, default=200,
        help="Maximum number of autoregressive decode steps per utterance",
    )
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.num_workers < 0:
        ap.error("--num_workers must be non-negative")

    # ``ours_random`` deliberately takes the shared adapter route so it keeps
    # freshly initialized weights instead of accidentally loading the checkpoint.
    is_clae = args.model == "ours"
    lm = None
    adapter: Any = None
    if is_clae:
        if not args.config or not args.ckpt:
            ap.error("--config and --ckpt are required for --model ours")
        lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
        seg = args.segment_seconds if args.segment_seconds is not None else lm.cfg.eval.asr.segment_seconds
        chunk = args.chunk_seconds if args.chunk_seconds is not None else lm.cfg.data.segment_seconds
    else:
        if args.features != "encoder":
            ap.error("external --model adapters support --features encoder only")
        spec = model_spec(args.model)
        if not spec.supports_asr_probe:
            ap.error(f"{args.model} exposes utterance-only features and is not an ASR baseline")
        seg = args.segment_seconds if args.segment_seconds is not None else 15.0
        chunk = None
        if args.model == "ours_random" and not args.ckpt:
            ap.error("--ckpt is required for --model ours_random to recover its architecture")
    max_utt = args.max_utt_seconds if args.max_utt_seconds is not None else seg
    if chunk is not None and chunk <= 0:
        chunk = None
    print(
        f"  [ASR-ATTN] segment_seconds={seg:g}, max_utt_seconds={max_utt:g}, "
        f"chunk_seconds={'off' if chunk is None else f'{chunk:g}'}, "
        f"features={args.features} model={args.model}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Filter manifests for CLAE. External adapters apply the duration
    #    check while streaming into their cache, so they never materialize a
    #    second full JSONL file or list of all manifest rows.
    # ------------------------------------------------------------------
    out_path = pathlib.Path(args.out)
    if is_clae:
        train_manifest, _, n_filtered_tr, n_unknown_tr = _filter_manifest_by_duration(
            args.train_manifest,
            max_utt,
            out_path.with_suffix(".attn.train_filtered.jsonl"),
            "Filter train",
        )
        dev_manifest, _, n_filtered_de, n_unknown_de = _filter_manifest_by_duration(
            args.dev_manifest,
            max_utt,
            out_path.with_suffix(".attn.dev_filtered.jsonl"),
            "Filter dev",
        )
    else:
        train_manifest, dev_manifest = args.train_manifest, args.dev_manifest
        n_filtered_tr = n_filtered_de = n_unknown_tr = n_unknown_de = 0

    # ------------------------------------------------------------------
    # 4. Dry-run: pull one batch, write shape info, exit
    # ------------------------------------------------------------------
    if args.dry_run:
        if is_clae:
            from eval.common import iter_frame_features
            feats, _, meta = next(iter_frame_features(
                lm, train_manifest, sample_rate=lm.cfg.data.sample_rate,
                segment_seconds=seg, batch_size=args.batch_size, chunk_seconds=chunk,
                source=args.features, mel_hop=args.mel_hop,
            ))
        else:
            adapter = build_embedder(args.model, ckpt=args.ckpt)
            dry_cache, _, _ = _cache_external_features(
                args.model, train_manifest, text_key=args.text_key, max_samples=1,
                segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter,
                cache_dir=out_path.with_suffix(".attn_dryrun_features"), split_name="dry-run",
            )
            frame, text = _CachedFeatureDataset(dry_cache)[0]
            feats, meta = frame.unsqueeze(0), [text]
        out = {
            "dry_run": True,
            "feats_shape": list(feats.shape),
            "num_samples": len(meta),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ------------------------------------------------------------------
    # 5. Extract features. CLAE retains its legacy in-memory path; external
    #    adapters cache one frame tensor per utterance and stream it through
    #    DataLoaders below. This keeps RAM proportional to batch size.
    # ------------------------------------------------------------------
    max_s = args.max_samples
    print(
        f"  [ASR-ATTN] Extracting train features{f' (max {max_s})' if max_s else ''}...",
        flush=True,
    )
    if is_clae:
        feats_tr, lens_tr, text_tr = _load_feats_and_text(
            lm, train_manifest, text_key=args.text_key, batch_size=args.batch_size,
            segment_seconds=seg, chunk_seconds=chunk, source=args.features,
            mel_hop=args.mel_hop, log_name="ASR-ATTN train", max_samples=max_s,
        )
    else:
        cache_root = (
            pathlib.Path(args.feature_cache_dir)
            if args.feature_cache_dir
            else out_path.with_suffix(".attn_features")
        )
        # One frozen adapter serves both splits. Re-instantiating WavLM for
        # dev extraction needlessly reloads 95M parameters onto the GPU.
        adapter = build_embedder(args.model, ckpt=args.ckpt)
        train_cache, n_filtered_tr, n_unknown_tr = _cache_external_features(
            args.model, train_manifest, text_key=args.text_key, max_samples=max_s,
            segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter, cache_dir=cache_root / "train",
            split_name="train",
        )
    print(
        f"  [ASR-ATTN] Extracting dev features{f' (max {max_s})' if max_s else ''}...",
        flush=True,
    )
    if is_clae:
        feats_de, lens_de, text_de = _load_feats_and_text(
            lm, dev_manifest, text_key=args.text_key, batch_size=args.batch_size,
            segment_seconds=seg, chunk_seconds=chunk, source=args.features,
            mel_hop=args.mel_hop, log_name="ASR-ATTN dev", max_samples=max_s,
        )
    else:
        dev_cache, n_filtered_de, n_unknown_de = _cache_external_features(
            args.model, dev_manifest, text_key=args.text_key, max_samples=max_s,
            segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter, cache_dir=cache_root / "dev",
            split_name="dev",
        )

    # ------------------------------------------------------------------
    # 6. Free the frozen encoder to reclaim GPU memory
    # ------------------------------------------------------------------
    if lm is not None or adapter is not None:
        del lm, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_clae:
        print(f"  [ASR-ATTN] Train: {feats_tr.shape}, Dev: {feats_de.shape}", flush=True)
    else:
        print(
            f"  [ASR-ATTN] Train: {len(train_cache.records)} disk-cached features "
            f"(D={train_cache.feature_dim}), Dev: {len(dev_cache.records)} disk-cached features "
            f"(D={dev_cache.feature_dim})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 7. Build / load attention vocab (DISTINCT cache from CTC .charset.json)
    # ------------------------------------------------------------------
    charset_path = pathlib.Path(args.train_manifest + ".charset_attn.json")
    if charset_path.exists():
        vocab_list: List[str] = json.loads(charset_path.read_text(encoding="utf-8"))
        print(
            f"  [ASR-ATTN] Loaded cached charset ({len(vocab_list)} symbols) from {charset_path}",
            flush=True,
        )
    else:
        vocab_list = build_attn_vocab(text_tr if is_clae else train_cache.texts)
        charset_path.write_text(
            json.dumps(vocab_list, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"  [ASR-ATTN] Built and cached charset ({len(vocab_list)} symbols) at {charset_path}",
            flush=True,
        )
    vocab: Dict[str, int] = {c: i for i, c in enumerate(vocab_list)}
    id2tok: List[str] = vocab_list

    # ------------------------------------------------------------------
    # 8. Encode train transcripts to id lists
    #    (No infeasibility filtering — attention has no T >= L constraint)
    # ------------------------------------------------------------------
    targets_tr: List[List[int]] = (
        [[vocab[c] for c in t.lower() if c in vocab] for t in text_tr]
        if is_clae else []
    )

    # ------------------------------------------------------------------
    # 9. Build model, optimizer, loss
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Features stay on CPU — only per-batch slices are moved to GPU.

    head = AttnDecoderHead(
        feat_dim=feats_tr.size(-1) if is_clae else train_cache.feature_dim,
        vocab_size=len(vocab_list),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.dec_layers,
        dim_ff=args.dim_ff,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    use_amp = amp_enabled(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    head.train()
    t0 = time.perf_counter()
    log_interval = max(1, args.steps // 10)

    if is_clae:
        N = feats_tr.size(0)
    else:
        train_dataset = _CachedFeatureDataset(train_cache)
        sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=args.steps * args.batch_size
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=_collate_cached_features,
        )
        train_iter = iter(train_loader)

    for step_i in range(args.steps):
        if is_clae:
            idx = torch.randint(0, N, (args.batch_size,))
            xb = feats_tr[idx].to(device)   # (B, T, D) — only batch on GPU
            vl = lens_tr[idx].to(device)    # (B,)
            batch_ids = [targets_tr[int(i)] for i in idx.tolist()]
        else:
            xb_cpu, vl_cpu, batch_texts = next(train_iter)
            xb = xb_cpu.to(device)
            vl = vl_cpu.to(device)
            batch_ids = [[vocab[c] for c in text.lower() if c in vocab] for text in batch_texts]
        T = xb.size(1)

        # Memory padding mask: True where frame index >= valid length.
        mem_kpm = torch.arange(T, device=device)[None, :] >= vl[:, None]  # (B, T)

        # Build teacher-forced targets for this batch.
        tgt_in, tgt_out, tgt_kpm = _make_batch_targets(batch_ids, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            memory = head.encode_memory(xb)
            logits = head(memory, mem_kpm, tgt_in, tgt_kpm)  # (B, Lt, V)

        # Cross-entropy in fp32 (same discipline as CTC probe's log-softmax).
        V = logits.size(-1)
        loss = loss_fn(logits.float().reshape(-1, V), tgt_out.reshape(-1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (step_i + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            rate = (step_i + 1) / elapsed
            eta = (args.steps - step_i - 1) / rate
            print(
                f"  [ASR-ATTN] step {step_i + 1}/{args.steps}  loss={loss.item():.4f}  "
                f"({rate:.0f} steps/s, ETA {eta:.0f}s)",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 11. Evaluate (greedy autoregressive decode)
    # ------------------------------------------------------------------
    print("  [ASR-ATTN] Evaluating...", flush=True)
    if is_clae:
        result_tr = _decode(
            head, feats_tr, lens_tr, text_tr, id2tok, device,
            max_decode_len=args.max_decode_len,
        )
        result_de = _decode(
            head, feats_de, lens_de, text_de, id2tok, device,
            max_decode_len=args.max_decode_len,
        )
    else:
        train_eval_loader = DataLoader(
            _CachedFeatureDataset(train_cache), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, collate_fn=_collate_cached_features,
        )
        dev_eval_loader = DataLoader(
            _CachedFeatureDataset(dev_cache), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, collate_fn=_collate_cached_features,
        )
        result_tr = _decode_cached_loader(
            head, train_eval_loader, id2tok, device, max_decode_len=args.max_decode_len,
        )
        result_de = _decode_cached_loader(
            head, dev_eval_loader, id2tok, device, max_decode_len=args.max_decode_len,
        )

    # ------------------------------------------------------------------
    # 12. Write output JSON
    # ------------------------------------------------------------------
    print(
        f"  [ASR-ATTN] Train WER: {result_tr['wer']:.4f} CER: {result_tr['cer']:.4f}, "
        f"Dev WER: {result_de['wer']:.4f} CER: {result_de['cer']:.4f}",
        flush=True,
    )
    out_data: Dict[str, Any] = {
        "train": result_tr,
        "dev": result_de,
        "vocab_size": len(vocab_list),
        "max_utt_seconds": float(max_utt),
        "chunk_seconds": chunk,
        "features": args.features,
        "model": args.model,
        "feature_cache_dir": str(cache_root) if not is_clae else None,
        "num_workers": args.num_workers if not is_clae else None,
        "mel_hop": args.mel_hop if args.features == "mel" else None,
        "n_filtered_train": n_filtered_tr,
        "n_filtered_dev": n_filtered_de,
        "n_unknown_duration_train": n_unknown_tr,
        "n_unknown_duration_dev": n_unknown_de,
        "decoder": {
            "d_model": args.d_model,
            "layers": args.dec_layers,
            "nhead": args.nhead,
            "dim_ff": args.dim_ff,
            "max_decode_len": args.max_decode_len,
        },
        "ctc_free": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
