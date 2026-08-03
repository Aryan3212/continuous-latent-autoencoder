from __future__ import annotations

import argparse
import hashlib
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
from eval.common import amp_enabled, iter_frame_features, load_frozen_encoder
from eval.eval_asr import _filter_manifest_by_duration
from eval.repr_bench import (
    RANDOM_BASELINE_SEED,
    TARGET_SR,
    _resample,
    build_embedder,
    model_spec,
)

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
            xb = xb_cpu.to(device, non_blocking=True)
            vl = vl_cpu.to(device, non_blocking=True)
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
class _FeatureCache:
    """Small index for a feature cache whose frame tensors remain on disk."""

    root: pathlib.Path
    records: List[Dict[str, Any]]
    feature_dim: int

    @property
    def texts(self) -> List[str]:
        return [str(record["text"]) for record in self.records]


class _CachedFeatureDataset(Dataset[Tuple[torch.Tensor, str]]):
    """Map-style view that loads one cached, variable-length feature tensor."""

    def __init__(self, cache: _FeatureCache) -> None:
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
    extractor_identity: Dict[str, Any],
) -> Dict[str, Any]:
    stat = pathlib.Path(manifest).resolve().stat()
    return {
        "format": 2,
        "model": model_name,
        "manifest": str(pathlib.Path(manifest).resolve()),
        "manifest_size": stat.st_size,
        "manifest_mtime_ns": stat.st_mtime_ns,
        "text_key": text_key,
        "max_samples": max_samples,
        "segment_seconds": segment_seconds,
        "max_utt_seconds": max_utt_seconds,
        "extractor": extractor_identity,
    }


def _file_identity(path: str) -> Dict[str, Any]:
    resolved = pathlib.Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _config_fingerprint(config: Any) -> str:
    """Hash the fully resolved config, including inherited bases and overrides."""
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reset_probe_seed(seed: int) -> None:
    """Make probe initialization independent of cache-hit extraction work."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_cache_records(index_path: pathlib.Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        return []
    with index_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_cache_metadata(path: pathlib.Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _source_record(path: str) -> Dict[str, Any]:
    stat = pathlib.Path(path).stat()
    return {
        "source": path,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _validate_cache_records(
    cache_dir: pathlib.Path,
    records: List[Dict[str, Any]],
) -> None:
    """Fail before training if cached features or their source audio changed."""
    for index, record in enumerate(records):
        expected_feature = pathlib.Path("features") / f"{index:08d}.pt"
        if record.get("feature") != str(expected_feature):
            raise RuntimeError(
                f"Feature cache {cache_dir} has a non-contiguous index at row {index}"
            )
        feature_path = cache_dir / expected_feature
        if not feature_path.is_file() or feature_path.stat().st_size <= 0:
            raise RuntimeError(f"Feature cache file is missing/empty: {feature_path}")
        source = record.get("source")
        if not isinstance(source, str):
            raise RuntimeError(f"Feature cache row {index} has no source path")
        stat = pathlib.Path(source).stat()
        if (
            stat.st_size != record.get("source_size")
            or stat.st_mtime_ns != record.get("source_mtime_ns")
        ):
            raise RuntimeError(
                f"Source audio changed after caching: {source}; remove the "
                "feature cache and rerun extraction"
            )


def _cache_external_features(
    model_name: str,
    manifest: str,
    *,
    text_key: str,
    max_samples: int,
    segment_seconds: float,
    max_utt_seconds: float,
    embedder: Any,
    extractor_identity: Dict[str, Any],
    cache_dir: pathlib.Path,
    split_name: str,
    n_filtered: int,
    n_unknown_duration: int,
) -> Tuple[_FeatureCache, int, int]:
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
        extractor_identity={
            **extractor_identity,
            "n_filtered": n_filtered,
            "n_unknown_duration": n_unknown_duration,
        },
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
    _validate_cache_records(cache_dir, records)
    existing = len(records)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if bool(metadata.get("complete")):
        if not records:
            raise RuntimeError(f"Completed feature cache {cache_dir} has no records")
        feature_dim = int(metadata["feature_dim"])
        print(f"  [ASR-ATTN] Reusing {split_name} feature cache: {existing} samples at {cache_dir}", flush=True)
        return (
            _FeatureCache(cache_dir, records, feature_dim),
            int(metadata.get("n_filtered", 0)),
            int(metadata.get("n_unknown_duration", 0)),
        )

    features_dir.mkdir(exist_ok=True)
    root: pathlib.Path | None = None
    truncated = accepted = 0
    dropped = n_filtered
    unknown_duration = n_unknown_duration
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
                raise RuntimeError(
                    f"Filtered ASR manifest contains invalid duration: {row}"
                )
            elif duration > max_utt_seconds:
                raise RuntimeError(
                    f"Filtered ASR manifest contains overlong audio: {row}"
                )
            text = row.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    f"Filtered ASR manifest contains invalid {text_key!r} text"
                )
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
            if accepted < existing:
                accepted += 1
                continue
            if max_samples and accepted >= max_samples:
                break
            info_fn = getattr(torchaudio, "info", None)
            if info_fn is not None:
                info = info_fn(path)
                if info.sample_rate <= 0 or info.num_frames <= 0:
                    raise ValueError(f"Invalid audio metadata for {path}")
                actual_duration = info.num_frames / info.sample_rate
                if actual_duration > max_utt_seconds:
                    raise RuntimeError(
                        f"Audio changed after duration filtering: {path}"
                    )
                max_frames = math.ceil(segment_seconds * info.sample_rate)
                wav, sr = torchaudio.load(path, num_frames=max_frames)
                truncated += int(info.num_frames > max_frames)
            else:
                # Some torchaudio builds expose ``load`` but not ``info``.
                wav, sr = torchaudio.load(path)
                if sr <= 0 or wav.size(-1) <= 0:
                    raise ValueError(f"Invalid audio data for {path}")
                actual_duration = wav.size(-1) / sr
                if actual_duration > max_utt_seconds:
                    raise RuntimeError(
                        f"Audio changed after duration filtering: {path}"
                    )
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
            record = {
                "feature": str(feature_rel),
                "text": text,
                **_source_record(path),
            }
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
    return _FeatureCache(cache_dir, records, feature_dim), dropped, unknown_duration


def _cache_clae_features(
    manifest: str,
    *,
    text_key: str,
    max_samples: int,
    segment_seconds: float,
    max_utt_seconds: float,
    lm: Any,
    chunk_seconds: float | None,
    source: str,
    mel_hop: int,
    extraction_batch_size: int,
    num_workers: int,
    extractor_identity: Dict[str, Any],
    cache_dir: pathlib.Path,
    split_name: str,
    n_filtered: int,
    n_unknown_duration: int,
) -> Tuple[_FeatureCache, int, int]:
    """Batch CLAE extraction into the same resumable variable-length cache."""
    config = _cache_config(
        "ours",
        manifest,
        text_key=text_key,
        max_samples=max_samples,
        segment_seconds=segment_seconds,
        max_utt_seconds=max_utt_seconds,
        extractor_identity={
            **extractor_identity,
            "n_filtered": n_filtered,
            "n_unknown_duration": n_unknown_duration,
        },
    )
    meta_path = cache_dir / "metadata.json"
    index_path = cache_dir / "index.jsonl"
    features_dir = cache_dir / "features"
    if cache_dir.exists():
        if not meta_path.exists():
            raise RuntimeError(
                f"Feature cache {cache_dir} has no metadata; remove it or "
                "choose --feature_cache_dir"
            )
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        previous_config = {key: previous.get(key) for key in config}
        if previous_config != config:
            raise RuntimeError(
                f"Feature cache {cache_dir} was made for different inputs; "
                "remove it or choose --feature_cache_dir"
            )
    else:
        cache_dir.mkdir(parents=True)
        features_dir.mkdir()
        _write_cache_metadata(
            meta_path,
            {**config, "complete": False, "feature_dim": None},
        )

    records = _read_cache_records(index_path)
    _validate_cache_records(cache_dir, records)
    existing = len(records)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if bool(metadata.get("complete")):
        if not records:
            raise RuntimeError(f"Completed feature cache {cache_dir} has no records")
        print(
            f"  [ASR-ATTN] Reusing {split_name} feature cache: "
            f"{existing} samples at {cache_dir}",
            flush=True,
        )
        return (
            _FeatureCache(cache_dir, records, int(metadata["feature_dim"])),
            int(metadata.get("n_filtered", 0)),
            int(metadata.get("n_unknown_duration", 0)),
        )

    features_dir.mkdir(exist_ok=True)
    feature_dim: int | None = None
    if records:
        cached = torch.load(
            cache_dir / str(records[0]["feature"]),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(cached, torch.Tensor) or cached.ndim != 2:
            raise RuntimeError(
                f"Invalid cached ASR feature file: {records[0]['feature']}"
            )
        feature_dim = int(cached.size(1))

    remaining = 0 if max_samples <= 0 else max(0, max_samples - existing)
    iterator: Iterable[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]
    if max_samples > 0 and remaining == 0:
        iterator = ()
    else:
        iterator = iter_frame_features(
            lm,
            manifest,
            sample_rate=lm.cfg.data.sample_rate,
            segment_seconds=segment_seconds,
            batch_size=extraction_batch_size,
            num_workers=num_workers,
            log_name=f"ASR-ATTN {split_name}",
            chunk_seconds=chunk_seconds,
            source=source,
            mel_hop=mel_hop,
            start_index=existing,
            max_samples=remaining,
        )
    accepted = existing
    with index_path.open("a", encoding="utf-8") as index_file:
        for feats, lens, meta in iterator:
            for index, row in enumerate(meta):
                text = row.get(text_key)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        f"Filtered ASR manifest contains invalid {text_key!r} text"
                    )
                valid_frames = int(lens[index])
                frames = feats[index, :valid_frames].float().contiguous()
                if frames.ndim != 2 or frames.size(0) < 1:
                    raise RuntimeError(
                        f"ours returned invalid ASR feature shape {tuple(frames.shape)}"
                    )
                if feature_dim is None:
                    feature_dim = int(frames.size(1))
                elif frames.size(1) != feature_dim:
                    raise RuntimeError("ours changed feature dimension within one run")

                feature_rel = pathlib.Path("features") / f"{accepted:08d}.pt"
                feature_path = cache_dir / feature_rel
                temp_path = feature_path.with_suffix(".tmp")
                torch.save(frames, temp_path)
                temp_path.replace(feature_path)
                record = {
                    "feature": str(feature_rel),
                    "text": text,
                    **_source_record(str(row["audio_filepath"])),
                }
                index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                index_file.flush()
                records.append(record)
                accepted += 1
                if accepted % 50 == 0:
                    print(
                        f"  [ASR-ATTN] ours: cached {accepted} "
                        f"{split_name} features",
                        flush=True,
                    )

    if not records or feature_dim is None:
        raise ValueError(f"No usable transcript/audio rows in {manifest}")
    _write_cache_metadata(
        meta_path,
        {
            **config,
            "complete": True,
            "feature_dim": feature_dim,
            "n_filtered": n_filtered,
            "n_unknown_duration": n_unknown_duration,
        },
    )
    print(
        f"  [ASR-ATTN] ours: cached {len(records)} {split_name} "
        f"variable-length features at {cache_dir}",
        flush=True,
    )
    return (
        _FeatureCache(cache_dir, records, feature_dim),
        n_filtered,
        n_unknown_duration,
    )


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
    ap.add_argument(
        "--ckpt",
        default=None,
        help="CLAE checkpoint; also supplies ours_random architecture",
    )
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--segment_seconds", type=float, default=None,
        help=(
            "Maximum audio duration passed to the feature extractor "
            "(also defaults --max_utt_seconds). Actual audio duration is "
            "verified before extraction."
        ),
    )
    ap.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap train/dev samples (0=unlimited)"
    )
    ap.add_argument(
        "--feature_cache_dir", default=None,
        help=(
            "Directory for variable-length frame tensors for every model "
            "(default: derived from --out). The cache is resumable and lets "
            "training/evaluation stream batches from disk."
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
    if args.steps < 1 or args.batch_size < 1:
        ap.error("--steps and --batch_size must be positive")
    if args.max_samples < 0:
        ap.error("--max_samples must be non-negative")
    if min(args.d_model, args.dec_layers, args.nhead, args.dim_ff, args.max_decode_len) < 1:
        ap.error("decoder dimensions/layers/max_decode_len must be positive")
    if args.d_model % args.nhead:
        ap.error("--d_model must be divisible by --nhead")
    if args.d_model % 2:
        ap.error("--d_model must be even for sinusoidal positional encoding")
    _reset_probe_seed(args.seed)

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
    if seg <= 0 or max_utt <= 0:
        ap.error("--segment_seconds and --max_utt_seconds must be positive")
    if max_utt > seg:
        ap.error(
            "--max_utt_seconds cannot exceed --segment_seconds: cropping audio "
            "while retaining the full transcript would invalidate the probe"
        )
    if chunk is not None and chunk <= 0:
        chunk = None
    print(
        f"  [ASR-ATTN] segment_seconds={seg:g}, max_utt_seconds={max_utt:g}, "
        f"chunk_seconds={'off' if chunk is None else f'{chunk:g}'}, "
        f"features={args.features} model={args.model}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Produce deterministic, text-valid manifests with explicit durations.
    #    This makes cache resume align one source row to one feature record.
    # ------------------------------------------------------------------
    out_path = pathlib.Path(args.out)
    cache_root = (
        pathlib.Path(args.feature_cache_dir)
        if args.feature_cache_dir
        else out_path.with_suffix(".attn_features")
    )
    train_manifest, _, n_filtered_tr, n_unknown_tr = _filter_manifest_by_duration(
        args.train_manifest,
        max_utt,
        out_path.with_suffix(".attn.train_filtered.jsonl"),
        "Filter train",
        text_key=args.text_key,
    )
    dev_manifest, _, n_filtered_de, n_unknown_de = _filter_manifest_by_duration(
        args.dev_manifest,
        max_utt,
        out_path.with_suffix(".attn.dev_filtered.jsonl"),
        "Filter dev",
        text_key=args.text_key,
    )

    if is_clae:
        extractor_identity = {
            "checkpoint": _file_identity(args.ckpt),
            "config_file": _file_identity(args.config),
            "resolved_config_sha256": _config_fingerprint(lm.cfg),
            "overrides": list(args.overrides),
            "sample_rate": lm.cfg.data.sample_rate,
            "features": args.features,
            "chunk_seconds": chunk,
            "mel_hop": args.mel_hop if args.features == "mel" else None,
        }
    else:
        spec = model_spec(args.model)
        extractor_identity = {
            "repo": spec.repo,
            "revision": spec.revision,
            "feature_layer": spec.feature_layer,
            "random_seed": (
                RANDOM_BASELINE_SEED if args.model == "ours_random" else None
            ),
            "checkpoint": (
                _file_identity(args.ckpt) if args.model == "ours_random" else None
            ),
        }

    # ------------------------------------------------------------------
    # 4. Dry-run: pull one batch, write shape info, exit
    # ------------------------------------------------------------------
    if args.dry_run:
        dry_cache_dir = out_path.with_suffix(".attn_dryrun_features")
        if is_clae:
            dry_cache, _, _ = _cache_clae_features(
                train_manifest,
                text_key=args.text_key,
                max_samples=1,
                segment_seconds=seg,
                max_utt_seconds=max_utt,
                lm=lm,
                chunk_seconds=chunk,
                source=args.features,
                mel_hop=args.mel_hop,
                extraction_batch_size=1,
                num_workers=args.num_workers,
                extractor_identity=extractor_identity,
                cache_dir=dry_cache_dir,
                split_name="dry-run",
                n_filtered=n_filtered_tr,
                n_unknown_duration=n_unknown_tr,
            )
        else:
            adapter = build_embedder(args.model, ckpt=args.ckpt)
            dry_cache, _, _ = _cache_external_features(
                args.model, train_manifest, text_key=args.text_key, max_samples=1,
                segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter,
                extractor_identity=extractor_identity,
                cache_dir=dry_cache_dir, split_name="dry-run",
                n_filtered=n_filtered_tr,
                n_unknown_duration=n_unknown_tr,
            )
        frame, text = _CachedFeatureDataset(dry_cache)[0]
        out = {
            "dry_run": True,
            "feats_shape": [1, *frame.shape],
            "num_samples": 1,
            "text_chars": len(text),
            "feature_cache_dir": str(dry_cache_dir),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ------------------------------------------------------------------
    # 5. Extract every model into the same resumable, variable-length cache.
    #    Training and evaluation below only pad/load the current DataLoader batch.
    # ------------------------------------------------------------------
    max_s = args.max_samples
    print(
        f"  [ASR-ATTN] Extracting train features{f' (max {max_s})' if max_s else ''}...",
        flush=True,
    )
    if is_clae:
        train_cache, n_filtered_tr, n_unknown_tr = _cache_clae_features(
            train_manifest,
            text_key=args.text_key,
            max_samples=max_s,
            segment_seconds=seg,
            max_utt_seconds=max_utt,
            lm=lm,
            chunk_seconds=chunk,
            source=args.features,
            mel_hop=args.mel_hop,
            extraction_batch_size=args.batch_size,
            num_workers=args.num_workers,
            extractor_identity=extractor_identity,
            cache_dir=cache_root / "train",
            split_name="train",
            n_filtered=n_filtered_tr,
            n_unknown_duration=n_unknown_tr,
        )
    else:
        # One frozen adapter serves both splits. Re-instantiating WavLM for
        # dev extraction needlessly reloads 95M parameters onto the GPU.
        adapter = build_embedder(args.model, ckpt=args.ckpt)
        train_cache, n_filtered_tr, n_unknown_tr = _cache_external_features(
            args.model, train_manifest, text_key=args.text_key, max_samples=max_s,
            segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter,
            extractor_identity=extractor_identity, cache_dir=cache_root / "train",
            split_name="train",
            n_filtered=n_filtered_tr,
            n_unknown_duration=n_unknown_tr,
        )
    print(
        f"  [ASR-ATTN] Extracting dev features{f' (max {max_s})' if max_s else ''}...",
        flush=True,
    )
    if is_clae:
        dev_cache, n_filtered_de, n_unknown_de = _cache_clae_features(
            dev_manifest,
            text_key=args.text_key,
            max_samples=max_s,
            segment_seconds=seg,
            max_utt_seconds=max_utt,
            lm=lm,
            chunk_seconds=chunk,
            source=args.features,
            mel_hop=args.mel_hop,
            extraction_batch_size=args.batch_size,
            num_workers=args.num_workers,
            extractor_identity=extractor_identity,
            cache_dir=cache_root / "dev",
            split_name="dev",
            n_filtered=n_filtered_de,
            n_unknown_duration=n_unknown_de,
        )
    else:
        dev_cache, n_filtered_de, n_unknown_de = _cache_external_features(
            args.model, dev_manifest, text_key=args.text_key, max_samples=max_s,
            segment_seconds=seg, max_utt_seconds=max_utt, embedder=adapter,
            extractor_identity=extractor_identity, cache_dir=cache_root / "dev",
            split_name="dev",
            n_filtered=n_filtered_de,
            n_unknown_duration=n_unknown_de,
        )

    # ------------------------------------------------------------------
    # 6. Free the frozen encoder to reclaim GPU memory
    # ------------------------------------------------------------------
    if lm is not None or adapter is not None:
        del lm, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"  [ASR-ATTN] Train: {len(train_cache.records)} disk-cached features "
        f"(D={train_cache.feature_dim}), Dev: {len(dev_cache.records)} "
        f"disk-cached features (D={dev_cache.feature_dim})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 7. Build the tiny vocabulary from this run's exact training transcripts.
    # ------------------------------------------------------------------
    vocab_list = build_attn_vocab(train_cache.texts)
    print(
        f"  [ASR-ATTN] Built charset with {len(vocab_list)} symbols.",
        flush=True,
    )
    vocab: Dict[str, int] = {c: i for i, c in enumerate(vocab_list)}
    id2tok: List[str] = vocab_list

    # ------------------------------------------------------------------
    # 8. Build model, optimizer, loss
    # ------------------------------------------------------------------
    # Feature extraction consumes RNG on a cache miss but not a cache hit.
    # Reset here so the recorded seed always identifies the same probe head.
    _reset_probe_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Features stay on CPU — only per-batch slices are moved to GPU.

    head = AttnDecoderHead(
        feat_dim=train_cache.feature_dim,
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
    # 9. Training loop
    # ------------------------------------------------------------------
    head.train()
    t0 = time.perf_counter()
    log_interval = max(1, args.steps // 10)

    train_dataset = _CachedFeatureDataset(train_cache)
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=args.steps * args.batch_size,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate_cached_features,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_iter = iter(train_loader)

    for step_i in range(args.steps):
        xb_cpu, vl_cpu, batch_texts = next(train_iter)
        xb = xb_cpu.to(device, non_blocking=True)
        vl = vl_cpu.to(device, non_blocking=True)
        batch_ids = [
            [vocab[c] for c in text.lower() if c in vocab]
            for text in batch_texts
        ]
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
    del train_iter, train_loader

    # ------------------------------------------------------------------
    # 10. Evaluate (greedy autoregressive decode)
    # ------------------------------------------------------------------
    print("  [ASR-ATTN] Evaluating...", flush=True)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": _collate_cached_features,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_eval_loader = DataLoader(_CachedFeatureDataset(train_cache), **loader_kwargs)
    result_tr = _decode_cached_loader(
        head, train_eval_loader, id2tok, device,
        max_decode_len=args.max_decode_len,
    )
    del train_eval_loader
    dev_eval_loader = DataLoader(_CachedFeatureDataset(dev_cache), **loader_kwargs)
    result_de = _decode_cached_loader(
        head, dev_eval_loader, id2tok, device,
        max_decode_len=args.max_decode_len,
    )
    del dev_eval_loader

    # ------------------------------------------------------------------
    # 11. Write output JSON
    # ------------------------------------------------------------------
    print(
        f"  [ASR-ATTN] Train WER: {result_tr['wer']:.4f} CER: {result_tr['cer']:.4f}, "
        f"Dev WER: {result_de['wer']:.4f} CER: {result_de['cer']:.4f}",
        flush=True,
    )
    out_data: Dict[str, Any] = {
        "protocol": "frozen_features_attention_decoder_v2",
        "train": result_tr,
        "dev": result_de,
        "vocab_size": len(vocab_list),
        "segment_seconds": float(seg),
        "max_utt_seconds": float(max_utt),
        "chunk_seconds": chunk,
        "features": args.features,
        "model": args.model,
        "text_key": args.text_key,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "n_train": len(train_cache.records),
        "n_dev": len(dev_cache.records),
        "feature_cache_dir": str(cache_root),
        "num_workers": args.num_workers,
        "disk_backed_dataloader": True,
        "seed": args.seed,
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
