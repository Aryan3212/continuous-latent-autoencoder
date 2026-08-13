"""Temporal-pooling emotion probe on SUBESCO — does a sequence-aware head
recover emotion that mean+std pooling misses?

Instead of collapsing frames to mean+std and fitting a linear classifier, this
trains a small **attentive statistics pooling** head over the frozen frame
features (the standard pooling for speaker/emotion systems). Emotion lives in
the temporal contour of pitch/energy, which mean+std discards — so if our
model's representation contains emotion at all, this head should expose it.

Decisive comparison: ours vs ours_random (is there a learned signal?) with a
strong baseline as ceiling. Single speaker-disjoint split.

    uv run python -m eval.eval_emotion_temporal [--max-utts N] [--models ...]

Writes ``runs/eval/emotion_temporal.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from eval.repr_bench import (
    DEVICE,
    EVAL_DIR,
    RANDOM_BASELINE_SEED,
    _resolve_our_ckpt,
    build_embedder,
    load_subesco_utterances,
    model_spec,
)

# Speaker-disjoint test set: 2 female + 2 male held out (SUBESCO has F_01..F_10,
# M_01..M_10). The head never sees these speakers during training.
TEST_SPEAKERS = {"F_09", "F_10", "M_09", "M_10"}
FRAME_CACHE_VERSION = 1
FRAME_EXTRACTOR_ID = "eval.emotion_sequence_frames"


class AttnStatsHead(nn.Module):
    """Attentive statistics pooling + linear classifier over frame features."""

    def __init__(self, feat_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh())
        self.attn = nn.Linear(hidden, 1)
        self.cls = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) True where padding.
        w = self.attn(self.proj(x)).squeeze(-1)        # (B, T)
        w = w.masked_fill(mask, float("-inf"))
        a = torch.softmax(w, dim=1).unsqueeze(-1)       # (B, T, 1)
        mean = (a * x).sum(dim=1)                        # (B, D)
        var = (a * (x - mean.unsqueeze(1)) ** 2).sum(dim=1).clamp_min(1e-6)
        stats = torch.cat([mean, var.sqrt()], dim=-1)   # (B, 2D)
        return self.cls(stats)


def get_frames(
    name: str,
    utts,
    max_frames: int,
    *,
    ckpt: str | None = None,
    feature_cache_dir: pathlib.Path | None = None,
) -> tuple[List[torch.Tensor], dict]:
    """Frame features per utterance (CPU tensors, length-capped)."""
    resolved_ckpt = (
        _resolve_our_ckpt(ckpt) if name in {"ours", "ours_random"} else ckpt
    )
    spec = model_spec(name)
    checkpoint_identity: dict | None = None
    if resolved_ckpt is not None:
        checkpoint_path = pathlib.Path(resolved_ckpt).expanduser()
        if checkpoint_path.is_file():
            stat = checkpoint_path.stat()
            checkpoint_identity = {
                "path": str(checkpoint_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            checkpoint_identity = {"value": str(resolved_ckpt)}

    audio_digest = hashlib.sha256()
    items = []
    for utterance in utts:
        waveform = utterance.wav.detach().cpu().contiguous()
        items.append({
            "item_id": str(utterance.id),
            "speaker_id": str(utterance.speaker),
            "num_samples": int(waveform.numel()),
        })
        audio_digest.update(str(utterance.id).encode("utf-8"))
        audio_digest.update(b"\0")
        audio_digest.update(waveform.numpy().tobytes())
    metadata = {
        "cache_version": FRAME_CACHE_VERSION,
        "extractor": FRAME_EXTRACTOR_ID,
        "model": name,
        "model_identity": {
            "repo": spec.repo,
            "revision": spec.revision,
            "feature_layer": spec.feature_layer,
            "component": spec.component,
            "native_sample_rate": spec.native_sample_rate,
            "random_baseline_seed": (
                RANDOM_BASELINE_SEED if name == "ours_random" else None
            ),
        },
        "checkpoint": checkpoint_identity,
        "device": str(DEVICE),
        "max_frames": max_frames,
        "items": items,
        "audio_fingerprint": audio_digest.hexdigest(),
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
    cache_path = (
        feature_cache_dir / f"frames.{cache_key}.npz"
        if feature_cache_dir is not None
        else None
    )

    if cache_path is not None and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_metadata = str(cached["metadata"].item())
                cached_ids = cached["item_ids"]
                lengths = cached["lengths"].astype(np.int64, copy=False)
                features = cached["features"].astype(np.float32, copy=False)
            valid = (
                cached_metadata == metadata_json
                and cached_ids.tolist() == [item["item_id"] for item in items]
                and lengths.ndim == 1
                and len(lengths) == len(items)
                and np.all(lengths > 0)
                and np.all(lengths <= max_frames)
                and features.ndim == 2
                and features.shape[1] > 0
                and int(lengths.sum()) == len(features)
            )
            if valid:
                offsets = np.concatenate(([0], np.cumsum(lengths)))
                frames = [
                    torch.from_numpy(features[start:end].copy())
                    for start, end in zip(offsets[:-1], offsets[1:])
                ]
                print(f"[{name}] using cached frames {cache_path}", flush=True)
                return frames, {
                    "enabled": True,
                    "hit": True,
                    "key": cache_key,
                    "path": str(cache_path.resolve()),
                }
            print(f"[{name}] ignoring invalid frame cache {cache_path}", flush=True)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            print(f"[{name}] ignoring unreadable frame cache {cache_path}: {exc}", flush=True)

    emb = build_embedder(
        name,
        ckpt=resolved_ckpt,
        random_seed=RANDOM_BASELINE_SEED,
    )
    out: List[torch.Tensor] = []
    for i, u in enumerate(utts):
        f = torch.from_numpy(emb.fn(u.wav)).float()  # (T, D)
        if f.size(0) > max_frames:
            f = f[:max_frames]
        out.append(f)
        if (i + 1) % 500 == 0:
            print(f"[{name}] frames {i + 1}/{len(utts)}", flush=True)
    cache_info = {
        "enabled": cache_path is not None,
        "hit": False,
        "key": cache_key,
        "path": str(cache_path.resolve()) if cache_path is not None else None,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.asarray([frame.size(0) for frame in out], dtype=np.int32)
        features = np.concatenate(
            [frame.cpu().numpy().astype(np.float32, copy=False) for frame in out],
            axis=0,
        )
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
        try:
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(metadata_json),
                item_ids=np.asarray([item["item_id"] for item in items]),
                lengths=lengths,
                features=features,
            )
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"[{name}] cached frames at {cache_path}", flush=True)
    return out, cache_info


def _collate(items: List[Tuple[torch.Tensor, int]]):
    feats = [f for f, _ in items]
    labels = torch.tensor([y for _, y in items], dtype=torch.long)
    T = max(f.size(0) for f in feats)
    D = feats[0].size(1)
    x = torch.zeros(len(feats), T, D)
    mask = torch.ones(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, : f.size(0)] = f
        mask[i, : f.size(0)] = False
    return x, mask, labels


def run_model(
    name: str,
    utts,
    y,
    groups,
    max_frames: int,
    epochs: int,
    *,
    ckpt: str | None,
    seed: int,
    feature_cache_dir: pathlib.Path | None,
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    frames, cache_info = get_frames(
        name,
        utts,
        max_frames,
        ckpt=ckpt,
        feature_cache_dir=feature_cache_dir,
    )
    D = frames[0].size(1)
    is_test = np.array([g in TEST_SPEAKERS for g in groups])
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]
    if len(tr_idx) == 0 or len(te_idx) == 0:
        raise ValueError(
            "SUBESCO subset must include both the fixed held-out speakers and training speakers"
        )

    classes = sorted(set(y))
    c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])

    train_items = [(frames[i], int(yi[i])) for i in tr_idx]
    test_items = [(frames[i], int(yi[i])) for i in te_idx]

    fork_devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        head = AttnStatsHead(D, len(classes)).to(DEVICE)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        bs = 64
        rng = np.random.default_rng(seed)

        head.train()
        for _ in range(epochs):
            order = rng.permutation(len(train_items))
            for s in range(0, len(order), bs):
                batch = [train_items[i] for i in order[s : s + bs]]
                x, mask, lab = _collate(batch)
                x, mask, lab = x.to(DEVICE), mask.to(DEVICE), lab.to(DEVICE)
                logits = head(x, mask)
                loss = loss_fn(logits, lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        head.eval()
        preds: List[int] = []
        with torch.no_grad():
            for s in range(0, len(test_items), bs):
                x, mask, _ = _collate(test_items[s : s + bs])
                logits = head(x.to(DEVICE), mask.to(DEVICE))
                preds.extend(logits.argmax(-1).cpu().tolist())
    gold = [c2i[y[i]] for i in te_idx]
    acc = float(accuracy_score(gold, preds))
    f1 = float(f1_score(gold, preds, average="macro"))
    print(f"[{name}] temporal: acc={acc*100:.1f}% macroF1={f1*100:.1f}% (D={D})", flush=True)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "dim": int(D),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "probe_seed": seed,
        "feature_cache": cache_info,
        "prediction_protocol": "heldout_test",
        "predictions": [
            {
                "item_id": str(utts[index].id),
                "speaker_id": str(groups[index]),
                "fold": 0,
                "gold": str(y[index]),
                "prediction": str(classes[prediction]),
            }
            for index, prediction in zip(te_idx, preds)
        ],
    }


def _split_hash(utts, groups: np.ndarray) -> str:
    rows = [
        [
            str(utterance.id),
            str(speaker),
            "test" if speaker in TEST_SPEAKERS else "train",
        ]
        for utterance, speaker in zip(utts, groups)
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=2100)
    ap.add_argument("--models", default="ours,ours_random,wavlm")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--subesco-dir", required=False, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--feature-cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=EVAL_DIR / "emotion_temporal.json")
    args = ap.parse_args()
    if args.max_utts < 1 or args.max_frames < 1 or args.epochs < 1:
        ap.error("utterance/frame budgets and epochs must be positive")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_subesco_utterances(
        max_utts=args.max_utts,
        seed=args.data_seed,
        root=args.subesco_dir,
    )
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    split_hash = _split_hash(utts, groups)
    chance = 1.0 / len(set(y))

    results = {}
    for name in models:
        results[name] = run_model(
            name,
            utts,
            y,
            groups,
            args.max_frames,
            args.epochs,
            ckpt=args.ckpt,
            seed=args.seed,
            feature_cache_dir=args.feature_cache_dir,
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": "SUBESCO", "n_utts": len(utts), "chance": chance,
         "pool": "attentive_stats", "test_speakers": sorted(TEST_SPEAKERS),
         "seed": args.seed, "data_seed": args.data_seed,
         "split_seed": args.split_seed, "probe_seed": args.seed,
         "split_hash": split_hash, "split_protocol": "fixed_heldout_speakers",
         "bootstrap_unit": "speaker", "checkpoint": args.ckpt,
         "feature_cache_dir": (
             str(args.feature_cache_dir.resolve()) if args.feature_cache_dir else None
         ),
         "results": results}, indent=2), encoding="utf-8")

    print(f"\nSUBESCO emotion — TEMPORAL (attentive-stats) head  "
          f"({len(utts)} utts, speaker-disjoint, chance={chance*100:.1f}%)")
    print(f"{'model':<14}{'macro-F1':>10}{'accuracy':>10}{'dim':>7}")
    print("-" * 41)
    for name in models:
        r = results[name]
        print(f"{name:<14}{r['macro_f1']*100:>9.1f}{r['accuracy']*100:>10.1f}{r['dim']:>7}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
