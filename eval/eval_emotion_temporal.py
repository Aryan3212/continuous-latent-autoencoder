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
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from eval.repr_bench import (
    DEVICE,
    EVAL_DIR,
    build_embedder,
    load_subesco_utterances,
)

# Speaker-disjoint test set: 2 female + 2 male held out (SUBESCO has F_01..F_10,
# M_01..M_10). The head never sees these speakers during training.
TEST_SPEAKERS = {"F_09", "F_10", "M_09", "M_10"}


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


def get_frames(name: str, utts, max_frames: int) -> List[torch.Tensor]:
    """Frame features per utterance (CPU tensors, length-capped)."""
    emb = build_embedder(name)
    out: List[torch.Tensor] = []
    for i, u in enumerate(utts):
        f = torch.from_numpy(emb.fn(u.wav)).float()  # (T, D)
        if f.size(0) > max_frames:
            f = f[:max_frames]
        out.append(f)
        if (i + 1) % 500 == 0:
            print(f"[{name}] frames {i + 1}/{len(utts)}", flush=True)
    return out


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


def run_model(name: str, utts, y, groups, max_frames: int, epochs: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    frames = get_frames(name, utts, max_frames)
    D = frames[0].size(1)
    is_test = np.array([g in TEST_SPEAKERS for g in groups])
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]

    classes = sorted(set(y))
    c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])

    train_items = [(frames[i], int(yi[i])) for i in tr_idx]
    test_items = [(frames[i], int(yi[i])) for i in te_idx]

    head = AttnStatsHead(D, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    bs = 64
    rng = np.random.default_rng(0)

    head.train()
    for ep in range(epochs):
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
    return {"accuracy": acc, "macro_f1": f1, "dim": int(D),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=2100)
    ap.add_argument("--models", default="ours,ours_random,wavlm")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_subesco_utterances(max_utts=args.max_utts)
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    chance = 1.0 / len(set(y))

    results = {}
    for name in models:
        results[name] = run_model(name, utts, y, groups, args.max_frames, args.epochs)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "emotion_temporal.json"
    out.write_text(json.dumps(
        {"dataset": "SUBESCO", "n_utts": len(utts), "chance": chance,
         "pool": "attentive_stats", "test_speakers": sorted(TEST_SPEAKERS),
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
