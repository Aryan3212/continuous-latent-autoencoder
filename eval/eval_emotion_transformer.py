"""Transformer-probe emotion recognition on SUBESCO.

Treats each frozen frame embedding as a token (BERT-style: prepend a [CLS]
token, add positional encoding, run a small Transformer encoder, classify from
[CLS]). Unlike mean+std or attentive-stats pooling, self-attention models
cross-frame *dynamics* — the pitch/energy contour where emotion lives — so this
is the strongest test of whether emotion is recoverable from a representation.

Interpretation stays anchored on **lift over ours_random**: a transformer is
powerful enough to exploit speaker-correlated shortcuts, so the random-init
control (identical probe, untrained encoder) is the honest baseline. Full 7000
utts, speaker-disjoint GroupKFold.

    uv run python -m eval.eval_emotion_transformer [--models ...] [--folds 4]

Writes ``runs/eval/emotion_transformer.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from eval.eval_emotion_temporal import get_frames
from eval.repr_bench import DEVICE, EVAL_DIR, load_subesco_utterances


class TransformerProbe(nn.Module):
    """Small Transformer encoder over frame tokens with a [CLS] readout."""

    def __init__(self, feat_dim: int, n_classes: int, d_model: int = 128,
                 nhead: int = 4, layers: int = 2, dim_ff: int = 256,
                 dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); pad_mask: (B, T) True where padding.
        B = x.size(0)
        h = self.in_proj(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)                       # (B, 1+T, d)
        h = h + self.pe[:, : h.size(1)]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        kpm = torch.cat([cls_mask, pad_mask], dim=1)         # CLS never masked
        h = self.encoder(h, src_key_padding_mask=kpm)
        return self.head(h[:, 0])                            # [CLS]


def _collate(items):
    feats = [f for f, _ in items]
    labels = torch.tensor([y for _, y in items], dtype=torch.long)
    T = max(f.size(0) for f in feats)
    x = torch.zeros(len(feats), T, feats[0].size(1))
    mask = torch.ones(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, : f.size(0)] = f
        mask[i, : f.size(0)] = False
    return x, mask, labels


def _train_eval(frames, yi, tr, te, n_classes, epochs):
    from sklearn.metrics import accuracy_score, f1_score

    D = frames[0].size(1)
    train_items = [(frames[i], int(yi[i])) for i in tr]
    test_items = [(frames[i], int(yi[i])) for i in te]
    net = TransformerProbe(D, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    bs = 64
    rng = np.random.default_rng(0)
    net.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_items))
        for s in range(0, len(order), bs):
            x, mask, lab = _collate([train_items[i] for i in order[s : s + bs]])
            x, mask, lab = x.to(DEVICE), mask.to(DEVICE), lab.to(DEVICE)
            logits = net(x, mask)
            loss = loss_fn(logits, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    preds: List[int] = []
    with torch.no_grad():
        for s in range(0, len(test_items), bs):
            x, mask, _ = _collate(test_items[s : s + bs])
            preds.extend(net(x.to(DEVICE), mask.to(DEVICE)).argmax(-1).cpu().tolist())
    gold = [int(yi[i]) for i in te]
    return accuracy_score(gold, preds), f1_score(gold, preds, average="macro")


def run_model(name, utts, y, groups, max_frames, folds, epochs):
    from sklearn.model_selection import GroupKFold

    frames = get_frames(name, utts, max_frames)
    classes = sorted(set(y))
    c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])
    accs, f1s = [], []
    n_splits = min(folds, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits).split(frames, yi, groups):
        a, f = _train_eval(frames, yi, tr, te, len(classes), epochs)
        accs.append(a)
        f1s.append(f)
    print(f"[{name}] transformer: acc={np.mean(accs)*100:.1f}% "
          f"macroF1={np.mean(f1s)*100:.1f}% (D={frames[0].size(1)})", flush=True)
    return {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "dim": int(frames[0].size(1)), "n_splits": int(n_splits)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--models", default="ours,ours_random,wavlm,mms")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_subesco_utterances(max_utts=args.max_utts)
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    chance = 1.0 / len(set(y))

    results = {}
    for name in models:
        results[name] = run_model(name, utts, y, groups, args.max_frames, args.folds, args.epochs)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "emotion_transformer.json"
    out.write_text(json.dumps(
        {"dataset": "SUBESCO", "n_utts": len(utts), "chance": chance,
         "probe": "transformer_cls", "results": results}, indent=2), encoding="utf-8")

    rand = results.get("ours_random", {}).get("macro_f1")
    print(f"\nSUBESCO emotion — TRANSFORMER probe  "
          f"({len(utts)} utts, speaker-disjoint {results[models[0]]['n_splits']}-fold, "
          f"chance={chance*100:.1f}%)")
    print(f"{'model':<14}{'macro-F1':>14}{'accuracy':>14}{'lift_vs_rand':>14}{'dim':>6}")
    print("-" * 62)
    for name in models:
        r = results[name]
        lift = f"{(r['macro_f1']-rand)*100:+.1f}" if rand is not None else "n/a"
        print(f"{name:<14}{r['macro_f1']*100:>7.1f} ±{r['macro_f1_std']*100:>4.1f}  "
              f"{r['accuracy']*100:>7.1f} ±{r['accuracy_std']*100:>4.1f}  {lift:>12}  {r['dim']:>5}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
