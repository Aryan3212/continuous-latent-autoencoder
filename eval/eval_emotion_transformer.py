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
import hashlib
import json
import math
import pathlib
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


def _train_eval(frames, yi, tr, te, n_classes, epochs, seed):
    from sklearn.metrics import accuracy_score, f1_score

    D = frames[0].size(1)
    train_items = [(frames[i], int(yi[i])) for i in tr]
    test_items = [(frames[i], int(yi[i])) for i in te]
    fork_devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        net = TransformerProbe(D, n_classes).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        bs = 64
        rng = np.random.default_rng(seed)
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
                preds.extend(
                    net(x.to(DEVICE), mask.to(DEVICE)).argmax(-1).cpu().tolist()
                )
    gold = [int(yi[i]) for i in te]
    return (
        float(accuracy_score(gold, preds)),
        float(f1_score(gold, preds, average="macro")),
        preds,
    )


def run_model(
    name,
    utts,
    y,
    groups,
    max_frames,
    folds,
    epochs,
    *,
    ckpt,
    seed,
    feature_cache_dir,
):
    frames, cache_info = get_frames(
        name,
        utts,
        max_frames,
        ckpt=ckpt,
        feature_cache_dir=feature_cache_dir,
    )
    classes = sorted(set(y))
    c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])
    accs, f1s = [], []
    oof_predictions: list[dict | None] = [None] * len(y)
    for fold, (tr, te) in enumerate(folds):
        a, f, predictions = _train_eval(
            frames, yi, tr, te, len(classes), epochs, seed + fold
        )
        accs.append(a)
        f1s.append(f)
        for index, prediction in zip(te, predictions):
            oof_predictions[int(index)] = {
                "item_id": str(utts[index].id),
                "speaker_id": str(groups[index]),
                "fold": fold,
                "gold": str(y[index]),
                "prediction": str(classes[prediction]),
            }
    print(f"[{name}] transformer: acc={np.mean(accs)*100:.1f}% "
          f"macroF1={np.mean(f1s)*100:.1f}% (D={frames[0].size(1)})", flush=True)
    return {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "dim": int(frames[0].size(1)), "n_splits": int(len(folds)),
            "probe_seed": seed, "prediction_protocol": "out_of_fold",
            "feature_cache": cache_info,
            "predictions": [row for row in oof_predictions if row is not None]}


def _fixed_folds(y, groups, n_folds, split_seed):
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_folds, len(np.unique(groups)))
    return list(
        GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_seed,
        ).split(np.zeros(len(y)), y, groups)
    )


def _split_hash(utts, groups, folds):
    fold_by_item = np.full(len(utts), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(folds):
        fold_by_item[test_indices] = fold
    rows = [
        [str(utterance.id), str(speaker), int(fold)]
        for utterance, speaker, fold in zip(utts, groups, fold_by_item)
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--models", default="ours,ours_random,wavlm,mms")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--subesco-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--feature-cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=EVAL_DIR / "emotion_transformer.json")
    args = ap.parse_args()
    if args.max_utts is not None and args.max_utts < 1:
        ap.error("--max-utts must be positive")
    if min(args.max_frames, args.folds, args.epochs) < 1 or args.folds < 2:
        ap.error("frame/epoch budgets must be positive and --folds at least 2")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_subesco_utterances(
        max_utts=args.max_utts, seed=args.data_seed, root=args.subesco_dir
    )
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    folds = _fixed_folds(y, groups, args.folds, args.split_seed)
    split_hash = _split_hash(utts, groups, folds)
    chance = 1.0 / len(set(y))

    results = {}
    for name in models:
        results[name] = run_model(
            name,
            utts,
            y,
            groups,
            args.max_frames,
            folds,
            args.epochs,
            ckpt=args.ckpt,
            seed=args.seed,
            feature_cache_dir=args.feature_cache_dir,
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": "SUBESCO", "n_utts": len(utts), "chance": chance,
         "probe": "transformer_cls", "seed": args.seed,
         "data_seed": args.data_seed, "split_seed": args.split_seed,
         "probe_seed": args.seed, "split_hash": split_hash,
         "split_protocol": "speaker_group_kfold", "bootstrap_unit": "speaker",
         "checkpoint": args.ckpt,
         "feature_cache_dir": (
             str(args.feature_cache_dir.resolve()) if args.feature_cache_dir else None
         ),
         "results": results}, indent=2), encoding="utf-8")

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
