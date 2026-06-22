"""Speech emotion recognition (SER) probe on SUBESCO, across models.

Trains a linear classifier on mean+std-pooled embeddings to predict the 7
SUBESCO emotions, using **speaker-disjoint** GroupKFold so the score reflects
emotion decodability rather than speaker leakage. Reports macro-F1 and accuracy
(mean over folds) per model.

    uv run python -m eval.eval_emotion [--max-utts N] [--models ours,mimi,...] [--folds 5]

Writes ``runs/eval/emotion_probe.json`` and prints a table.
"""
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_subesco_utterances


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups))
    n_splits = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    accs: List[float] = []
    f1s: List[float] = []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))

    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_splits": int(n_splits),
        "dim": int(X.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=None, help="Cap clips (default: all 7000).")
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_subesco_utterances(max_utts=args.max_utts)
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    results = {}
    for name in models:
        data = extract(name, utts, ckpt=args.ckpt, pool=args.pool, use_cache=not args.no_cache)
        results[name] = probe(data["X"], y, groups, args.folds)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "emotion_probe.json"
    payload = {
        "dataset": "SUBESCO",
        "n_utts": len(utts),
        "n_speakers": int(len(np.unique(groups))),
        "n_classes": n_classes,
        "chance": chance,
        "pool": args.pool,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSUBESCO emotion recognition  ({len(utts)} utts, {n_classes} emotions, "
          f"speaker-disjoint {payload['results'][models[0]]['n_splits']}-fold, "
          f"chance={chance*100:.1f}%, pool={args.pool})")
    print(f"{'model':<14} {'macro-F1':>14} {'accuracy':>14} {'dim':>6}")
    print("-" * 52)
    for name in models:
        r = results[name]
        print(f"{name:<14} {r['macro_f1']*100:>7.1f} ±{r['macro_f1_std']*100:>4.1f}  "
              f"{r['accuracy']*100:>7.1f} ±{r['accuracy_std']*100:>4.1f}  {r['dim']:>6}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
