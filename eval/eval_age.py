"""Speaker-disjoint Bengali Common Voice age probe across frozen embeddings.

Usage:
    uv run python -m eval.eval_age --cv_root datasets/common_voice_bn \
      --models ours,wavlm,whisper_tiny,ecapa --ckpt runs/.../last.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_common_voice_age_utterances


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Age probe needs at least two labelled speakers")
    scores, f1s = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        scores.append(balanced_accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {
        "balanced_accuracy": float(np.mean(scores)),
        "balanced_accuracy_std": float(np.std(scores)),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_splits": n_splits,
        "dim": int(X.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_root", required=True, help="Common Voice release directory containing validated.tsv + clips/")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    utts = load_common_voice_age_utterances(args.cv_root, args.max_utts, args.seed)
    y = np.asarray([u.age for u in utts])
    groups = np.asarray([u.speaker for u in utts])
    results = {
        name: probe(extract(name, utts, ckpt=args.ckpt, pool=args.pool, use_cache=not args.no_cache)["X"], y, groups, args.folds)
        for name in models
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "age_probe.json"
    payload = {
        "dataset": "Common Voice Bengali validated", "n_utts": len(utts),
        "n_speakers": int(len(np.unique(groups))), "n_classes": int(len(np.unique(y))),
        "pool": args.pool, "seed": args.seed, "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCommon Voice Bengali age ({len(utts)} clips, {payload['n_speakers']} speaker-disjoint)")
    for name, result in results.items():
        print(f"{name:<16} balanced acc={result['balanced_accuracy']*100:.1f}%  macro-F1={result['macro_f1']*100:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
