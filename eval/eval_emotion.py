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
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np

from eval.repr_bench import (
    DEFAULT_MODELS,
    EVAL_DIR,
    MODEL_ORDER,
    extract,
    load_subesco_utterances,
)


def probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    item_ids: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    accs: List[float] = []
    f1s: List[float] = []
    fold_results = []
    oof_predictions: list[dict | None] = [None] * len(y)
    for fold, (tr, te) in enumerate(folds):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accuracy = float(accuracy_score(y[te], pred))
        macro_f1 = float(
            f1_score(y[te], pred, average="macro", zero_division=0)
        )
        accs.append(accuracy)
        f1s.append(macro_f1)
        for index, prediction in zip(te, pred):
            oof_predictions[int(index)] = {
                "item_id": str(item_ids[index]),
                "speaker_id": str(groups[index]),
                "fold": fold,
                "gold": str(y[index]),
                "prediction": str(prediction),
            }
        fold_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_speakers": int(len(np.unique(groups[tr]))),
            "n_test_speakers": int(len(np.unique(groups[te]))),
        })

    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_splits": int(len(folds)),
        "dim": int(X.shape[1]),
        "folds": fold_results,
        "prediction_protocol": "out_of_fold",
        "predictions": [row for row in oof_predictions if row is not None],
    }


def _fixed_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    split_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Emotion probe needs at least two labelled speakers")
    return list(
        GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_seed,
        ).split(np.zeros(len(y)), y, groups)
    )


def _split_hash(
    item_ids: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    fold_by_item = np.full(len(item_ids), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(folds):
        fold_by_item[test_indices] = fold
    rows = [
        [str(item_id), str(speaker_id), int(fold)]
        for item_id, speaker_id, fold in zip(item_ids, groups, fold_by_item)
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=None, help="Cap clips (default: all 7000).")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--subesco-dir", default=None)
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "emotion_probe.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        ap.error("--models must name at least one model")
    if args.folds < 2:
        ap.error("--folds must be at least 2")
    if args.max_utts is not None and args.max_utts < 1:
        ap.error("--max-utts must be positive")
    utts = load_subesco_utterances(
        max_utts=args.max_utts,
        seed=args.data_seed,
        root=args.subesco_dir,
    )
    item_ids = np.asarray([u.id for u in utts])
    y = np.array([u.emotion for u in utts])
    groups = np.array([u.speaker for u in utts])
    folds = _fixed_folds(y, groups, args.folds, args.split_seed)
    split_hash = _split_hash(item_ids, groups, folds)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    results = {}
    for name in models:
        data = extract(name, utts, ckpt=args.ckpt, pool=args.pool, use_cache=not args.no_cache)
        results[name] = probe(
            data["X"], y, groups, item_ids, folds, args.seed
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "SUBESCO",
        "n_utts": len(utts),
        "n_speakers": int(len(np.unique(groups))),
        "n_classes": n_classes,
        "chance": chance,
        "pool": args.pool,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "probe_seed": args.seed,
        "split_hash": split_hash,
        "split_protocol": "speaker_group_kfold",
        "bootstrap_unit": "speaker",
        "checkpoint": args.ckpt,
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
