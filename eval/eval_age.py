"""Speaker-disjoint Bengali Common Voice age probe across frozen embeddings.

Usage:
    uv run python -m eval.eval_age --cv_root datasets/common_voice_bn \
      --models ours,wavlm,whisper_tiny,ecapa --ckpt runs/.../last.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval.repr_bench import (
    DEFAULT_MODELS,
    EVAL_DIR,
    MODEL_ORDER,
    extract,
    load_common_voice_age_utterances,
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
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    scores, f1s, fold_results = [], [], []
    oof_predictions: list[dict | None] = [None] * len(y)
    for fold, (tr, te) in enumerate(folds):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        balanced_accuracy = float(balanced_accuracy_score(y[te], pred))
        macro_f1 = float(
            f1_score(y[te], pred, average="macro", zero_division=0)
        )
        scores.append(balanced_accuracy)
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
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_speakers": int(len(np.unique(groups[tr]))),
            "n_test_speakers": int(len(np.unique(groups[te]))),
        })
    return {
        "balanced_accuracy": float(np.mean(scores)),
        "balanced_accuracy_std": float(np.std(scores)),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_splits": len(folds),
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
        raise ValueError("Age probe needs at least two labelled speakers")
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
    ap.add_argument("--cv_root", required=True, help="Common Voice release directory containing validated.tsv + clips/")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated subset of: " + ",".join(MODEL_ORDER),
    )
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "age_probe.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not models:
        ap.error("--models must name at least one model")
    if args.folds < 2:
        ap.error("--folds must be at least 2")
    if args.max_utts is not None and args.max_utts < 1:
        ap.error("--max-utts must be positive")
    utts = load_common_voice_age_utterances(
        args.cv_root, args.max_utts, args.data_seed
    )
    item_ids = np.asarray([u.id for u in utts])
    y = np.asarray([u.age for u in utts])
    groups = np.asarray([u.speaker for u in utts])
    folds = _fixed_folds(y, groups, args.folds, args.split_seed)
    split_hash = _split_hash(item_ids, groups, folds)
    results = {
        name: probe(
            extract(
                name,
                utts,
                ckpt=args.ckpt,
                pool=args.pool,
                use_cache=not args.no_cache,
            )["X"],
            y,
            groups,
            item_ids,
            folds,
            args.seed,
        )
        for name in models
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "Common Voice Bengali validated", "n_utts": len(utts),
        "n_speakers": int(len(np.unique(groups))), "n_classes": int(len(np.unique(y))),
        "pool": args.pool, "seed": args.seed,
        "data_seed": args.data_seed, "split_seed": args.split_seed,
        "probe_seed": args.seed, "split_hash": split_hash,
        "split_protocol": "speaker_group_kfold", "bootstrap_unit": "speaker",
        "checkpoint": args.ckpt,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCommon Voice Bengali age ({len(utts)} clips, {payload['n_speakers']} speaker-disjoint)")
    for name, result in results.items():
        print(f"{name:<16} balanced acc={result['balanced_accuracy']*100:.1f}%  macro-F1={result['macro_f1']*100:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
