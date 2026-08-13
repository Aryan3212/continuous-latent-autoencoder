"""Closed-set speaker-ID linear probe on mean-pooled embeddings.

Trains a linear classifier (multinomial logistic regression on standardized
features) to predict speaker id from a single pooled utterance embedding, with a
per-speaker train/test split. Reports top-1 accuracy per model — a direct
measure of how much *linearly decodable* speaker (utterance-level) information
each representation carries. Reuses the embedding cache from the other repr
scripts, so it's effectively free after a prior run.

    uv run python -m eval.eval_speaker_id [--max-utts 300] [--test-per-speaker 2]

Writes ``runs/eval/speaker_id_probe.json`` and prints a table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.repr_bench import DEFAULT_MODELS, EVAL_DIR, MODEL_ORDER, extract, load_utterances


def _per_speaker_split(speakers: np.ndarray, test_per_speaker: int, seed: int):
    """Indices split so each speaker contributes the same #clips to test."""
    by_spk: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(speakers):
        by_spk[str(s)].append(i)
    rng = random.Random(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for s, idxs in by_spk.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        # Keep at least one training clip per speaker.
        n_test = min(test_per_speaker, max(0, len(idxs) - 1))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return np.array(train_idx), np.array(test_idx)


def probe(
    X: np.ndarray,
    speakers: np.ndarray,
    item_ids: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    test_per_speaker: int,
    seed: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr, te = train_indices, test_indices
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(
            "Speaker-ID split needs at least one train and test utterance; "
            "increase --max-utts or reduce --test-per-speaker"
        )
    if set(speakers[te]) - set(speakers[tr]):
        raise RuntimeError("Closed-set speaker-ID test contains an unenrolled speaker")
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(Xtr, speakers[tr])
    train_acc = float(clf.score(Xtr, speakers[tr]))
    predictions = clf.predict(Xte)
    test_acc = float(np.mean(predictions == speakers[te]))
    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "dim": int(X.shape[1]),
        "seed": seed,
        "probe_seed": seed,
        "test_per_speaker": test_per_speaker,
        "prediction_protocol": "heldout_test",
        "predictions": [
            {
                "item_id": str(item_ids[index]),
                "speaker_id": str(speakers[index]),
                "fold": 0,
                "gold": str(speakers[index]),
                "prediction": str(prediction),
            }
            for index, prediction in zip(te, predictions)
        ],
    }


def _split_hash(
    item_ids: np.ndarray,
    speakers: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> str:
    roles = np.full(len(item_ids), "unassigned", dtype=object)
    roles[train_indices] = "train"
    roles[test_indices] = "test"
    rows = [
        [str(item_id), str(speaker_id), str(role)]
        for item_id, speaker_id, role in zip(item_ids, speakers, roles)
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--source", default="openslr53", choices=["openslr53", "cv"])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test-per-speaker", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "speaker_id_probe.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        ap.error("--models must name at least one model")
    if args.max_utts < 2 or args.test_per_speaker < 1:
        ap.error("--max-utts must be at least 2 and --test-per-speaker positive")
    utts = load_utterances(
        args.source, max_utts=args.max_utts, seed=args.data_seed
    )
    item_ids = np.asarray([u.id for u in utts])
    speakers = np.asarray([u.speaker for u in utts])
    train_indices, test_indices = _per_speaker_split(
        speakers, args.test_per_speaker, args.split_seed
    )
    split_hash = _split_hash(
        item_ids, speakers, train_indices, test_indices
    )
    n_spk = len({u.speaker for u in utts})
    chance = 1.0 / n_spk

    results = {}
    for name in models:
        data = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)
        results[name] = probe(
            data["X"],
            speakers,
            item_ids,
            train_indices,
            test_indices,
            args.test_per_speaker,
            args.seed,
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "closed_set_same_speakers_heldout_utterances",
        "n_utts": len(utts),
        "n_speakers": n_spk,
        "chance": chance,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "probe_seed": args.seed,
        "split_hash": split_hash,
        "split_protocol": "per_speaker_heldout_utterances",
        "bootstrap_unit": "speaker",
        "checkpoint": args.ckpt,
        "test_per_speaker": args.test_per_speaker,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nClosed-set speaker-ID linear probe  "
          f"({len(utts)} utts, {n_spk} speakers, chance={chance*100:.1f}%)")
    print(f"{'model':<14} {'test acc':>9} {'train acc':>10} {'dim':>6}")
    print("-" * 44)
    for name in models:
        r = results[name]
        print(f"{name:<14} {r['test_acc']*100:>8.1f}% {r['train_acc']*100:>9.1f}% {r['dim']:>6}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
