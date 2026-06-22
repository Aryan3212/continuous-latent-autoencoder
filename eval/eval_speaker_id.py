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
import json
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_utterances


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


def probe(X: np.ndarray, speakers: np.ndarray, test_per_speaker: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr, te = _per_speaker_split(speakers, test_per_speaker, seed)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, speakers[tr])
    train_acc = float(clf.score(Xtr, speakers[tr]))
    test_acc = float(clf.score(Xte, speakers[te]))
    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "dim": int(X.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--source", default="openslr53", choices=["openslr53", "cv"])
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test-per-speaker", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_utterances(args.source, max_utts=args.max_utts)
    n_spk = len({u.speaker for u in utts})
    chance = 1.0 / n_spk

    results = {}
    for name in models:
        data = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)
        results[name] = probe(
            data["X"], data["speakers"], args.test_per_speaker, args.seed
        )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "speaker_id_probe.json"
    payload = {"n_utts": len(utts), "n_speakers": n_spk, "chance": chance, "results": results}
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
