"""Speaker-verification EER on mean-pooled encoder latents.

For each model, forms every same-speaker / different-speaker utterance pair,
scores pairs by cosine similarity of their mean-pooled embeddings, and reports
the Equal Error Rate. Lower EER = the representation separates speakers better.

    uv run python -m eval.eval_repr_eer [--max-utts 300] [--models ours,mimi,...]

Writes ``runs/eval/speaker_eer.json`` and prints a single table.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from typing import List

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_utterances


def compute_eer(X: np.ndarray, speakers: np.ndarray) -> dict:
    """EER over all utterance pairs, scored by cosine similarity."""
    from sklearn.metrics import roc_curve

    # Cosine similarity = dot product of L2-normalised embeddings.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    idx = list(range(len(X)))
    scores: List[float] = []
    labels: List[int] = []
    for i, j in combinations(idx, 2):
        scores.append(float(Xn[i] @ Xn[j]))
        labels.append(int(speakers[i] == speakers[j]))

    labels_arr = np.asarray(labels)
    n_pos = int(labels_arr.sum())
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"eer": float("nan"), "n_pos_pairs": n_pos, "n_neg_pairs": n_neg}

    fpr, tpr, _ = roc_curve(labels_arr, np.asarray(scores))
    fnr = 1.0 - tpr
    # EER is where the false-accept and false-reject rates cross.
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[k] + fnr[k]) / 2.0)
    return {"eer": eer, "n_pos_pairs": n_pos, "n_neg_pairs": n_neg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--source", default="openslr53", choices=["openslr53", "cv"],
                    help="Utterance source (default: local OpenSLR-53).")
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None, help="Local path or HF repo for our model.")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_utterances(args.source, max_utts=args.max_utts)

    results = {}
    for name in models:
        data = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)
        results[name] = compute_eer(data["X"], data["speakers"])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "speaker_eer.json"
    payload = {
        "n_utts": len(utts),
        "n_speakers": len({u.speaker for u in utts}),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSpeaker-verification EER  ({len(utts)} utts, "
          f"{payload['n_speakers']} speakers)")
    print(f"{'model':<14} {'EER %':>8}   {'+pairs':>8} {'-pairs':>8}")
    print("-" * 44)
    for name in models:
        r = results[name]
        print(f"{name:<14} {r['eer'] * 100:>7.2f}   {r['n_pos_pairs']:>8} {r['n_neg_pairs']:>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
