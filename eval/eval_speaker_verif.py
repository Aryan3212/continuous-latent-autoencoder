"""Rigorous speaker-verification benchmark across models.

Larger and more trustworthy than eval_speaker_eer.py: many speakers, all
same-/different-speaker trial pairs, EER **and** minDCF, evaluated under both
mean and mean+std pooling. This is the candidate "win" — our 0.4M-param / 64-dim
encoder beat WavLM/MMS/Mimi on EER at small scale; this checks whether it holds.

Pair scoring is vectorized (full cosine matrix), so thousands of utterances and
millions of trials are cheap.

    uv run python -m eval.eval_speaker_verif [--max-utts 2000] [--pools mean,meanstd]

Writes ``runs/eval/speaker_verif.json`` and prints a table.
"""
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_openslr53_utterances


def _trials(X: np.ndarray, speakers: np.ndarray):
    """All upper-triangle pairs: cosine scores + same-speaker labels (vectorized)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T                                  # (N, N) cosine
    same = speakers[:, None] == speakers[None, :]    # (N, N) bool
    iu = np.triu_indices(len(X), k=1)
    return sim[iu], same[iu].astype(np.int8)


def eer_and_mindcf(scores: np.ndarray, labels: np.ndarray,
                   p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> dict:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[k] + fnr[k]) / 2.0)

    # minDCF over the same operating points.
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = float(np.min(dcf) / norm)
    return {"eer": eer, "min_dcf": min_dcf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=2000)
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--pools", default="mean,meanstd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    utts = load_openslr53_utterances(max_utts=args.max_utts)
    n_spk = len({u.speaker for u in utts})

    results: dict = {p: {} for p in pools}
    for pool in pools:
        for name in models:
            data = extract(name, utts, ckpt=args.ckpt, pool=pool, use_cache=not args.no_cache)
            scores, labels = _trials(data["X"], data["speakers"])
            m = eer_and_mindcf(scores, labels)
            m.update({"n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                      "dim": int(data["X"].shape[1])})
            results[pool][name] = m

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "speaker_verif.json"
    out.write_text(json.dumps(
        {"n_utts": len(utts), "n_speakers": n_spk, "results": results}, indent=2),
        encoding="utf-8")

    print(f"\nSpeaker verification  ({len(utts)} utts, {n_spk} speakers)")
    for pool in pools:
        print(f"\n  pool = {pool}")
        print(f"  {'model':<14}{'EER %':>9}{'minDCF':>10}{'dim':>7}")
        print("  " + "-" * 38)
        ranked = sorted(results[pool].items(), key=lambda kv: kv[1]["eer"])
        for name, r in ranked:
            print(f"  {name:<14}{r['eer']*100:>8.2f}{r['min_dcf']:>10.3f}{r['dim']:>7}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
