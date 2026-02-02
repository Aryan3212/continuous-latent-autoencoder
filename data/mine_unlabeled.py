from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

from eval.common import iter_embeddings, load_frozen_encoder


def _mahalanobis_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.sum(((x - mean) ** 2) / var, axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_manifest", required=True)
    ap.add_argument("--seed_index_npz", required=True, help="from data/embed_index.py")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_stats", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_mahal", type=float, default=200.0)
    ap.add_argument("--top_k", type=int, default=0, help="if >0, keep top_k closest instead of threshold")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    seed = np.load(args.seed_index_npz)
    mean = seed["mean"]
    var = seed["var"]

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    dcfg = lm.cfg["data"]

    # Load candidate rows to preserve metadata order.
    cand_rows = [json.loads(l) for l in pathlib.Path(args.candidates_manifest).read_text().splitlines() if l.strip()]
    scores: List[float] = []

    i = 0
    for e, meta in iter_embeddings(
        lm,
        args.candidates_manifest,
        sample_rate=int(dcfg["sample_rate"]),
        segment_seconds=float(dcfg["segment_seconds"]),
        batch_size=int(args.batch_size),
    ):
        x = e.numpy()
        d = _mahalanobis_diag(x, mean, var)
        scores.extend(d.tolist())
        i += len(meta)

    if len(scores) != len(cand_rows):
        raise RuntimeError(f"embedding count mismatch: {len(scores)} vs {len(cand_rows)}")

    order = np.argsort(np.asarray(scores))
    if int(args.top_k) > 0:
        keep_idx = set(order[: int(args.top_k)].tolist())
    else:
        keep_idx = {i for i, s in enumerate(scores) if float(s) <= float(args.max_mahal)}

    out_path = pathlib.Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(cand_rows):
            if i in keep_idx:
                row2 = dict(row)
                row2["seed_mahal"] = float(scores[i])
                f.write(json.dumps(row2) + "\n")
                kept += 1

    stats = {
        "candidates": len(cand_rows),
        "kept": kept,
        "rejected": len(cand_rows) - kept,
        "max_mahal": float(args.max_mahal),
        "top_k": int(args.top_k),
        "kept_score_min": float(min([scores[i] for i in keep_idx], default=0.0)),
        "kept_score_med": float(np.median([scores[i] for i in keep_idx])) if keep_idx else 0.0,
        "kept_score_max": float(max([scores[i] for i in keep_idx], default=0.0)),
    }
    sp = pathlib.Path(args.out_stats)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

