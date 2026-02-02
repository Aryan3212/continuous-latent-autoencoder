from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

import numpy as np
import torch

from eval.common import iter_embeddings, load_frozen_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed_manifest", required=True)
    ap.add_argument("--out_npz", required=True, help="writes mean/var_diag and optionally embeddings")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_embeddings", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    dcfg = lm.cfg["data"]

    embs: List[np.ndarray] = []
    for e, _meta in iter_embeddings(
        lm,
        args.seed_manifest,
        sample_rate=int(dcfg["sample_rate"]),
        segment_seconds=float(dcfg["segment_seconds"]),
        batch_size=int(args.batch_size),
    ):
        embs.append(e.numpy())
    X = np.concatenate(embs, axis=0)  # (N,2d)
    mean = X.mean(axis=0)
    var = X.var(axis=0) + 1e-6

    out = {"n": int(X.shape[0]), "dim": int(X.shape[1])}
    pathlib.Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        np.savez(args.out_npz, mean=mean, var=var, embeddings=X, meta=json.dumps(out))
    else:
        np.savez(args.out_npz, mean=mean, var=var, meta=json.dumps(out))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

