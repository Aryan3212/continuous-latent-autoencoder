from __future__ import annotations

import argparse
import json
import pathlib

from eval.common import iter_embeddings, load_frozen_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    dcfg = lm.cfg.data
    seg = args.segment_seconds if args.segment_seconds is not None else dcfg.segment_seconds

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for emb, meta in iter_embeddings(
            lm,
            args.manifest,
            sample_rate=dcfg.sample_rate,
            segment_seconds=seg,
            batch_size=int(args.batch_size),
        ):
            for row, e in zip(meta, emb):
                row2 = dict(row)
                row2["embedding"] = e.tolist()
                f.write(json.dumps(row2) + "\n")


if __name__ == "__main__":
    main()

