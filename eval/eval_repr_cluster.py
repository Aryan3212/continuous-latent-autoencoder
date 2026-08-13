"""Cluster-structure comparison across models: t-SNE *and* UMAP, colored by
UTMOSv2 MOS.

For each model we project its mean-pooled per-utterance embeddings to 2-D with
both t-SNE and UMAP and draw them as paired columns (one row per model). Points
are colored by the UTMOSv2 predicted MOS (audio naturalness/quality), so you can
eyeball whether the different encoders organize utterances by quality in a
similar way.

    uv run python -m eval.eval_repr_cluster [--max-utts 300] [--models ours,mimi,...]

Writes ``runs/eval/cluster_tsne_umap.png``.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval.repr_bench import (
    DEFAULT_MODELS,
    EVAL_DIR,
    MODEL_ORDER,
    compute_utmos_scores,
    extract,
    load_utterances,
    utmos_runtime_identity,
)


def _tsne_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, (len(X) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        random_state=seed,
    ).fit_transform(X)


def _umap_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    import umap

    n_neighbors = min(15, max(2, len(X) - 1))
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    ).fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--source", default="openslr53", choices=["openslr53", "cv", "subesco"])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated subset of: " + ",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None, help="Local path or HF repo for our model.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subesco-dir", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=EVAL_DIR / "cluster_tsne_umap.png")
    ap.add_argument("--metadata-out", type=pathlib.Path, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.max_utts < 6:
        ap.error("--max-utts must be at least 6 for t-SNE")
    if args.source == "subesco" and not args.subesco_dir:
        ap.error("--subesco-dir is required with --source subesco")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_utterances(
        args.source,
        max_utts=args.max_utts,
        subesco_dir=args.subesco_dir,
        seed=args.seed,
    )

    # Coloring: UTMOSv2 MOS per utterance (shared across all panels).
    mos = compute_utmos_scores(utts, use_cache=not args.no_cache)

    reducers = [("t-SNE", _tsne_2d), ("UMAP", _umap_2d)]
    nrows, ncols = len(models), len(reducers)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.3 * nrows), squeeze=False
    )

    sc = None
    for r, name in enumerate(models):
        data = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)
        X = data["X"]
        for c, (label, fn) in enumerate(reducers):
            ax = axes[r][c]
            emb2d = fn(X, seed=args.seed)
            sc = ax.scatter(
                emb2d[:, 0], emb2d[:, 1], c=mos, cmap="viridis", s=14, alpha=0.85
            )
            ax.set_title(f"{name} — {label}")
            ax.set_xticks([])
            ax.set_yticks([])

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.6, location="right")
        cbar.set_label("UTMOSv2 MOS")

    fig.suptitle(
        f"t-SNE vs UMAP, colored by UTMOSv2 MOS "
        f"({len(utts)} utts; MOS {mos.min():.2f}–{mos.max():.2f})",
        fontsize=13,
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    metadata_out = args.metadata_out or out.with_suffix(".json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(
            {
                "source": args.source,
                "subesco_dir": str(pathlib.Path(args.subesco_dir).resolve()) if args.subesco_dir else None,
                "checkpoint": args.ckpt,
                "models": models,
                "num_utterances": len(utts),
                "seed": args.seed,
                "reducers": ["t-SNE", "UMAP"],
                "color": "UTMOSv2 predicted MOS of source audio",
                "utmosv2_identity": utmos_runtime_identity(),
                "mos_min": float(mos.min()),
                "mos_max": float(mos.max()),
                "figure": str(out.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
