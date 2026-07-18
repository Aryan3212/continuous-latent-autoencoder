"""PCA and UMAP of frozen utterance embeddings, coloured by a speech attribute.

Use ``--source openslr53 --color speaker`` for identity geometry, or
``--source subesco --color emotion`` for affective geometry.
"""
from __future__ import annotations

import argparse

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_utterances


def _pca(X: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def _umap(X: np.ndarray, seed: int) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_components=2, n_neighbors=min(15, max(2, len(X) - 1)), min_dist=0.1,
        metric="cosine", random_state=seed,
    ).fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="openslr53", choices=["openslr53", "cv", "subesco"])
    ap.add_argument("--color", default="speaker", choices=["speaker", "emotion"])
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    utts = load_utterances(args.source, max_utts=args.max_utts)
    labels = [getattr(u, args.color) for u in utts]
    if any(label is None for label in labels):
        raise ValueError(f"{args.source} does not provide a non-empty {args.color} label for every utterance")
    labels = [str(label) for label in labels]
    label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
    colors = np.asarray([label_to_id[label] for label in labels])
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(models), 2, figsize=(10, 4.2 * len(models)), squeeze=False)
    for row, name in enumerate(models):
        X = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)["X"]
        for col, (title, reducer) in enumerate((("PCA", _pca), ("UMAP", _umap))):
            xy = reducer(X, args.seed)
            axes[row, col].scatter(xy[:, 0], xy[:, 1], c=colors, cmap="tab20", s=13, alpha=.8)
            axes[row, col].set_title(f"{name} — {title}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"Frozen embeddings coloured by {args.color} ({len(utts)} utterances)")
    fig.tight_layout(rect=(0, 0, 1, .97))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"pca_umap_{args.source}_{args.color}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
