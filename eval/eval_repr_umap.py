"""UMAP of mean-pooled encoder latents, colored by speaker.

Projects each model's per-utterance embeddings to 2-D with UMAP and draws one
scatter panel per model in a single figure, points colored by speaker id.
Well-structured speaker representations form tight, separated clusters.

    uv run python -m eval.eval_repr_umap [--max-utts 300] [--models ours,mimi,...]

Writes ``runs/eval/umap_speakers.png``.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, extract, load_utterances


def _umap_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    import umap

    n_neighbors = min(15, max(2, len(X) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(X)


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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    utts = load_utterances(args.source, max_utts=args.max_utts)

    # Stable per-speaker colour across all panels.
    speakers_all = sorted({u.speaker for u in utts})
    spk_to_color = {s: i for i, s in enumerate(speakers_all)}
    cmap = plt.get_cmap("tab20", max(len(speakers_all), 1))

    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for ax_idx, name in enumerate(models):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        data = extract(name, utts, ckpt=args.ckpt, use_cache=not args.no_cache)
        emb2d = _umap_2d(data["X"])
        colors = [spk_to_color[s] for s in data["speakers"]]
        ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, cmap=cmap, s=12, alpha=0.8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused panels.
    for k in range(len(models), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(
        f"UMAP of mean-pooled latents, colored by speaker "
        f"({len(utts)} utts, {len(speakers_all)} speakers)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "umap_speakers.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
