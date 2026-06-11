from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from eval.common import checkpoint_step, embedding_stats, iter_embeddings_masked, load_frozen_encoder


def _build_label_map(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    labels = sorted({r[key] for r in rows})
    return {lbl: i for i, lbl in enumerate(labels)}


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum().item()
        fp = ((y_true != c) & (y_pred == c)).sum().item()
        fn = ((y_true == c) & (y_pred != c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / max(1, len(f1s)))


def _load_embs(
    lm,
    manifest: str,
    label_key: str,
    batch_size: int,
    segment_seconds: float,
    log_name: str = "",
    label_map: Dict[str, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], Dict[str, int]]:
    # Single pass: collect embeddings and metadata together
    xs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    for emb, meta in iter_embeddings_masked(
        lm,
        manifest,
        sample_rate=lm.cfg.data.sample_rate,
        segment_seconds=segment_seconds,
        batch_size=batch_size,
        log_name=log_name,
    ):
        xs.append(emb)
        metas.extend(meta)

    x = torch.cat(xs, dim=0)
    if label_map is None:
        label_map = _build_label_map(metas, label_key)
    keep = [i for i, m in enumerate(metas) if m[label_key] in label_map]
    if len(keep) != len(metas):
        print(f"  [{log_name or label_key}] Dropping {len(metas) - len(keep)} samples with labels outside the train label set", flush=True)
        x = x[keep]
        metas = [metas[i] for i in keep]
    ys = torch.tensor([label_map[m[label_key]] for m in metas], dtype=torch.long)
    return x, ys, metas, label_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--label_key", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    label_key = args.label_key
    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    seg = args.segment_seconds if args.segment_seconds is not None else lm.cfg.data.segment_seconds

    print(f"  [{label_key}] Extracting train embeddings...", flush=True)
    x_tr, y_tr, _, label_map = _load_embs(lm, args.train_manifest, label_key, args.batch_size, seg, log_name=f"{label_key} train")
    print(f"  [{label_key}] Extracting dev embeddings...", flush=True)
    x_de, y_de, _, _ = _load_embs(lm, args.dev_manifest, label_key, args.batch_size, seg, log_name=f"{label_key} dev", label_map=label_map)

    # Free frozen encoder
    del lm
    torch.cuda.empty_cache()

    # Collapse gauge: participation-ratio effective rank of train embeddings
    emb_stats = embedding_stats(x_tr)
    print(f"  [{label_key}] Embedding effective rank: {emb_stats['embed_effective_rank']:.2f} / {emb_stats['embed_dim']}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_de, y_de = x_de.to(device), y_de.to(device)

    num_classes = len(label_map)
    print(f"  [{label_key}] Train: {x_tr.shape[0]}, Dev: {x_de.shape[0]}, Classes: {num_classes}", flush=True)
    head = nn.Sequential(nn.Linear(x_tr.size(1), args.hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(args.hidden, num_classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    head.train()
    t0 = time.perf_counter()
    log_interval = max(1, args.steps // 5)
    for step_i in range(args.steps):
        idx = torch.randint(0, x_tr.size(0), (args.batch_size,), device=device)
        xb, yb = x_tr[idx], y_tr[idx]
        logits = head(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step_i + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{label_key}] step {step_i+1}/{args.steps}  loss={loss.item():.4f}  ({elapsed:.1f}s)", flush=True)

    head.eval()
    with torch.no_grad():
        pred = head(x_de).argmax(dim=-1)
        acc = (pred == y_de).float().mean().item()
        mf1 = _macro_f1(y_de, pred, num_classes)

    print(f"  [{label_key}] Accuracy: {acc:.4f}, Macro-F1: {mf1:.4f}", flush=True)
    out = {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "num_classes": num_classes,
        "num_train": int(x_tr.size(0)),
        "num_dev": int(x_de.size(0)),
        "label_map": label_map,
        "embed_dim": emb_stats["embed_dim"],
        "embed_effective_rank": emb_stats["embed_effective_rank"],
        "checkpoint": str(args.ckpt),
        "checkpoint_step": checkpoint_step(args.ckpt),
        "segment_seconds": float(seg),
        "probe_steps": int(args.steps),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
