from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from eval.common import iter_embeddings, load_frozen_encoder


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
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    # Gather meta first to build consistent label map.
    metas: List[Dict[str, Any]] = []
    for _, meta in iter_embeddings(
        lm,
        manifest,
        sample_rate=int(lm.cfg["data"]["sample_rate"]),
        segment_seconds=segment_seconds,
        batch_size=batch_size,
    ):
        metas.extend(meta)

    label_map = _build_label_map(metas, label_key)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    meta_i = 0
    for emb, meta in iter_embeddings(
        lm,
        manifest,
        sample_rate=int(lm.cfg["data"]["sample_rate"]),
        segment_seconds=segment_seconds,
        batch_size=batch_size,
    ):
        b = emb.size(0)
        y = torch.tensor([label_map[metas[meta_i + j][label_key]] for j in range(b)], dtype=torch.long)
        meta_i += b
        xs.append(emb)
        ys.append(y)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), label_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--label_key", default="emotion")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    seg = float(args.segment_seconds if args.segment_seconds is not None else lm.cfg["data"]["segment_seconds"])

    x_tr, y_tr, label_map = _load_embs(lm, args.train_manifest, args.label_key, args.batch_size, seg)
    x_de, y_de, _ = _load_embs(lm, args.dev_manifest, args.label_key, args.batch_size, seg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_de, y_de = x_de.to(device), y_de.to(device)

    num_classes = len(label_map)
    head = nn.Sequential(nn.Linear(x_tr.size(1), 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, num_classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    head.train()
    for _ in range(args.steps):
        idx = torch.randint(0, x_tr.size(0), (args.batch_size,), device=device)
        xb, yb = x_tr[idx], y_tr[idx]
        logits = head(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(x_de).argmax(dim=-1)
        acc = (pred == y_de).float().mean().item()
        mf1 = _macro_f1(y_de, pred, num_classes)

    out = {"accuracy": float(acc), "macro_f1": float(mf1), "num_classes": num_classes, "num_train": int(x_tr.size(0)), "num_dev": int(x_de.size(0))}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

