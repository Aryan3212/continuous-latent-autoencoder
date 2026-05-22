from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from eval.common import iter_embeddings, load_frozen_encoder


def _build_label_map(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    labels = sorted({r[key] for r in rows})
    return {lbl: i for i, lbl in enumerate(labels)}


def _load_embs(
    lm,
    manifest: str,
    label_key: str,
    batch_size: int,
    segment_seconds: float,
    log_name: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    xs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    for emb, meta in iter_embeddings(
        lm,
        manifest,
        sample_rate=lm.cfg.data.sample_rate,
        segment_seconds=segment_seconds,
        batch_size=batch_size,
        log_name=log_name,
    ):
        xs.append(emb)
        metas.extend(meta)

    label_map = _build_label_map(metas, label_key)
    ys = torch.tensor([label_map[m[label_key]] for m in metas], dtype=torch.long)
    return torch.cat(xs, dim=0), ys, label_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--label_key", default="gender")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    seg = args.segment_seconds if args.segment_seconds is not None else lm.cfg.data.segment_seconds

    print("  [Gender] Extracting train embeddings...", flush=True)
    x_tr, y_tr, label_map = _load_embs(lm, args.train_manifest, args.label_key, args.batch_size, seg, log_name="Gender train")
    print("  [Gender] Extracting dev embeddings...", flush=True)
    x_de, y_de, _ = _load_embs(lm, args.dev_manifest, args.label_key, args.batch_size, seg, log_name="Gender dev")

    # Free frozen encoder
    del lm
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_de, y_de = x_de.to(device), y_de.to(device)

    num_classes = len(label_map)
    print(f"  [Gender] Train: {x_tr.shape[0]}, Dev: {x_de.shape[0]}, Classes: {num_classes}", flush=True)
    head = nn.Sequential(nn.Linear(x_tr.size(1), 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes)).to(device)
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
            print(f"  [Gender] step {step_i+1}/{args.steps}  loss={loss.item():.4f}  ({elapsed:.1f}s)", flush=True)

    head.eval()
    with torch.no_grad():
        pred = head(x_de).argmax(dim=-1)
        acc = (pred == y_de).float().mean().item()

    print(f"  [Gender] Accuracy: {acc:.4f}", flush=True)
    out = {"accuracy": float(acc), "num_classes": num_classes, "num_train": int(x_tr.size(0)), "num_dev": int(x_de.size(0))}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
