from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval.common import iter_embeddings, load_frozen_encoder
from utils.logging import JsonlLogger, maybe_init_wandb

def _build_label_map(manifest: str, key: str) -> Dict[str, int]:
    labels = set()
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if key in row:
                labels.add(row[key])
    return {lbl: i for i, lbl in enumerate(sorted(list(labels)))}

def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum().item()
        fp = ((y_true != c) & (y_pred == c)).sum().item()
        fn = ((y_true == c) & (y_pred != c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / max(1, len(f1s)))

def _load_data(
    lm,
    manifest: str,
    label_key: str,
    label_map: Dict[str, int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    
    # We use pooled embeddings from common.py
    for emb, meta in iter_embeddings(
        lm,
        manifest,
        sample_rate=int(lm.cfg["data"]["sample_rate"]),
        segment_seconds=float(lm.cfg["data"]["segment_seconds"]),
        batch_size=batch_size,
    ):
        y = torch.tensor([label_map[m[label_key]] for m in meta if label_key in m], dtype=torch.long)
        # Handle cases where some rows might miss the key (though they shouldn't)
        if y.size(0) == emb.size(0):
            xs.append(emb)
            ys.append(y)
        else:
            # Filter emb if needed
            valid_indices = [i for i, m in enumerate(meta) if label_key in m]
            if valid_indices:
                xs.append(emb[valid_indices])
                ys.append(torch.tensor([label_map[meta[i][label_key]] for i in valid_indices], dtype=torch.long))

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--label_key", required=True, help="Key in manifest for categorical label")
    parser.add_argument("--out_dir", default="runs/classifier_probes")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--wandb_project", default="continuous-latent-ae-probes")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.label_key}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = pathlib.Path(args.out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl = JsonlLogger(str(out_root / "train.jsonl"))

    lm = load_frozen_encoder(args.config, args.ckpt, [])
    device = lm.device

    wb_cfg = {
        "run": {"wandb": {"enabled": True, "project": args.wandb_project, "name": run_id}},
        "probe": vars(args),
        "encoder_ckpt": args.ckpt
    }
    wb = maybe_init_wandb(wb_cfg, run_id, str(out_root))

    print(f"Loading data for {args.label_key}...")
    label_map = _build_label_map(args.train_manifest, args.label_key)
    num_classes = len(label_map)
    print(f"Classes: {label_map}")

    x_tr, y_tr = _load_data(lm, args.train_manifest, args.label_key, label_map, args.batch_size)
    x_val, y_val = _load_data(lm, args.val_manifest, args.label_key, label_map, args.batch_size)
    
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    
    print(f"Train size: {x_tr.size(0)}, Val size: {x_val.size(0)}")

    # Simple MLP head
    in_dim = x_tr.size(1)
    head = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    ).to(device)
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for step in range(args.steps + 1):
        head.train()
        idx = torch.randint(0, x_tr.size(0), (min(args.batch_size, x_tr.size(0)),), device=device)
        xb, yb = x_tr[idx], y_tr[idx]
        
        logits = head(xb)
        loss = criterion(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            acc = (logits.argmax(dim=-1) == yb).float().mean().item()
            stats = {"step": step, "loss": loss.item(), "train_acc": acc}
            jsonl.log(stats)
            if wb: wb.log(stats, step=step)
            print(f"Step {step}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

        if step > 0 and step % args.val_interval == 0:
            head.eval()
            with torch.no_grad():
                v_logits = head(x_val)
                v_loss = criterion(v_logits, y_val).item()
                v_preds = v_logits.argmax(dim=-1)
                v_acc = (v_preds == y_val).float().mean().item()
                v_f1 = _macro_f1(y_val, v_preds, num_classes)
            
            print(f">>> Step {step}, Val Acc: {v_acc:.4f}, Val F1: {v_f1:.4f}")
            if wb: wb.log({"val_acc": v_acc, "val_f1": v_f1, "val_loss": v_loss}, step=step)
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save({
                    "step": step,
                    "head": head.state_dict(),
                    "label_map": label_map,
                    "acc": best_acc,
                    "f1": v_f1
                }, str(out_root / "best_head.pt"))

    torch.save({"head": head.state_dict(), "label_map": label_map}, str(out_root / "last_head.pt"))

if __name__ == "__main__":
    main()
