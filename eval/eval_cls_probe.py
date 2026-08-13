from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from eval.common import embedding_stats, iter_embeddings_masked, load_frozen_encoder


def _sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | pathlib.Path, *, include_sha256: bool = False) -> Dict[str, Any]:
    resolved = pathlib.Path(path).resolve()
    stat = resolved.stat()
    identity: Dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        identity["sha256"] = _sha256_file(resolved)
    return identity


def _speaker_id(meta: Dict[str, Any]) -> str | None:
    for key in ("speaker_id", "speaker", "client_id"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _resolved_audio_path(meta: Dict[str, Any], manifest: str) -> str:
    raw_path = str(meta.get("audio_filepath", ""))
    path = pathlib.Path(raw_path)
    if raw_path and not path.is_absolute():
        manifest_parent = pathlib.Path(manifest).resolve().parent
        direct = manifest_parent / path
        parent = manifest_parent.parent / path
        path = direct if direct.exists() or not parent.exists() else parent
    return str(path.resolve()) if raw_path else ""


def _item_id(index: int, meta: Dict[str, Any], manifest: str) -> str:
    """Return a model-independent ID for paired resampling of one manifest row."""
    source = _resolved_audio_path(meta, manifest)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"item_{index:06d}_{digest}"


def _split_sha256(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(
        [
            {
                "item_id": row["item_id"],
                "speaker_id": row["speaker_id"],
                "gold": row["gold"],
            }
            for row in rows
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--predictions-out",
        "--predictions_out",
        dest="predictions_out",
        default=None,
        help="Per-dev-item JSONL output (default: <out>.predictions.jsonl).",
    )
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if min(args.steps, args.hidden, args.batch_size) < 1 or args.lr <= 0:
        ap.error("--steps, --hidden, --batch_size, and --lr must be positive")
    label_key = args.label_key
    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    loaded_checkpoint_step = lm.checkpoint_step
    seg = args.segment_seconds if args.segment_seconds is not None else lm.cfg.data.segment_seconds
    if seg <= 0:
        ap.error("--segment_seconds must be positive")

    print(f"  [{label_key}] Extracting train embeddings...", flush=True)
    x_tr, y_tr, train_meta, label_map = _load_embs(lm, args.train_manifest, label_key, args.batch_size, seg, log_name=f"{label_key} train")
    print(f"  [{label_key}] Extracting dev embeddings...", flush=True)
    x_de, y_de, dev_meta, _ = _load_embs(lm, args.dev_manifest, label_key, args.batch_size, seg, log_name=f"{label_key} dev", label_map=label_map)

    # Free frozen encoder
    del lm
    torch.cuda.empty_cache()

    # Collapse gauge: participation-ratio effective rank of train embeddings
    emb_stats = embedding_stats(x_tr)
    print(f"  [{label_key}] Embedding effective rank: {emb_stats['embed_effective_rank']:.2f} / {emb_stats['embed_dim']}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_map)
    print(f"  [{label_key}] Train: {x_tr.shape[0]}, Dev: {x_de.shape[0]}, Classes: {num_classes}", flush=True)
    # Extraction may consume RNG differently across devices or future cache paths.
    # Reset immediately before probe construction so --seed identifies the probe.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    head = nn.Sequential(nn.Linear(x_tr.size(1), args.hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(args.hidden, num_classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    head.train()
    t0 = time.perf_counter()
    log_interval = max(1, args.steps // 5)
    for step_i in range(args.steps):
        idx = torch.randint(0, x_tr.size(0), (args.batch_size,))
        xb = x_tr[idx].to(device)
        yb = y_tr[idx].to(device)
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
        preds = []
        for start in range(0, x_de.size(0), args.batch_size):
            xb = x_de[start : start + args.batch_size].to(device)
            preds.append(head(xb).argmax(dim=-1).cpu())
        pred = torch.cat(preds)
        acc = (pred == y_de).float().mean().item()
        mf1 = _macro_f1(y_de, pred, num_classes)

    print(f"  [{label_key}] Accuracy: {acc:.4f}, Macro-F1: {mf1:.4f}", flush=True)
    id_to_label = {index: label for label, index in label_map.items()}
    prediction_rows = [
        {
            "item_id": _item_id(index, meta, args.dev_manifest),
            "speaker_id": _speaker_id(meta),
            "source_audio_filepath": _resolved_audio_path(meta, args.dev_manifest),
            "gold": id_to_label[int(gold)],
            "prediction": id_to_label[int(predicted)],
            "gold_label": id_to_label[int(gold)],
            "predicted_label": id_to_label[int(predicted)],
            "gold_index": int(gold),
            "predicted_index": int(predicted),
        }
        for index, (meta, gold, predicted) in enumerate(
            zip(dev_meta, y_de.tolist(), pred.tolist())
        )
    ]
    if len({row["item_id"] for row in prediction_rows}) != len(prediction_rows):
        raise RuntimeError("Dev manifest does not yield unique item IDs")
    split_sha256 = _split_sha256(prediction_rows)
    out_path = pathlib.Path(args.out)
    predictions_path = (
        pathlib.Path(args.predictions_out)
        if args.predictions_out
        else out_path.with_suffix(".predictions.jsonl")
    )
    _write_jsonl(predictions_path, prediction_rows)
    out = {
        "protocol": "frozen_embedding_classification_probe_v2",
        "label_key": label_key,
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "num_classes": num_classes,
        "num_train": int(x_tr.size(0)),
        "num_dev": int(x_de.size(0)),
        "label_map": label_map,
        "embed_dim": emb_stats["embed_dim"],
        "embed_effective_rank": emb_stats["embed_effective_rank"],
        "checkpoint": str(args.ckpt),
        "checkpoint_step": loaded_checkpoint_step,
        "segment_seconds": float(seg),
        "probe_steps": int(args.steps),
        "hidden": int(args.hidden),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "seed": int(args.seed),
        "predictions_descriptor": {
            "path": str(predictions_path.resolve()),
            "format": "jsonl",
            "schema": "classification_predictions_v1",
            "count": len(prediction_rows),
            "split_sha256": split_sha256,
        },
        "predictions_artifact": str(predictions_path.resolve()),
        "data_provenance": {
            "train_manifest": _file_identity(args.train_manifest, include_sha256=True),
            "dev_manifest": _file_identity(args.dev_manifest, include_sha256=True),
            "checkpoint": _file_identity(args.ckpt),
            "checkpoint_step": loaded_checkpoint_step,
            "train_item_count": len(train_meta),
            "dev_split_sha256": split_sha256,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
