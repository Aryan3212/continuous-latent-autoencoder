from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from jiwer import wer

from eval.common import iter_frame_features, load_frozen_encoder


def _build_charset(texts: List[str]) -> List[str]:
    chars = set()
    for t in texts:
        chars.update(list(t.lower()))
    chars.discard("\n")
    chars = sorted(chars)
    return ["<blank>"] + chars


def _encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[c] for c in text.lower() if c in vocab]


def _greedy_decode(log_probs: torch.Tensor, id2ch: List[str]) -> List[str]:
    pred = log_probs.argmax(dim=-1)  # (B,T)
    outs = []
    for seq in pred.tolist():
        last = None
        chars = []
        for i in seq:
            if i == 0:
                last = i
                continue
            if last != i:
                chars.append(id2ch[i])
            last = i
        outs.append("".join(chars))
    return outs


def _load_feats_and_text(
    lm,
    manifest: str,
    *,
    text_key: str,
    batch_size: int,
    segment_seconds: float,
    use_latent: bool,
) -> Tuple[torch.Tensor, List[str]]:
    feats_list: List[torch.Tensor] = []
    texts: List[str] = []
    for feats, meta in iter_frame_features(
        lm,
        manifest,
        sample_rate=int(lm.cfg["data"]["sample_rate"]),
        segment_seconds=segment_seconds,
        batch_size=batch_size,
        use_latent=use_latent,
    ):
        feats_list.append(feats)  # (B,T,D)
        texts.extend([m[text_key] for m in meta])
    return torch.cat(feats_list, dim=0), texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--use_latent", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    seg = float(args.segment_seconds if args.segment_seconds is not None else lm.cfg["data"]["segment_seconds"])

    if args.dry_run:
        feats_iter = iter_frame_features(
            lm,
            args.train_manifest,
            sample_rate=int(lm.cfg["data"]["sample_rate"]),
            segment_seconds=seg,
            batch_size=args.batch_size,
            use_latent=bool(args.use_latent),
        )
        feats, meta = next(feats_iter)
        out = {
            "dry_run": True,
            "feats_shape": list(feats.shape),
            "num_samples": len(meta),
            "use_latent": bool(args.use_latent),
        }
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        return

    feats_tr, text_tr = _load_feats_and_text(
        lm,
        args.train_manifest,
        text_key=args.text_key,
        batch_size=args.batch_size,
        segment_seconds=seg,
        use_latent=bool(args.use_latent),
    )
    feats_de, text_de = _load_feats_and_text(
        lm,
        args.dev_manifest,
        text_key=args.text_key,
        batch_size=args.batch_size,
        segment_seconds=seg,
        use_latent=bool(args.use_latent),
    )

    charset = _build_charset(text_tr)
    vocab = {c: i for i, c in enumerate(charset)}
    id2ch = charset

    targets_tr = [torch.tensor(_encode(t, vocab), dtype=torch.long) for t in text_tr]
    targets_de = [torch.tensor(_encode(t, vocab), dtype=torch.long) for t in text_de]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats_tr = feats_tr.to(device)
    feats_de = feats_de.to(device)

    head = nn.Linear(feats_tr.size(-1), len(charset)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    head.train()
    n = feats_tr.size(0)
    for _ in range(args.steps):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        xb = feats_tr[idx]  # (B,T,D)
        log_probs = head(xb).log_softmax(dim=-1)  # (B,T,V)
        input_lens = torch.full((xb.size(0),), xb.size(1), dtype=torch.long, device=device)
        yb = [targets_tr[i] for i in idx.tolist()]
        target_lens = torch.tensor([t.numel() for t in yb], dtype=torch.long, device=device)
        ycat = torch.cat([t.to(device) for t in yb], dim=0) if target_lens.sum().item() > 0 else torch.zeros((0,), dtype=torch.long, device=device)
        loss = ctc(log_probs.transpose(0, 1), ycat, input_lens, target_lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def _eval(feats: torch.Tensor, texts: List[str], targets: List[torch.Tensor]) -> Dict[str, Any]:
        head.eval()
        with torch.no_grad():
            log_probs = head(feats).log_softmax(dim=-1)
            hyp = _greedy_decode(log_probs, id2ch)
            w = wer(texts, hyp)
        return {"wer": float(w), "num_samples": len(texts)}

    out = {
        "train": _eval(feats_tr, text_tr, targets_tr),
        "dev": _eval(feats_de, text_de, targets_de),
        "vocab_size": len(charset),
        "use_latent": bool(args.use_latent),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
