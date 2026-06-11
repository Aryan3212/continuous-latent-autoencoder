from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from jiwer import wer

from eval.common import build_charset, greedy_decode_ctc, iter_frame_features, load_frozen_encoder


def _encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[c] for c in text.lower() if c in vocab]


def _min_ctc_frames(target: List[int]) -> int:
    # CTC needs >= one input frame per label, plus one extra frame per adjacent
    # repeated label (a blank must separate repeats).
    return len(target) + sum(1 for a, b in zip(target, target[1:]) if a == b)


def _filter_manifest_by_duration(
    manifest: str, max_seconds: float, out_path: pathlib.Path, log_name: str
) -> Tuple[str, int, int, int]:
    """Write a filtered copy of `manifest`, dropping rows with duration > max_seconds.

    Rationale: iter_frame_features start-crops audio to segment_seconds (see
    _start_crop in data/dataset.py) while the probe keeps the FULL transcript as
    CTC target, so utterances longer than the segment would train on misaligned
    audio/text. Drop them before feature extraction.

    Missing-duration policy: rows with a missing, non-numeric, or non-positive
    `duration` (placeholders) are KEPT — we cannot prove they are too long, and
    dropping them would empty manifests that simply lack durations — but they
    are counted and reported so the user can audit them.

    AudioDataset resolves relative audio_filepath against the manifest's parent
    directory, so paths are absolutized before relocating the manifest next to
    --out.
    """
    root = pathlib.Path(manifest).resolve().parent
    kept = dropped = unknown = 0
    out_lines: List[str] = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dur = row.get("duration")
            if not isinstance(dur, (int, float)) or isinstance(dur, bool) or dur <= 0:
                unknown += 1
            elif dur > max_seconds:
                dropped += 1
                continue
            p = row.get("audio_filepath")
            if isinstance(p, str) and not os.path.isabs(p):
                row["audio_filepath"] = str(root / p)
            out_lines.append(json.dumps(row, ensure_ascii=False))
            kept += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    msg = f"  [ASR] {log_name}: kept {kept} rows, dropped {dropped} rows over {max_seconds:.1f}s"
    if unknown:
        msg += f" ({unknown} rows with missing/placeholder duration kept, unaudited)"
    print(msg, flush=True)
    return str(out_path), kept, dropped, unknown


class _BiLSTMHead(nn.Module):
    """Single bidirectional LSTM -> Linear. Stronger than the linear probe; use
    only when probe purity is not the point."""

    def __init__(self, dim: int, vocab_size: int, hidden: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, vocab_size)

    def forward(self, x: torch.Tensor, lens: torch.Tensor | None = None) -> torch.Tensor:
        # Pack when valid lengths are known so the backward direction starts
        # at the last real frame instead of carrying state across padding.
        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1)
            )
        else:
            x, _ = self.lstm(x)
        return self.proj(x)


def _load_feats_and_text(
    lm,
    manifest: str,
    *,
    text_key: str,
    batch_size: int,
    segment_seconds: float,
    chunk_seconds: float | None,
    source: str,
    log_name: str = "",
    max_samples: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    feats_list: List[torch.Tensor] = []
    lens_list: List[torch.Tensor] = []
    texts: List[str] = []
    n = 0
    for feats, lens, meta in iter_frame_features(
        lm,
        manifest,
        sample_rate=lm.cfg.data.sample_rate,
        segment_seconds=segment_seconds,
        batch_size=batch_size,
        chunk_seconds=chunk_seconds,
        source=source,
        log_name=log_name,
    ):
        feats_list.append(feats)  # already on CPU from iter_frame_features
        lens_list.append(lens)
        texts.extend([m[text_key] for m in meta])
        n += feats.size(0)
        if max_samples > 0 and n >= max_samples:
            break
    all_feats = torch.cat(feats_list, dim=0)
    all_lens = torch.cat(lens_list, dim=0)
    if max_samples > 0 and all_feats.size(0) > max_samples:
        all_feats = all_feats[:max_samples]
        all_lens = all_lens[:max_samples]
        texts = texts[:max_samples]
    return all_feats, all_lens, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--max_samples", type=int, default=0, help="Cap train/dev samples (0=unlimited)")
    ap.add_argument("--upsample_factor", type=int, default=4,
                    help="repeat_interleave features along time before the CTC head (1=off)")
    ap.add_argument("--max_utt_seconds", type=float, default=None,
                    help="Drop manifest rows longer than this (default: effective segment_seconds)")
    ap.add_argument("--chunk_seconds", type=float, default=None,
                    help="Encode audio in independent windows of this length and concatenate "
                         "features (default: pretraining data.segment_seconds; <=0 disables). "
                         "The encoder has unmasked global attention and only ever saw "
                         "segment-length inputs, so longer single passes are OOD.")
    ap.add_argument("--head", choices=["linear", "bilstm"], default="linear",
                    help="Probe head: linear (pure probe) or bilstm (1x BiLSTM-256 -> Linear)")
    ap.add_argument("--features", choices=["encoder", "frontend", "mel"], default="encoder",
                    help="encoder: the model under test; frontend: conv frontend only "
                         "(does phonetic info exist before the conformer?); mel: log-mel "
                         "fbank control bypassing the model — verifies the probe harness "
                         "and gives a ceiling (50 Hz frames: use --upsample_factor 1)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    lm = load_frozen_encoder(args.config, args.ckpt, args.overrides)
    # eval.asr.segment_seconds, NOT data.segment_seconds: pretraining crops to
    # short segments (e.g. 2.5s), but the probe needs the full utterance or the
    # transcript no longer matches the start-cropped audio.
    seg = args.segment_seconds if args.segment_seconds is not None else lm.cfg.eval.asr.segment_seconds
    max_utt = args.max_utt_seconds if args.max_utt_seconds is not None else seg
    chunk = args.chunk_seconds if args.chunk_seconds is not None else lm.cfg.data.segment_seconds
    if chunk <= 0:
        chunk = None
    print(f"  [ASR] segment_seconds={seg:g}, max_utt_seconds={max_utt:g}, "
          f"chunk_seconds={'off' if chunk is None else f'{chunk:g}'}, "
          f"features={args.features}", flush=True)
    upf = max(1, int(args.upsample_factor))

    # Drop utterances longer than max_utt BEFORE extraction: _start_crop would
    # truncate their audio while the full transcript stays the CTC target.
    out_path = pathlib.Path(args.out)
    train_manifest, _, n_filtered_tr, n_unknown_tr = _filter_manifest_by_duration(
        args.train_manifest, max_utt, out_path.with_suffix(".train_filtered.jsonl"), "Filter train"
    )
    dev_manifest, _, n_filtered_de, n_unknown_de = _filter_manifest_by_duration(
        args.dev_manifest, max_utt, out_path.with_suffix(".dev_filtered.jsonl"), "Filter dev"
    )

    if args.dry_run:
        feats_iter = iter_frame_features(
            lm,
            train_manifest,
            sample_rate=lm.cfg.data.sample_rate,
            segment_seconds=seg,
            batch_size=args.batch_size,
            chunk_seconds=chunk,
            source=args.features,
        )
        feats, _, meta = next(feats_iter)
        out = {
            "dry_run": True,
            "feats_shape": list(feats.shape),
            "num_samples": len(meta),
        }
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # Free the frozen encoder before loading features — we don't need it after extraction
    max_s = args.max_samples
    print(f"  [ASR] Extracting train features{f' (max {max_s})' if max_s else ''}...", flush=True)
    feats_tr, lens_tr, text_tr = _load_feats_and_text(
        lm,
        train_manifest,
        text_key=args.text_key,
        batch_size=args.batch_size,
        segment_seconds=seg,
        chunk_seconds=chunk,
        source=args.features,
        log_name="ASR train",
        max_samples=max_s,
    )
    print(f"  [ASR] Extracting dev features{f' (max {max_s})' if max_s else ''}...", flush=True)
    feats_de, lens_de, text_de = _load_feats_and_text(
        lm,
        dev_manifest,
        text_key=args.text_key,
        batch_size=args.batch_size,
        segment_seconds=seg,
        chunk_seconds=chunk,
        source=args.features,
        log_name="ASR dev",
        max_samples=max_s,
    )

    # Free frozen encoder to reclaim GPU memory for probe training
    del lm
    torch.cuda.empty_cache()

    print(f"  [ASR] Train: {feats_tr.shape}, Dev: {feats_de.shape}", flush=True)

    # Charset is a function of the training manifest's text — cache next to the
    # manifest so probe-CTC training across runs reuses the same vocab.
    charset_path = pathlib.Path(args.train_manifest + ".charset.json")
    if charset_path.exists():
        charset = json.loads(charset_path.read_text(encoding="utf-8"))
        print(f"  [ASR] Loaded cached charset ({len(charset)} symbols) from {charset_path}", flush=True)
    else:
        charset = build_charset(text_tr)
        charset_path.write_text(json.dumps(charset, ensure_ascii=False), encoding="utf-8")
        print(f"  [ASR] Built and cached charset ({len(charset)} symbols) at {charset_path}", flush=True)
    vocab = {c: i for i, c in enumerate(charset)}
    id2ch = charset

    targets_tr = [torch.tensor(_encode(t, vocab), dtype=torch.long) for t in text_tr]

    # Feasibility accounting: zero_infinity=True silently zeroes the loss of any
    # sample whose (upsampled) VALID input is shorter than its CTC-minimum
    # target length — the user must SEE how many samples never contribute
    # gradient. Uses per-sample valid lengths, not the padded frame count.
    def _count_infeasible(lens: torch.Tensor, encoded: List[List[int]]) -> int:
        return sum(1 for t, L in zip(encoded, lens.tolist()) if _min_ctc_frames(t) > L * upf)

    enc_tr = [t.tolist() for t in targets_tr]
    enc_de = [_encode(t, vocab) for t in text_de]
    n_inf_tr = _count_infeasible(lens_tr, enc_tr)
    n_inf_de = _count_infeasible(lens_de, enc_de)
    pct_inf_tr = 100.0 * n_inf_tr / max(1, len(enc_tr))
    pct_inf_de = 100.0 * n_inf_de / max(1, len(enc_de))
    if n_inf_tr or n_inf_de:
        print(f"  [ASR] WARNING: CTC-infeasible samples at upsample x{upf} "
              f"(per-sample valid frames x {upf}): "
              f"train {n_inf_tr}/{len(enc_tr)} ({pct_inf_tr:.1f}%), "
              f"dev {n_inf_de}/{len(enc_de)} ({pct_inf_de:.1f}%). "
              f"zero_infinity=True silently zeroes their training loss — "
              f"consider a larger --upsample_factor.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep features on CPU — only move mini-batches to GPU during training
    # This is critical: feats_tr can be >1GB and would OOM on a 16GB card

    if args.head == "bilstm":
        head: nn.Module = _BiLSTMHead(feats_tr.size(-1), len(charset)).to(device)
    else:
        head = nn.Linear(feats_tr.size(-1), len(charset)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    head.train()
    n = feats_tr.size(0)
    t0 = time.perf_counter()
    log_interval = max(1, args.steps // 10)
    for step_i in range(args.steps):
        idx = torch.randint(0, n, (args.batch_size,))
        xb = feats_tr[idx].to(device)  # (B,T,D) — only batch on GPU
        # Time-upsampling: at ~12.5 Hz frames, Bengali transcripts (~8-15
        # chars/s) routinely EXCEED the CTC input length, making alignment
        # infeasible. repeat_interleave adds no information (honest probe);
        # it only gives CTC enough timesteps to emit every character. Done
        # per-batch on GPU: x4 quadruples per-batch head memory, fine for a
        # linear head, while stored features stay compact on CPU.
        if upf > 1:
            xb = xb.repeat_interleave(upf, dim=1)
        # Valid (real-audio) lengths: CTC must not be allowed to align target
        # characters into the padding region.
        ulens = (lens_tr[idx] * upf).clamp(max=xb.size(1))
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = head(xb, ulens) if isinstance(head, _BiLSTMHead) else head(xb)
        log_probs = logits.float().log_softmax(dim=-1)  # (B,T,V), CTC in fp32
        input_lens = ulens.to(device)
        yb = [targets_tr[i] for i in idx.tolist()]
        target_lens = torch.tensor([t.numel() for t in yb], dtype=torch.long, device=device)
        ycat = torch.cat([t.to(device) for t in yb], dim=0) if target_lens.sum().item() > 0 else torch.zeros((0,), dtype=torch.long, device=device)
        loss = ctc(log_probs.transpose(0, 1), ycat, input_lens, target_lens)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (step_i + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            rate = (step_i + 1) / elapsed
            eta = (args.steps - step_i - 1) / rate
            print(f"  [ASR] step {step_i+1}/{args.steps}  loss={loss.item():.4f}  "
                  f"({rate:.0f} steps/s, ETA {eta:.0f}s)", flush=True)

    def _eval(feats: torch.Tensor, lens: torch.Tensor, texts: List[str], eval_batch: int = 256) -> Dict[str, Any]:
        head.eval()
        all_hyp: List[str] = []
        # Upsampling multiplies frames per sample, so shrink the eval batch to
        # keep the same GPU memory envelope.
        eval_batch = max(1, eval_batch // upf)
        with torch.no_grad():
            # Batch the eval to avoid OOM on large feature tensors
            for start in range(0, feats.size(0), eval_batch):
                xb = feats[start : start + eval_batch].to(device)
                if upf > 1:
                    # Same time-upsampling as training — the head was trained
                    # on upsampled inputs.
                    xb = xb.repeat_interleave(upf, dim=1)
                ulens = (lens[start : start + eval_batch] * upf).clamp(max=xb.size(1))
                lp = (head(xb, ulens) if isinstance(head, _BiLSTMHead) else head(xb)).log_softmax(dim=-1)
                # Decode only the valid frames — padding must not emit chars.
                all_hyp.extend(greedy_decode_ctc(lp, id2ch, lens=ulens.tolist()))
        w = wer(texts, all_hyp)
        examples = []
        for i in range(min(5, len(texts))):
            examples.append({"ref": texts[i], "hyp": all_hyp[i]})
        return {"wer": float(w), "num_samples": len(texts), "examples": examples}

    print("  [ASR] Evaluating...", flush=True)
    out = {
        "train": _eval(feats_tr, lens_tr, text_tr),
        "dev": _eval(feats_de, lens_de, text_de),
        "vocab_size": len(charset),
        "upsample_factor": upf,
        "max_utt_seconds": float(max_utt),
        "chunk_seconds": chunk,
        "head": args.head,
        "features": args.features,
        "n_filtered_train": n_filtered_tr,
        "n_filtered_dev": n_filtered_de,
        "n_unknown_duration_train": n_unknown_tr,
        "n_unknown_duration_dev": n_unknown_de,
        "n_infeasible": n_inf_tr,
        "pct_infeasible": pct_inf_tr,
        "n_infeasible_dev": n_inf_de,
        "pct_infeasible_dev": pct_inf_de,
    }
    print(f"  [ASR] Train WER: {out['train']['wer']:.4f}, Dev WER: {out['dev']['wer']:.4f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
