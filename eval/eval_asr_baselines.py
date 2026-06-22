"""Compare frozen-feature ASR (CTC linear/BiLSTM probe) across models.

Trains an identical CTC probe head on top of FROZEN frame features from each
model and reports WER/CER on a held-out dev split, so the only thing that
varies is the representation. Models share the charset, head, optimizer, and
schedule.

Supported feature sources (all expose per-frame ``(T, D)`` features):
  - ours   : our frontend+encoder (via repr_bench.build_embedder; ~12.5 Hz)
  - wavlm  : microsoft/wavlm-base-plus last hidden state (~50 Hz)
  - mimi   : kyutai/mimi continuous pre-quant latent (~12.5 Hz)
  - mel    : 64-bin log-mel + per-utterance CMVN control (hop sets frame rate)
  - utmos  : UTMOSv2's SSL branch — the learned weighted mix of its wav2vec2-base
             hidden states (~50 Hz, 768-d), the same per-frame representation it
             feeds to its MOS head. NOTE this is *not* UTMOSv2's MOS output (a
             scalar quality score has no ASR meaning); it is the content-bearing
             SSL features inside it. Effectively a wav2vec2-base baseline.

Frame rate differs across models, so CTC feasibility (>= 1 input frame per
output char, +1 per repeat) differs too. ``--upsample`` defaults per-model
(time repeat_interleave, adds no information, just timesteps); the run prints
how many samples are CTC-infeasible at the chosen factor.

fp16 autocast is off by default behavior on CUDA but can be disabled with
CLAE_AMP=0 (recommended on GTX 16-series — see eval/common.py:amp_enabled).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from jiwer import cer, wer

from eval.common import amp_enabled, build_charset, greedy_decode_ctc

TARGET_SR = 16000

# Approx encoder frame rate (Hz) per model -> default upsample so Bengali
# transcripts (~8-15 chars/s) fit the CTC input length. WavLM is already 50 Hz.
_DEFAULT_UPSAMPLE = {"ours": 6, "mimi": 6, "mel": 1, "wavlm": 1, "utmos": 1}


# --------------------------------------------------------------------------- #
# Per-utterance frame extractors -> (T, D) float32 numpy
# --------------------------------------------------------------------------- #
def _mel_extractor(mel_hop: int = 320) -> Callable[[torch.Tensor], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fft = 512 if mel_hop <= 512 else 2048
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=n_fft, win_length=n_fft, hop_length=mel_hop, n_mels=64
    ).to(device)

    @torch.no_grad()
    def fn(wav16k: torch.Tensor) -> np.ndarray:
        w = wav16k.to(device).view(1, -1)
        h = torch.log(melspec(w) + 1e-5)  # (1, M, T)
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-5)
        return h.squeeze(0).t().float().cpu().numpy()  # (T, M)

    return fn


def _utmos_extractor() -> Callable[[torch.Tensor], np.ndarray]:
    """Frame features from UTMOSv2's SSL branch: the learned weighted sum of its
    wav2vec2-base hidden states (T, 768) — what it pools for the MOS head. This
    is a content SSL representation, NOT the scalar MOS score."""
    import utmosv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utmosv2.create_model(pretrained=True)
    ssl = model._model.ssl.eval().to(device)
    weights = ssl.weights.detach().to(device)

    @torch.no_grad()
    def fn(wav16k: torch.Tensor) -> np.ndarray:
        hs = ssl.encoder([wav16k.view(-1).cpu()])  # tuple of (1, T, 768)
        feat = sum(t * w for t, w in zip(hs, weights))  # (1, T, 768)
        return feat.squeeze(0).float().cpu().numpy()

    return fn


def build_extractor(name: str, *, ckpt: str | None, mel_hop: int) -> Callable[[torch.Tensor], np.ndarray]:
    """Return a fn mapping a 16 kHz mono waveform (1-D tensor) to (T, D) frames."""
    if name == "mel":
        return _mel_extractor(mel_hop)
    if name in ("ours", "ours_random", "mimi", "wavlm", "mms"):
        # repr_bench's embedders already return (T, D) frame features.
        from eval.repr_bench import build_embedder

        emb = build_embedder(name, ckpt=ckpt)
        return lambda wav16k: np.asarray(emb.fn(wav16k.view(-1)), dtype=np.float32)
    if name in ("utmos", "utmosv2"):
        return _utmos_extractor()
    raise ValueError(f"unknown model {name!r}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_rows(manifest: str, *, text_key: str, max_utt_seconds: float, max_samples: int) -> List[Dict]:
    rows: List[Dict] = []
    dropped = 0
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dur = r.get("duration")
            if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > max_utt_seconds:
                dropped += 1
                continue
            if not r.get(text_key):
                continue
            rows.append(r)
            if max_samples and len(rows) >= max_samples:
                break
    print(f"  [data] {manifest}: {len(rows)} rows kept, {dropped} dropped over {max_utt_seconds:g}s", flush=True)
    return rows


def _load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != TARGET_SR:
        wav = torchaudio.transforms.Resample(int(sr), TARGET_SR)(wav)
    return wav.squeeze(0)  # (S,)


def _extract_split(
    extractor: Callable[[torch.Tensor], np.ndarray],
    rows: List[Dict],
    *,
    text_key: str,
    log_name: str,
) -> Tuple[List[torch.Tensor], List[str]]:
    feats: List[torch.Tensor] = []
    texts: List[str] = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        wav = _load_wav(r["audio_filepath"])
        f = torch.from_numpy(extractor(wav))  # (T, D)
        feats.append(f)
        texts.append(r[text_key])
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  [{log_name}] {i + 1}/{len(rows)} ({rate:.1f} utt/s)", flush=True)
    return feats, texts


# --------------------------------------------------------------------------- #
# CTC probe (variable-length, per-batch padding)
# --------------------------------------------------------------------------- #
class _BiLSTMHead(nn.Module):
    def __init__(self, dim: int, vocab: int, hidden: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, vocab)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.proj(x)


def _min_ctc_frames(target: List[int]) -> int:
    return len(target) + sum(1 for a, b in zip(target, target[1:]) if a == b)


def _pad_batch(feats: List[torch.Tensor], idx: List[int], upf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of (T_i, D) into (B, Tmax*upf, D) with per-sample valid lens."""
    chosen = [feats[i] for i in idx]
    lens = torch.tensor([f.size(0) * upf for f in chosen], dtype=torch.long)
    tmax = int(max(f.size(0) for f in chosen)) * upf
    d = chosen[0].size(1)
    out = torch.zeros(len(chosen), tmax, d)
    for b, f in enumerate(chosen):
        x = f.repeat_interleave(upf, dim=0) if upf > 1 else f
        out[b, : x.size(0)] = x
    return out, lens


def train_and_eval_probe(
    feats_tr: List[torch.Tensor],
    text_tr: List[str],
    feats_de: List[torch.Tensor],
    text_de: List[str],
    *,
    charset: List[str],
    upf: int,
    head_kind: str,
    steps: int,
    lr: float,
    batch_size: int,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = {c: i for i, c in enumerate(charset)}

    def enc(t: str) -> List[int]:
        return [vocab[c] for c in t.lower() if c in vocab]

    tgt_tr = [enc(t) for t in text_tr]
    tgt_de = [enc(t) for t in text_de]

    n_inf = sum(1 for f, t in zip(feats_tr, tgt_tr) if _min_ctc_frames(t) > f.size(0) * upf)
    pct_inf = 100.0 * n_inf / max(1, len(tgt_tr))
    if n_inf:
        print(f"  [probe] WARNING: {n_inf}/{len(tgt_tr)} ({pct_inf:.1f}%) train samples CTC-infeasible "
              f"at upsample x{upf}; their loss is silently zeroed. Raise --upsample.", flush=True)

    d = feats_tr[0].size(1)
    head: nn.Module = _BiLSTMHead(d, len(charset)).to(device) if head_kind == "bilstm" else nn.Linear(d, len(charset)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = amp_enabled(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    head.train()
    n = len(feats_tr)
    t0 = time.perf_counter()
    log_interval = max(1, steps // 10)
    for step_i in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
        xb, ulens = _pad_batch(feats_tr, idx, upf)
        xb = xb.to(device)
        ulens = ulens.clamp(max=xb.size(1))
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = head(xb, ulens.to(device)) if isinstance(head, _BiLSTMHead) else head(xb)
        log_probs = logits.float().log_softmax(dim=-1)
        yb = [tgt_tr[i] for i in idx]
        target_lens = torch.tensor([len(t) for t in yb], dtype=torch.long, device=device)
        ycat = torch.tensor([c for t in yb for c in t], dtype=torch.long, device=device)
        loss = ctc(log_probs.transpose(0, 1), ycat, ulens.to(device), target_lens)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (step_i + 1) % log_interval == 0:
            rate = (step_i + 1) / (time.perf_counter() - t0)
            eta = (steps - step_i - 1) / max(rate, 1e-6)
            print(f"  [probe] step {step_i + 1}/{steps} loss={loss.item():.4f} ({rate:.0f} it/s, ETA {eta:.0f}s)", flush=True)

    def _eval(feats: List[torch.Tensor], texts: List[str], eval_batch: int = 64) -> Dict:
        head.eval()
        eval_batch = max(1, eval_batch // upf)
        hyps: List[str] = []
        with torch.no_grad():
            for start in range(0, len(feats), eval_batch):
                idx = list(range(start, min(start + eval_batch, len(feats))))
                xb, ulens = _pad_batch(feats, idx, upf)
                xb = xb.to(device)
                ulens = ulens.clamp(max=xb.size(1))
                lp = (head(xb, ulens.to(device)) if isinstance(head, _BiLSTMHead) else head(xb)).log_softmax(dim=-1)
                hyps.extend(greedy_decode_ctc(lp, charset, lens=ulens.tolist()))
        return {
            "wer": float(wer(texts, hyps)),
            "cer": float(cer(texts, hyps)),
            "num_samples": len(texts),
            "examples": [{"ref": texts[i], "hyp": hyps[i]} for i in range(min(5, len(texts)))],
        }

    return {
        "train": _eval(feats_tr, text_tr),
        "dev": _eval(feats_de, text_de),
        "vocab_size": len(charset),
        "upsample_factor": upf,
        "head": head_kind,
        "feat_dim": d,
        "n_infeasible_train": n_inf,
        "pct_infeasible_train": pct_inf,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--models", default="ours,wavlm,mimi,utmos,mel",
                    help="comma list of: ours,wavlm,mimi,utmos,mel "
                         "(utmos = its wav2vec2-base SSL branch, not the MOS head)")
    ap.add_argument("--ckpt", default=None, help="our checkpoint (for models=ours)")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--head", choices=["linear", "bilstm"], default="linear")
    ap.add_argument("--upsample", type=int, default=0, help="0 = per-model default")
    ap.add_argument("--mel_hop", type=int, default=320, help="mel frame hop in samples (320=50Hz)")
    ap.add_argument("--max_utt_seconds", type=float, default=12.0)
    ap.add_argument("--max_samples", type=int, default=0, help="cap train/dev rows (0=all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    rows_tr = _load_rows(args.train_manifest, text_key=args.text_key,
                         max_utt_seconds=args.max_utt_seconds, max_samples=args.max_samples)
    rows_de = _load_rows(args.dev_manifest, text_key=args.text_key,
                         max_utt_seconds=args.max_utt_seconds, max_samples=args.max_samples)

    # Charset is fixed across models (from the FULL train text, not feature-derived).
    charset = build_charset([r[args.text_key] for r in rows_tr])
    print(f"  [probe] charset: {len(charset)} symbols", flush=True)

    results: Dict[str, Dict] = {}
    for name in requested:
        print(f"\n=== {name} ===", flush=True)
        try:
            extractor = build_extractor(name, ckpt=args.ckpt, mel_hop=args.mel_hop)
        except Exception as e:  # missing dep / unknown model -> record, keep going
            results[name] = {"error": str(e)}
            print(f"  [skip] {name}: {e}", flush=True)
            continue
        upf = args.upsample or _DEFAULT_UPSAMPLE.get(name, 4)
        print(f"  [extract] train features ({name}, upsample x{upf})...", flush=True)
        feats_tr, text_tr = _extract_split(extractor, rows_tr, text_key=args.text_key, log_name=f"{name} train")
        print(f"  [extract] dev features ({name})...", flush=True)
        feats_de, text_de = _extract_split(extractor, rows_de, text_key=args.text_key, log_name=f"{name} dev")
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        res = train_and_eval_probe(
            feats_tr, text_tr, feats_de, text_de,
            charset=charset, upf=upf, head_kind=args.head,
            steps=args.steps, lr=args.lr, batch_size=args.batch_size,
        )
        results[name] = res
        print(f"  [{name}] dev WER={res['dev']['wer']:.4f} CER={res['dev']['cer']:.4f} "
              f"(dim={res['feat_dim']}, upsample x{res['upsample_factor']})", flush=True)

    out = {"models": results, "charset_size": len(charset), "head": args.head,
           "n_train": len(rows_tr), "n_dev": len(rows_de)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ASR comparison (dev) ===", flush=True)
    print(f"{'model':<10}{'dev CER':>10}{'dev WER':>10}{'dim':>7}{'up':>5}", flush=True)
    for name, r in results.items():
        if "dev" in r:
            print(f"{name:<10}{r['dev']['cer']:>10.4f}{r['dev']['wer']:>10.4f}{r['feat_dim']:>7}{r['upsample_factor']:>5}", flush=True)
        else:
            print(f"{name:<10}{'(skipped)':>10}", flush=True)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
