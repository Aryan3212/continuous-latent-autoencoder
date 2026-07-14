"""Evaluate Mimi (Kyutai) encoder-decoder reconstruction quality.

Loads Mimi from HuggingFace, runs encode→decode on audio from a manifest,
computes the same MultiResSTFTLoss metrics as train.py (at 24 kHz), and
saves original / reconstructed WAV pairs for A/B listening.

Usage:
    uv run python scripts/eval_mimi_recon.py \
        --manifest staging/manifests/val.jsonl \
        --out_dir mimi_recon_out \
        --max_batches 50 \
        --num_recon_wavs 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F

from data_loading import AudioDataset, DatasetConfig, collate_fixed
from losses import MultiResSTFTLoss
from schema import STFTCfg


MIMI_SR = 24_000
MIMI_MODEL_ID = "kyutai/mimi"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="JSONL manifest (audio_filepath keys)")
    ap.add_argument("--out_dir", required=True, help="Directory for WAV pairs + metrics")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=6.0,
                    help="Segment length in seconds (longer = more padding for Mimi)")
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--num_recon_wavs", type=int, default=10,
                    help="Number of individual WAV pairs to save")
    ap.add_argument("--source_sr", type=int, default=16000,
                    help="Source sample rate of the manifest audio")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Mimi ---
    print(f"Loading Mimi from {MIMI_MODEL_ID} ...")
    t0 = time.perf_counter()
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(device).eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s  "
          f"(device={device}, dtype={next(mimi.parameters()).dtype})")

    # --- Dataset ---
    ds = AudioDataset(
        DatasetConfig(
            manifest=args.manifest,
            sample_rate=args.source_sr,
            segment_seconds=args.segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0,
        collate_fn=collate_fixed, drop_last=False,
    )

    # --- STFT loss at 24 kHz ---
    stft = MultiResSTFTLoss(STFTCfg()).to(device)

    resampler_src_to_mimi = None
    if args.source_sr != MIMI_SR:
        import torchaudio
        resampler_src_to_mimi = torchaudio.transforms.Resample(args.source_sr, MIMI_SR).to(device)

    # --- Accumulators ---
    sums = {"stft_loss": 0.0, "stft_sc": 0.0, "stft_mag": 0.0, "stft_log": 0.0, "wav_l1": 0.0}
    n_batches = 0
    wavs_saved = 0
    per_file_results: list[dict] = []

    print(f"Running Mimi encode→decode on {args.manifest} ({args.max_batches} max batches) ...")
    t0 = time.perf_counter()

    for i, batch in enumerate(dl):
        if n_batches >= args.max_batches:
            break
        wav_src = batch["wav"].to(device)  # (B, 1, T) @ source_sr

        # Resample to Mimi's 24 kHz
        wav_24k = resampler_src_to_mimi(wav_src) if resampler_src_to_mimi is not None else wav_src

        # Encode → decode
        enc = mimi.encode(wav_24k)
        x_hat = mimi.decode(enc.audio_codes).audio_values  # (B, 1, T')

        # Trim / pad to match input length (Mimi may slightly change length)
        T = wav_24k.size(-1)
        if x_hat.size(-1) > T:
            x_hat = x_hat[..., :T]
        elif x_hat.size(-1) < T:
            x_hat = F.pad(x_hat, (0, T - x_hat.size(-1)))

        # Metrics
        l_stft, stft_stats = stft(x_hat, wav_24k, return_per_sample=True)
        l_wav_ps = (x_hat - wav_24k).abs().mean(dim=(1, 2))

        for k in sums:
            if k == "wav_l1":
                sums[k] += float(l_wav_ps.mean().cpu())
            else:
                sums[k] += float(stft_stats[k].mean().cpu())
        n_batches += 1

        # Save WAV pairs (only until we hit the limit)
        if wavs_saved < args.num_recon_wavs:
            import soundfile as sf
            for b in range(wav_24k.size(0)):
                if wavs_saved >= args.num_recon_wavs:
                    break
                meta = batch["meta"][b]
                stem = pathlib.Path(meta.get("audio_filepath", f"utt_{wavs_saved}")).stem
                orig_path = out_dir / f"{stem}_orig.wav"
                recon_path = out_dir / f"{stem}_recon.wav"
                sf.write(str(orig_path), wav_24k[b, 0].cpu().numpy(), MIMI_SR)
                sf.write(str(recon_path), x_hat[b, 0].cpu().numpy(), MIMI_SR)

                file_stats = {
                    "file": meta.get("audio_filepath", stem),
                    "stft_loss": float(l_stft[b].cpu()),
                    "stft_sc": float(stft_stats["stft_sc"][b].cpu()),
                    "stft_mag": float(stft_stats["stft_mag"][b].cpu()),
                    "stft_log": float(stft_stats["stft_log"][b].cpu()),
                    "wav_l1": float(l_wav_ps[b].cpu()),
                }
                per_file_results.append(file_stats)
                print(f"  [{wavs_saved+1}] {stem}: stft={file_stats['stft_loss']:.4f}  "
                      f"wav_l1={file_stats['wav_l1']:.4f}")
                wavs_saved += 1

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  ... batch {i+1} ({n_batches} done, {elapsed:.1f}s)")

    # --- Aggregate ---
    agg = {k: v / max(1, n_batches) for k, v in sums.items()}
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_batches} batches, {elapsed:.1f}s ({n_batches*args.batch_size/max(1,elapsed):.1f} samples/s)")
    print(f"Aggregate metrics (at {MIMI_SR} Hz):")
    for k, v in agg.items():
        print(f"  {k:12s} = {v:.6f}")

    # Save results
    results_path = out_dir / "mimi_metrics.json"
    results_path.write_text(json.dumps({"aggregate": agg, "per_file": per_file_results}, indent=2))
    print(f"\nMetrics saved to {results_path}")
    print(f"WAV pairs saved in {out_dir}/")


if __name__ == "__main__":
    main()
