"""Evaluate pinned Mimi reconstruction at exactly 8 quantizers (1.1 kbps)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import time
from typing import Any

import torch

from data_loading import AudioDataset, DatasetConfig, collate_fixed
from eval.recon_metrics import (
    ReconstructionMetrics,
    EVALUATION_STFT_CONFIG_VERSION,
    align_length,
    evaluation_stft_config,
    strict_coverage_failures,
    valid_num_samples,
)
from losses import MultiResSTFTLoss


MIMI_SAMPLE_RATE = 24_000
MIMI_MODEL_ID = "kyutai/mimi"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
MIMI_NUM_QUANTIZERS = 8
MIMI_NOMINAL_BITRATE_KBPS = 1.1


def _item_id(index: int, meta: dict[str, Any]) -> str:
    source = str(meta.get("sample_id") or meta.get("audio_filepath", ""))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"item_{index:06d}_{digest}"


def _encode_exactly_8(mimi: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    try:
        encoded = mimi.encode(waveform, num_quantizers=MIMI_NUM_QUANTIZERS)
    except TypeError as exc:
        raise RuntimeError(
            "Installed transformers Mimi API cannot request num_quantizers=8; "
            "refusing to run an uncontrolled bitrate comparison."
        ) from exc
    codes = getattr(encoded, "audio_codes", None)
    if not isinstance(codes, torch.Tensor) or codes.ndim < 3:
        raise RuntimeError("Mimi encode did not return a (batch, quantizer, frame) code tensor")
    if int(codes.size(1)) != MIMI_NUM_QUANTIZERS:
        raise RuntimeError(
            f"Mimi returned {codes.size(1)} quantizers after requesting 8; "
            "refusing to label this result as 1.1 kbps."
        )
    return codes


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=6.0)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--num_recon_wavs", type=int, default=20)
    ap.add_argument("--source_sr", type=int, default=16_000)
    args = ap.parse_args()
    if min(args.batch_size, args.max_batches, args.source_sr) < 1:
        ap.error("batch size, batch budget, and source sample rate must be positive")
    if args.segment_seconds <= 0 or args.num_recon_wavs < 0:
        ap.error("segment length must be positive and WAV count non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(args.out_dir)
    audio_dir = out_dir / "audio_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    (out_dir / "mimi_metrics.json").unlink(missing_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    from transformers import MimiModel

    print(f"Loading Mimi from {MIMI_MODEL_ID}@{MIMI_REVISION} ...", flush=True)
    started = time.perf_counter()
    mimi = MimiModel.from_pretrained(
        MIMI_MODEL_ID,
        revision=MIMI_REVISION,
    ).to(device).eval()
    print(f"Loaded pinned Mimi in {time.perf_counter() - started:.1f}s.", flush=True)

    ds = AudioDataset(
        DatasetConfig(
            manifest=args.manifest,
            sample_rate=args.source_sr,
            segment_seconds=args.segment_seconds,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fixed,
        drop_last=False,
    )
    stft = MultiResSTFTLoss(evaluation_stft_config()).to(device)
    metrics = ReconstructionMetrics(stft, args.source_sr)

    import torchaudio

    to_mimi = (
        torchaudio.transforms.Resample(args.source_sr, MIMI_SAMPLE_RATE).to(device)
        if args.source_sr != MIMI_SAMPLE_RATE
        else None
    )
    from_mimi = (
        torchaudio.transforms.Resample(MIMI_SAMPLE_RATE, args.source_sr).to(device)
        if args.source_sr != MIMI_SAMPLE_RATE
        else None
    )

    per_item: list[dict[str, Any]] = []
    audio_index: list[dict[str, Any]] = []
    num_batches = 0
    num_items = 0
    for batch in dl:
        source = batch["wav"].to(device)
        mimi_input = to_mimi(source) if to_mimi is not None else source
        codes = _encode_exactly_8(mimi, mimi_input)
        decoded = mimi.decode(codes).audio_values
        if decoded.ndim == 2:
            decoded = decoded.unsqueeze(1)
        if decoded.ndim != 3 or decoded.size(1) != 1:
            raise RuntimeError(
                f"Mimi decode must return mono (batch, channel, sample) audio, got {tuple(decoded.shape)}"
            )
        decoded = align_length(decoded, int(mimi_input.size(-1)))
        reconstructed = from_mimi(decoded) if from_mimi is not None else decoded
        reconstructed = align_length(reconstructed, int(source.size(-1)))

        for batch_index, meta in enumerate(batch["meta"]):
            item_index = num_items
            item_id = _item_id(item_index, meta)
            valid_length = valid_num_samples(meta, args.source_sr, int(source.size(-1)))
            reference = source[batch_index, 0, :valid_length].float()
            estimate = reconstructed[batch_index, 0, :valid_length].float()
            item_metrics = metrics.evaluate(reference, estimate)
            per_item.append(
                {
                    "index": item_index,
                    "item_id": item_id,
                    "audio_filepath": meta.get("audio_filepath"),
                    "sample_id": meta.get("sample_id"),
                    "duration_seconds_scored": valid_length / args.source_sr,
                    "metrics": item_metrics,
                }
            )
            if len(audio_index) < args.num_recon_wavs:
                import soundfile as sf

                original_path = audio_dir / f"{item_id}.original.wav"
                recon_path = audio_dir / f"{item_id}.reconstruction.wav"
                sf.write(str(original_path), reference.cpu().numpy(), args.source_sr)
                sf.write(str(recon_path), estimate.cpu().numpy(), args.source_sr)
                audio_index.append(
                    {
                        "index": item_index,
                        "item_id": item_id,
                        "audio_filepath": meta.get("audio_filepath"),
                        "sample_id": meta.get("sample_id"),
                        "original": original_path.name,
                        "reconstruction": recon_path.name,
                        "sample_rate": args.source_sr,
                        "num_samples": valid_length,
                    }
                )
            num_items += 1
        num_batches += 1
        if num_batches >= args.max_batches:
            break

    if num_items == 0:
        raise RuntimeError(f"Reconstruction manifest contains no samples: {args.manifest}")
    metric_summary = metrics.summary()
    aggregate = metric_summary["aggregate"]
    incomplete_metrics = strict_coverage_failures(metric_summary["coverage"], num_items)
    payload = {
        **aggregate,
        "aggregate": aggregate,
        "coverage": metric_summary["coverage"],
        "per_item": per_item,
        "num_samples": num_items,
        "num_batches": num_batches,
        "sample_rate": args.source_sr,
        "manifest": str(pathlib.Path(args.manifest).resolve()),
        "fixed_source_order": True,
        "evaluation_stft_config": evaluation_stft_config().model_dump(),
        "evaluation_stft_config_version": EVALUATION_STFT_CONFIG_VERSION,
        "strict_complete": not incomplete_metrics,
        "incomplete_metrics": incomplete_metrics,
        "model": {
            "repo": MIMI_MODEL_ID,
            "revision": MIMI_REVISION,
            "num_quantizers": MIMI_NUM_QUANTIZERS,
            "nominal_bitrate_kbps": MIMI_NOMINAL_BITRATE_KBPS,
            "native_sample_rate": MIMI_SAMPLE_RATE,
        },
        "audio_pairs": {
            "directory": str(audio_dir.resolve()),
            "index": str((audio_dir / "index.json").resolve()),
            "count": len(audio_index),
        },
    }
    (audio_dir / "index.json").write_text(
        json.dumps(audio_index, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "mimi_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'mimi_metrics.json'}", flush=True)
    if incomplete_metrics:
        raise RuntimeError(
            "Mimi reconstruction produced incomplete metric coverage for: "
            f"{', '.join(incomplete_metrics)}. See coverage in mimi_metrics.json."
        )


if __name__ == "__main__":
    main()
