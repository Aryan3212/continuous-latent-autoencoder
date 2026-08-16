from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import torch

from config import apply_overrides, load_config
from data_loading import AudioDataset, DatasetConfig, collate_fixed
from eval.recon_metrics import (
    ReconstructionMetrics,
    EVALUATION_STFT_CONFIG_VERSION,
    evaluation_stft_config,
    strict_coverage_failures,
    valid_num_samples,
)
from losses import MultiResSTFTLoss
from models.decoder_generator import WaveformDecoder
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend


def _item_id(index: int, meta: dict[str, Any]) -> str:
    source = str(meta.get("sample_id") or meta.get("audio_filepath", ""))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"item_{index:06d}_{digest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--audio_dir",
        default=None,
        help="Optional directory for fixed-source original/reconstruction pairs.",
    )
    ap.add_argument("--audio_limit", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_batches < 1 or args.audio_limit < 0:
        ap.error("batch size/budget must be positive and --audio_limit non-negative")
    if args.audio_limit and not args.audio_dir:
        ap.error("--audio_dir is required when --audio_limit is positive")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = args.segment_seconds if args.segment_seconds is not None else cfg.data.segment_seconds
    if seg <= 0:
        ap.error("--segment_seconds must be positive")

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    decoder = WaveformDecoder(cfg.model.encoder.d_model, cfg.model.decoder)
    model = torch.nn.ModuleDict(
        {"frontend": frontend, "encoder": encoder, "decoder": decoder}
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    filtered = {
        key: value
        for key, value in state["model"].items()
        if key.split(".", 1)[0] in {"frontend", "encoder", "decoder"}
    }
    model.load_state_dict(filtered, strict=True)
    model.eval()

    stft = MultiResSTFTLoss(evaluation_stft_config()).to(device)
    metrics = ReconstructionMetrics(stft, cfg.data.sample_rate)
    ds = AudioDataset(
        DatasetConfig(
            manifest=args.manifest,
            sample_rate=cfg.data.sample_rate,
            segment_seconds=seg,
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

    audio_dir = pathlib.Path(args.audio_dir) if args.audio_dir else None
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)
    audio_index: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []
    num_batches = 0
    num_items = 0
    with torch.no_grad():
        for batch in dl:
            wav = batch["wav"].to(device)
            # Decoder activations can overflow under BF16 for otherwise finite
            # checkpoints and clean held-out inputs. Reconstruction metrics are
            # an offline correctness path, so keep this forward pass in FP32.
            encoded = model["encoder"](model["frontend"](wav))
            reconstructed = model["decoder"](encoded, target_len=wav.size(-1))
            for batch_index, meta in enumerate(batch["meta"]):
                item_index = num_items
                item_id = _item_id(item_index, meta)
                valid_length = valid_num_samples(
                    meta, cfg.data.sample_rate, int(wav.size(-1))
                )
                reference = wav[batch_index, 0, :valid_length].float()
                estimate = reconstructed[batch_index, 0, :valid_length].float()
                item_metrics = metrics.evaluate(reference, estimate)
                per_item.append(
                    {
                        "index": item_index,
                        "item_id": item_id,
                        "audio_filepath": meta.get("audio_filepath"),
                        "sample_id": meta.get("sample_id"),
                        "duration_seconds_scored": valid_length / cfg.data.sample_rate,
                        "metrics": item_metrics,
                    }
                )

                if audio_dir is not None and len(audio_index) < args.audio_limit:
                    import soundfile as sf

                    original_path = audio_dir / f"{item_id}.original.wav"
                    recon_path = audio_dir / f"{item_id}.reconstruction.wav"
                    sf.write(
                        str(original_path), reference.detach().cpu().numpy(), cfg.data.sample_rate
                    )
                    sf.write(
                        str(recon_path), estimate.detach().cpu().numpy(), cfg.data.sample_rate
                    )
                    audio_index.append(
                        {
                            "index": item_index,
                            "item_id": item_id,
                            "audio_filepath": meta.get("audio_filepath"),
                            "sample_id": meta.get("sample_id"),
                            "original": original_path.name,
                            "reconstruction": recon_path.name,
                            "sample_rate": cfg.data.sample_rate,
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
    payload: dict[str, Any] = {
        **aggregate,
        "aggregate": aggregate,
        "coverage": metric_summary["coverage"],
        "per_item": per_item,
        "num_samples": num_items,
        "num_batches": num_batches,
        "sample_rate": cfg.data.sample_rate,
        "segment_seconds": seg,
        "manifest": str(pathlib.Path(args.manifest).resolve()),
        "checkpoint": str(pathlib.Path(args.ckpt).resolve()),
        "fixed_source_order": True,
        "evaluation_stft_config": evaluation_stft_config().model_dump(),
        "evaluation_stft_config_version": EVALUATION_STFT_CONFIG_VERSION,
        "strict_complete": not incomplete_metrics,
        "incomplete_metrics": incomplete_metrics,
    }
    if audio_dir is not None:
        index_path = audio_dir / "index.json"
        index_path.write_text(json.dumps(audio_index, indent=2) + "\n", encoding="utf-8")
        payload["audio_pairs"] = {
            "directory": str(audio_dir.resolve()),
            "index": str(index_path.resolve()),
            "count": len(audio_index),
        }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if incomplete_metrics:
        raise RuntimeError(
            "Reconstruction produced incomplete metric coverage for: "
            f"{', '.join(incomplete_metrics)}. See coverage in the output JSON."
        )


if __name__ == "__main__":
    main()
