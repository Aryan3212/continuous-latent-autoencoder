from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import torch

from data_loading import AudioDataset, DatasetConfig, collate_fixed
from eval.common import amp_enabled
from losses import MultiResSTFTLoss
from models.decoder_generator import WaveformDecoder
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from config import apply_overrides, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=None)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_batches < 1:
        ap.error("--batch_size and --max_batches must be positive")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = args.segment_seconds if args.segment_seconds is not None else cfg.data.segment_seconds
    if seg <= 0:
        ap.error("--segment_seconds must be positive")

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    latent_dim = cfg.model.encoder.d_model
    decoder = WaveformDecoder(latent_dim, cfg.model.decoder)

    model = torch.nn.ModuleDict(
        {"frontend": frontend, "encoder": encoder, "decoder": decoder}
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    filtered = {k: v for k, v in state["model"].items() if k.split(".", 1)[0] in {"frontend", "encoder", "decoder"}}
    model.load_state_dict(filtered, strict=True)
    model.eval()

    stft = MultiResSTFTLoss(cfg.loss.stft).to(device)

    ds = AudioDataset(
        DatasetConfig(
            manifest=args.manifest,
            sample_rate=cfg.data.sample_rate,
            segment_seconds=seg,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0,
        collate_fn=collate_fixed, drop_last=False,
    )

    sums: Dict[str, float] = {"stft": 0.0, "wav_l1": 0.0}
    num_batches = 0
    num_samples = 0
    use_amp = amp_enabled(device)
    with torch.no_grad():
        for batch in dl:
            wav = batch["wav"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                h0 = model["frontend"](wav)
                hE = model["encoder"](h0)
                x_hat = model["decoder"](hE, target_len=wav.size(-1))
            l_stft, _ = stft(x_hat, wav)
            l_wav = (x_hat - wav).abs().mean()
            batch_samples = int(wav.size(0))
            sums["stft"] += float(l_stft.detach().cpu()) * batch_samples
            sums["wav_l1"] += float(l_wav.detach().cpu()) * batch_samples
            num_batches += 1
            num_samples += batch_samples
            if num_batches >= args.max_batches:
                break

    if num_samples == 0:
        raise RuntimeError(f"Reconstruction manifest contains no samples: {args.manifest}")
    out = {k: v / num_samples for k, v in sums.items()}
    out.update({"num_samples": num_samples, "num_batches": num_batches})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
