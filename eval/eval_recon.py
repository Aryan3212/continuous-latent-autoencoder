from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import torch

from data.dataset import AudioDataset, DatasetConfig, collate_fixed
from losses.multires_stft import MultiResSTFTLoss
from models.decoder_generator import WaveformDecoder
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from utils.config import apply_overrides, load_config


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

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = args.segment_seconds if args.segment_seconds is not None else cfg.data.segment_seconds

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    latent_dim = cfg.model.encoder.d_model
    decoder = WaveformDecoder(latent_dim, cfg.model.decoder)

    model = torch.nn.ModuleDict(
        {"frontend": frontend, "encoder": encoder, "decoder": decoder}
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
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
    n = 0
    with torch.no_grad():
        for batch in dl:
            wav = batch["wav"].to(device)
            h0 = model["frontend"](wav)
            hE = model["encoder"](h0)
            z = hE
            x_hat = model["decoder"](z, target_len=wav.size(-1))
            l_stft, _ = stft(x_hat, wav)
            l_wav = (x_hat - wav).abs().mean()
            sums["stft"] += float(l_stft.detach().cpu())
            sums["wav_l1"] += float(l_wav.detach().cpu())
            n += 1
            if n >= args.max_batches:
                break

    out = {k: v / max(1, n) for k, v in sums.items()}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
