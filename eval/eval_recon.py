from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

import torch

from data.dataset import AudioManifestDataset, ManifestConfig, collate_fixed
from losses.multires_stft import MultiResSTFTConfig, MultiResSTFTLoss
from models.decoder_generator import DecoderConfig, WaveformDecoder
from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
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
    seg = float(args.segment_seconds if args.segment_seconds is not None else cfg["data"]["segment_seconds"])

    mcfg = cfg["model"]
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    
    latent_dim = int(mcfg["encoder"]["d_model"])
    
    decoder_cfg = DecoderConfig(**mcfg["decoder"])
    decoder = WaveformDecoder(latent_dim, decoder_cfg)
    if decoder_cfg.latent_stats_path:
        stats = torch.load(decoder_cfg.latent_stats_path, map_location="cpu")
        decoder.set_latent_stats(stats["mean"], stats["var"])

    model = torch.nn.ModuleDict(
        {"frontend": frontend, "encoder": encoder, "decoder": decoder}
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    loss_cfg = MultiResSTFTConfig(**cfg["loss"]["stft"])
    stft = MultiResSTFTLoss(loss_cfg).to(device)

    ds = AudioManifestDataset(
        ManifestConfig(
            manifest_path=args.manifest,
            sample_rate=int(cfg["data"]["sample_rate"]),
            segment_seconds=seg,
            random_crop=False,
        )
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fixed)

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
