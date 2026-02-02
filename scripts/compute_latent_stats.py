import argparse
import pathlib

import torch

from data.dataset import AudioManifestDataset, ManifestConfig, collate_fixed
from models.encoder import Bottleneck, Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from utils.config import apply_overrides, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = torch.device(args.device)

    mcfg = cfg["model"]
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    bottleneck = Bottleneck(
        in_dim=mcfg["encoder"]["d_model"],
        latent_dim=int(mcfg["bottleneck"]["latent_dim"]),
        norm=str(mcfg["bottleneck"]["norm"]),
    )
    model = torch.nn.ModuleDict({"frontend": frontend, "encoder": encoder, "bottleneck": bottleneck}).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    dcfg = cfg["data"]
    if dcfg.get("train_manifest") is None:
        raise ValueError("Set data.train_manifest=/path/train.jsonl")
    ds = AudioManifestDataset(
        ManifestConfig(
            manifest_path=dcfg["train_manifest"],
            sample_rate=int(dcfg["sample_rate"]),
            segment_seconds=float(dcfg["segment_seconds"]),
        )
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fixed,
    )

    total = 0
    sum_z = None
    sumsq_z = None
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            wav = batch["wav"].to(device)
            h0 = model["frontend"](wav)
            hE = model["encoder"](h0)
            z = model["bottleneck"](hE)
            z = z.permute(0, 2, 1).reshape(-1, z.shape[1])
            z = z.to(torch.float64)
            if sum_z is None:
                sum_z = z.sum(dim=0)
                sumsq_z = (z**2).sum(dim=0)
            else:
                sum_z += z.sum(dim=0)
                sumsq_z += (z**2).sum(dim=0)
            total += z.shape[0]

    mean = sum_z / max(1, total)
    var = sumsq_z / max(1, total) - mean**2
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean.float(), "var": var.float(), "count": total}, str(out_path))
    print(f"saved stats to {out_path} (count={total})")


if __name__ == "__main__":
    main()
