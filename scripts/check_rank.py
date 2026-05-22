import torch
import argparse
import sys
import os

# Add current directory to path so we can import models
sys.path.append(os.getcwd())

from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from data.dataset import AudioDataset, DatasetConfig, collate_fixed
from utils.config import load_config
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_batches", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    frontend = ConvFrontend(FrontendConfig(**cfg.model.frontend.model_dump()))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**cfg.model.encoder.model_dump()))
    
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Handle both full state dict and partial
    sd = ckpt["model"] if "model" in ckpt else ckpt
    
    # Strip prefixes if they exist
    frontend_sd = {k.replace("frontend.", ""): v for k, v in sd.items() if k.startswith("frontend.")}
    encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    
    frontend.load_state_dict(frontend_sd if frontend_sd else {k: v for k, v in sd.items() if "frontend" in k})
    encoder.load_state_dict(encoder_sd if encoder_sd else {k: v for k, v in sd.items() if "encoder" in k})
    
    frontend.to(device).eval()
    encoder.to(device).eval()

    # Data
    dcfg = cfg.data
    ds = AudioDataset(DatasetConfig(
        manifest=dcfg.val_manifest,
        sample_rate=dcfg.sample_rate,
        segment_seconds=dcfg.segment_seconds,
        random_crop=False,
    ))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch_size,
        collate_fn=collate_fixed, drop_last=False,
    )

    all_z = []
    print(f"Collecting latents from {args.num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.num_batches: break
            wav = batch["wav"].to(device)
            h0 = frontend(wav)
            z = encoder(h0) # (B, D, T)
            # Flatten B and T to get a bunch of D-dim vectors
            z = z.permute(0, 2, 1).reshape(-1, z.size(1))
            all_z.append(z.cpu())
    
    if not all_z:
        print("No data collected. Check your val_manifest path.")
        return

    X = torch.cat(all_z, dim=0) # (N, D)
    N, D = X.shape
    print(f"Analyzing {N} vectors of dimension {D}")

    # 1. Basic Stats
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    print(f"\n--- Basic Stats ---")
    print(f"Mean: min={mean.min():.4f}, max={mean.max():.4f}, avg={mean.mean():.4f}")
    print(f"Std:  min={std.min():.4f}, max={std.max():.4f}, avg={std.mean():.4f}")

    # 2. Covariance and Eigenvalues
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (N - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.sort(descending=True)[0]
    
    # 3. Effective Rank (Participation Ratio)
    # PR = (sum lambda)^2 / sum(lambda^2)
    pr = (eigvals.sum()**2) / (eigvals.pow(2).sum())
    
    print(f"\n--- Rank Analysis ---")
    print(f"Effective Rank (Participation Ratio): {pr.item():.2f} / {D}")
    
    # 4. Energy Distribution
    cum_energy = torch.cumsum(eigvals, dim=0) / eigvals.sum()
    k_90 = (cum_energy < 0.9).sum().item() + 1
    k_99 = (cum_energy < 0.99).sum().item() + 1
    print(f"90% of variance in top {k_90} dimensions")
    print(f"99% of variance in top {k_99} dimensions")

    # 5. Dead Dimensions
    dead = (std < 1e-4).sum().item()
    print(f"Dead dimensions (std < 1e-4): {dead}")

    # Log spectrum check
    top_10 = eigvals[:10].tolist()
    print(f"\nTop 10 Eigenvalues: {[f'{v:.4f}' for v in top_10]}")
    
    # Condition Number
    cond = eigvals.max() / eigvals.min().clamp(min=1e-10)
    print(f"Condition Number (max/min eig): {cond.item():.2e}")

if __name__ == "__main__":
    main()
