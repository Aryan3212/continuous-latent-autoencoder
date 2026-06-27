import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from sklearn.decomposition import PCA
from data_loading import AudioDataset, DatasetConfig, collate_fixed
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="Output image path (e.g. latents.png)")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to visualize")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load Config & Model
    cfg = load_config(args.config)
    device = torch.device(args.device)

    frontend = ConvFrontend(cfg.model.frontend).to(device)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder).to(device)

    # Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    def load_part(model, key):
        part_dict = {k[len(key) + 1:]: v for k, v in ckpt["model"].items() if k.startswith(f"{key}.")}
        model.load_state_dict(part_dict, strict=True)

    load_part(frontend, "frontend")
    load_part(encoder, "encoder")

    frontend.eval()
    encoder.eval()

    # Load Data
    ds = AudioDataset(DatasetConfig(manifest=args.manifest, sample_rate=16000, random_crop=False))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fixed, drop_last=False)

    latents = []
    
    print("Extracting latents...")
    count = 0
    with torch.no_grad():
        for batch in dl:
            wav = batch["wav"].to(device)
            h0 = frontend(wav)
            hE = encoder(h0)
            z = hE # (B, D, T)
            
            # Pool over time for visualization (Mean pooling)
            z_pooled = z.mean(dim=-1).cpu().numpy() # (B, D)
            latents.append(z_pooled)
            
            count += len(wav)
            if count >= args.limit:
                break
    
    latents = np.concatenate(latents, axis=0)[:args.limit]
    print(f"Collected {latents.shape} embeddings.")

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    var_explained = pca.explained_variance_ratio_

    # UMAP (Optional)
    z_umap = None
    try:
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP()
        z_umap = reducer.fit_transform(latents)
    except ImportError:
        print("UMAP not installed. Skipping UMAP visualization.")
    except Exception as e:
        print(f"UMAP failed: {e}. Skipping.")

    # Plot
    if z_umap is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6, s=10)
        ax[0].set_title(f"PCA (Var: {var_explained[0]:.2f}, {var_explained[1]:.2f})")
        ax[1].scatter(z_umap[:, 0], z_umap[:, 1], alpha=0.6, s=10, c='orange')
        ax[1].set_title("UMAP")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6, s=10)
        ax.set_title(f"PCA (Var: {var_explained[0]:.2f}, {var_explained[1]:.2f})")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    main()
