import argparse
import random
import torch
import soundfile as sf
import json
import logging
from pathlib import Path

# Import model components
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.encoder import Encoder, EncoderConfig
from models.decoder_generator import WaveformDecoder, DecoderConfig
from data.dataset import _load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(ckpt_path, device):
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["cfg"]
    mcfg = cfg["model"]

    # Rebuild model structure
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    
    latent_dim = int(mcfg["encoder"]["d_model"])
    decoder = WaveformDecoder(latent_dim, DecoderConfig(**mcfg["decoder"]))

    model = torch.nn.ModuleDict({
        "frontend": frontend,
        "encoder": encoder,
        "decoder": decoder
    })

    # Load weights
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model, cfg

def reconstruct(model, wav, device):
    with torch.no_grad():
        wav = wav.to(device).unsqueeze(0).unsqueeze(0) # (1, 1, T)
        
        # Encode
        h0 = model["frontend"](wav)
        hE = model["encoder"](h0)
        z = hE
        
        # Decode
        # Target length is roughly original length, but let decoder decide or pad/trim
        # Decoder produces (1, 1, T_out)
        out = model["decoder"](z, target_len=wav.size(-1))
        
        return out.squeeze().cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--manifest", default="data/manifests/experiment_v1/train.jsonl")
    ap.add_argument("--out_dir", default="reconstructions")
    ap.add_argument("--num_samples", type=int, default=10, help="Number of samples to reconstruct")
    ap.add_argument("--sample_id", type=int, default=None, help="Index of sample to pick (if set, num_samples is ignored)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    
    # Load manifest
    lines = Path(args.manifest).read_text().splitlines()
    
    if args.sample_id is not None:
        selected_lines = [lines[args.sample_id]]
    else:
        # Pick N random samples
        selected_lines = random.sample(lines, min(args.num_samples, len(lines)))
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for i, line in enumerate(selected_lines):
        item = json.loads(line)
        path = item["audio_filepath"]
        logger.info(f"Processing sample {i}: {path}")

        # Load audio (full duration or fixed segment)
        # Let's load up to 5 seconds to hear enough context
        wav = _load_audio(path, int(cfg["data"]["sample_rate"]), start_sec=0.0, duration_sec=5.0)
        
        recon_wav = reconstruct(model, wav, device)

        # Save
        sf.write(out_dir / f"sample_{i}_orig.wav", wav.numpy(), int(cfg["data"]["sample_rate"]))
        sf.write(out_dir / f"sample_{i}_recon.wav", recon_wav.numpy(), int(cfg["data"]["sample_rate"]))
        
        logger.info(f"Saved sample {i} to {out_dir}")

    logger.info(f"Finished processing {len(selected_lines)} samples.")

if __name__ == "__main__":
    main()
