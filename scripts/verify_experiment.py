import argparse
import torch
import soundfile as sf
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from models.decoder_generator import WaveformDecoder
from utils.config import load_config
import matplotlib.pyplot as plt
import numpy as np

def test_reconstruction(model, wav, out_path):
    with torch.no_grad():
        h0 = model["frontend"](wav)
        hE = model["encoder"](h0)
        z = hE
        rec = model["decoder"](z, target_len=wav.size(-1))
    
    # Save audio
    sf.write(out_path, rec[0, 0].cpu().numpy(), 16000)
    print(f"Saved reconstruction to {out_path}")
    
    # Plot spectrograms
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.specgram(wav[0, 0].cpu().numpy(), Fs=16000)
    plt.title("Original")
    plt.subplot(2, 1, 2)
    plt.specgram(rec[0, 0].cpu().numpy(), Fs=16000)
    plt.title("Reconstructed")
    plt.tight_layout()
    plt.savefig(out_path.replace(".wav", ".png"))
    print(f"Saved spectrogram to {out_path.replace('.wav', '.png')}")

def test_shift_invariance(model, wav, shift_amount=1):
    with torch.no_grad():
        # Encode original
        h0 = model["frontend"](wav)
        hE = model["encoder"](h0)
        z1 = hE
        
        # Shift input
        wav_shifted = torch.roll(wav, shifts=shift_amount, dims=-1)
        # Handle edge effects by zeroing rolled part (simple approx)
        wav_shifted[..., :shift_amount] = 0
        
        h0s = model["frontend"](wav_shifted)
        hEs = model["encoder"](h0s)
        z2 = hEs
        
        # Expected shift in latent space depends on stride
        total_stride = np.prod(model["frontend"].cfg.strides) # 1280
        # If shift < stride, latent might not shift or shift slightly? 
        # Actually shift invariance test usually checks if output shifts same amount.
        
        rec1 = model["decoder"](z1, target_len=wav.size(-1))
        rec2 = model["decoder"](z2, target_len=wav.size(-1))
        
        # Shift rec1 to match rec2
        rec1_shifted = torch.roll(rec1, shifts=shift_amount, dims=-1)
        
        diff = (rec1_shifted - rec2).abs().mean()
        print(f"Shift Invariance Diff (shift={shift_amount}): {diff.item():.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_wav", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    latent_dim = cfg.model.encoder.d_model
    decoder = WaveformDecoder(latent_dim, cfg.model.decoder)
    
    model = torch.nn.ModuleDict({
        "frontend": frontend,
        "encoder": encoder,
        "decoder": decoder
    }).to(device)
    
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    wav, sr = sf.read(args.input_wav)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # Resample if needed? Assuming 16k input for now.
    
    print("Running Reconstruction Test...")
    test_reconstruction(model, wav, "verify_recon.wav")
    
    print("Running Shift Invariance Test...")
    test_shift_invariance(model, wav, shift_amount=1280) # 1 token shift

if __name__ == "__main__":
    main()
