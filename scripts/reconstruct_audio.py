"""Reconstruct audio through the trained autoencoder: wav -> z -> wav_hat.

Loads the frozen frontend + encoder + decoder from a checkpoint, encodes each
input file in independent training-length windows (the encoder has unmasked
global attention and only ever saw segment-length inputs, so a single pass
over a longer waveform is out-of-distribution — same reasoning as
eval/common.iter_frame_features), decodes each window back to a waveform, and
writes an `<stem>_orig.wav` / `<stem>_recon.wav` pair per file for A/B
listening, plus per-file multi-res STFT and L1 numbers.

Usage:
    uv run python scripts/reconstruct_audio.py \
        --config configs/local_6gb.yaml \
        --ckpt runs/<run_id>/ckpt_step30000.pt \
        --out_dir recon_out \
        path/to/utt1.flac path/to/utt2.wav
"""
from __future__ import annotations

import argparse
import math
import pathlib
from typing import Tuple

import torch
import torch.nn.functional as F

from losses.multires_stft import MultiResSTFTLoss
from models.decoder_generator import WaveformDecoder
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from utils.config import load_config


def _load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """File -> (1, 1, S) float mono at sample_rate. Mirrors AudioDataset.__getitem__."""
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.flatten()
    if int(sr) != sample_rate:
        wav = torchaudio.transforms.Resample(int(sr), sample_rate)(wav)
    return wav.view(1, 1, -1)


@torch.no_grad()
def reconstruct(
    model: torch.nn.ModuleDict, wav: torch.Tensor, chunk_samples: int | None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """(1,1,S) -> ((1,1,S) reconstruction, (n_latent_frames, latent_dim)).

    When chunk_samples is set and the file is longer, the waveform is
    zero-padded to a whole number of windows, every window is encoded and
    decoded independently (batched as one forward pass), and the audio is
    concatenated. Each window decodes to exactly chunk_samples, so the seams
    line up sample-exactly; mild boundary artifacts at the seams are expected
    since no window sees its neighbours.
    """
    S = wav.size(-1)
    if chunk_samples is None or chunk_samples >= S:
        z = model["encoder"](model["frontend"](wav))           # (1, d, T')
        return model["decoder"](z, target_len=S), (z.size(-1), z.size(1))
    n_chunks = math.ceil(S / chunk_samples)
    padded = F.pad(wav, (0, n_chunks * chunk_samples - S))
    chunks = padded.view(n_chunks, 1, chunk_samples)
    z = model["encoder"](model["frontend"](chunks))            # (n, d, Tc)
    x_hat = model["decoder"](z, target_len=chunk_samples)      # (n, 1, cs)
    return x_hat.reshape(1, 1, -1)[..., :S], (n_chunks * z.size(-1), z.size(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk_seconds", type=float, default=None,
                    help="Encode/decode in independent windows of this length "
                         "(default: pretraining data.segment_seconds; <=0 disables)")
    ap.add_argument("inputs", nargs="+", help="Audio files to reconstruct")
    args = ap.parse_args()

    import soundfile as sf

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg.data.sample_rate

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    decoder = WaveformDecoder(cfg.model.encoder.d_model, cfg.model.decoder)
    model = torch.nn.ModuleDict(
        {"frontend": frontend, "encoder": encoder, "decoder": decoder}
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    filtered = {k: v for k, v in state["model"].items()
                if k.split(".", 1)[0] in {"frontend", "encoder", "decoder"}}
    model.load_state_dict(filtered, strict=True)
    model.eval()

    chunk = args.chunk_seconds if args.chunk_seconds is not None else cfg.data.segment_seconds
    chunk_samples = int(round(chunk * sr)) if chunk > 0 else None

    stft = MultiResSTFTLoss(cfg.loss.stft).to(device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        wav = _load_audio(path, sr).to(device)
        x_hat, (n_frames, latent_dim) = reconstruct(model, wav, chunk_samples)
        l_stft, _ = stft(x_hat, wav)
        l1 = (x_hat - wav).abs().mean()

        stem = pathlib.Path(path).stem
        orig_path = out_dir / f"{stem}_orig.wav"
        recon_path = out_dir / f"{stem}_recon.wav"
        sf.write(str(orig_path), wav[0, 0].cpu().numpy(), sr)
        sf.write(str(recon_path), x_hat[0, 0].cpu().numpy(), sr)

        dur = wav.size(-1) / sr
        print(f"{path}: {dur:.2f}s -> {n_frames} latent frames x {latent_dim} dims "
              f"({wav.size(-1) / max(1, n_frames):.0f} samples/frame)")
        print(f"  stft={float(l_stft):.4f}  wav_l1={float(l1):.4f}")
        print(f"  wrote {orig_path} and {recon_path}")


if __name__ == "__main__":
    main()
