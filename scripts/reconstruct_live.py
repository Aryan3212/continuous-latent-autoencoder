"""Hear the autoencoder: record from the mic (or pass files) -> reconstruct -> play.

Two modes:
  * Mic mode (no file args): press Enter to start recording, Enter again to
    stop, then the recording is reconstructed and played back. Loops so you can
    keep testing without reloading the model. Ctrl-C to quit.
  * File mode (one or more audio paths): each file (mp3/flac/wav/...) is
    reconstructed and played; pass --out_dir to also write _orig/_recon wavs.

Audio is processed in independent training-length windows (data.segment_seconds,
3.0s for this model) exactly like scripts/reconstruct_audio.py: the encoder only
ever saw segment-length inputs, so each window is encoded/decoded on its own and
the pieces are concatenated (mild seam artifacts are expected).

The model is ~3M params, so CPU fp32 on an M1 is instant — no GPU, no
quantization. Setup on macOS:
    brew install portaudio              # PortAudio backend for sounddevice
    uv pip install sounddevice soundfile

Usage (run from the repo root):
    # local checkpoint, mic mode
    uv run python scripts/reconstruct_live.py \
        --config configs/kaggle_3m_gan.yaml --ckpt last.pt
    # fetch last.pt straight from the Hub, mic mode
    uv run python scripts/reconstruct_live.py \
        --config configs/kaggle_3m_gan.yaml --hf_repo aryan3212/clae-bengali-encoder
    # reconstruct files instead of recording
    uv run python scripts/reconstruct_live.py \
        --config configs/kaggle_3m_gan.yaml --ckpt last.pt clip.mp3 other.flac
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

# scripts/ is excluded from packaging (see pyproject.toml), so put the repo root
# on sys.path for the root-module imports — same idiom as housekeeping.py.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config  # noqa: E402
from reconstruct_audio import _load_audio, load_model, reconstruct  # noqa: E402


def record_until_enter(target_sr: int) -> np.ndarray:
    """Record mono from the default input device until Enter, return float32 @ target_sr.

    The built-in Mac mic usually runs at 44.1/48 kHz, so capture at the device's
    native rate and resample down to the model's 16 kHz.
    """
    import sounddevice as sd

    dev_sr = int(sd.query_devices(kind="input")["default_samplerate"]) or target_sr
    frames: list[np.ndarray] = []

    def _cb(indata, _frames, _time, status):  # called on the audio thread
        if status:
            print(f"  (audio status: {status})", file=sys.stderr)
        frames.append(indata.copy())

    print(f"● recording at {dev_sr} Hz — press Enter to stop ...", flush=True)
    with sd.InputStream(samplerate=dev_sr, channels=1, dtype="float32", callback=_cb):
        input()
    if not frames:
        return np.zeros(0, dtype=np.float32)

    wav = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)
    if dev_sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), dev_sr, target_sr
        ).numpy()
    return wav


def play(wav: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(np.clip(wav, -1.0, 1.0), sample_rate)
    sd.wait()


@torch.no_grad()
def reconstruct_array(
    model: torch.nn.ModuleDict, wav_np: np.ndarray, chunk_samples: int | None, device
):
    """(S,) float32 waveform -> ((S,) reconstruction, (n_frames, latent_dim))."""
    wav = torch.from_numpy(np.ascontiguousarray(wav_np, dtype=np.float32)).view(1, 1, -1).to(device)
    # Pad clips shorter than one window up to a full window so the conv stack
    # always has at least one latent frame to work with.
    if chunk_samples and wav.size(-1) < chunk_samples:
        wav = F.pad(wav, (0, chunk_samples - wav.size(-1)))
    x_hat, meta = reconstruct(model, wav, chunk_samples)
    return x_hat[0, 0].cpu().numpy(), meta


def _save_pair(out_dir: str, stem: str, orig: np.ndarray, recon: np.ndarray, sr: int) -> None:
    import soundfile as sf

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / f"{stem}_orig.wav"), orig, sr)
    sf.write(str(out / f"{stem}_recon.wav"), recon, sr)
    print(f"  wrote {out / (stem + '_orig.wav')} and {stem}_recon.wav")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="architecture config (e.g. configs/kaggle_3m_gan.yaml)")
    ap.add_argument("--ckpt", default=None, help="path to last.pt (or use --hf_repo)")
    ap.add_argument("--hf_repo", default=None, help="download the checkpoint from this HF repo if --ckpt is omitted")
    ap.add_argument("--hf_file", default="last.pt", help="checkpoint filename in the HF repo (default last.pt)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="default: cuda if available, else cpu")
    ap.add_argument("--chunk_seconds", type=float, default=None,
                    help="window length (default: data.segment_seconds; <=0 = whole file in one pass)")
    ap.add_argument("--out_dir", default=None, help="also write _orig/_recon wavs here")
    ap.add_argument("--no_play", action="store_true", help="reconstruct only, don't play audio")
    ap.add_argument("inputs", nargs="*", help="audio files; if omitted, record from the mic")
    args = ap.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        if args.hf_repo is None:
            ap.error("pass --ckpt PATH or --hf_repo REPO")
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)
        print(f"downloaded {args.hf_repo}/{args.hf_file} -> {ckpt}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    sr = cfg.data.sample_rate
    model = load_model(cfg, ckpt, device)
    chunk = args.chunk_seconds if args.chunk_seconds is not None else cfg.data.segment_seconds
    chunk_samples = int(round(chunk * sr)) if chunk > 0 else None
    window = f"{chunk:g}s" if chunk_samples else "whole-file"
    print(f"model on {device} | sr={sr} | window={window}")

    if args.inputs:
        for path in args.inputs:
            wav = _load_audio(path, sr).to(device)
            x_hat, (n_frames, latent_dim) = reconstruct(model, wav, chunk_samples)
            recon = x_hat[0, 0].cpu().numpy()
            print(f"{path}: {wav.size(-1) / sr:.2f}s -> {n_frames} latent frames x {latent_dim} dims")
            if args.out_dir:
                _save_pair(args.out_dir, pathlib.Path(path).stem, wav[0, 0].cpu().numpy(), recon, sr)
            if not args.no_play:
                print("  playing reconstruction ...")
                play(recon, sr)
        return

    # Mic mode: record -> reconstruct -> play, on a loop.
    print("mic mode — Enter to start recording, Enter again to stop. Ctrl-C to quit.")
    take = 0
    while True:
        try:
            input("\npress Enter to record ... ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        wav_np = record_until_enter(sr)
        if wav_np.size == 0:
            print("  (nothing recorded)")
            continue
        recon, (n_frames, latent_dim) = reconstruct_array(model, wav_np, chunk_samples, device)
        print(f"  recorded {wav_np.size / sr:.2f}s -> {n_frames} latent frames x {latent_dim} dims")
        if args.out_dir:
            _save_pair(args.out_dir, f"take{take:02d}", wav_np, recon, sr)
        if not args.no_play:
            print("  playing reconstruction ...")
            play(recon, sr)
        take += 1


if __name__ == "__main__":
    main()
