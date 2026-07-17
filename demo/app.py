"""Gradio demo for the CLAE Bengali speech autoencoder — Hugging Face Spaces.

Web version of scripts/reconstruct_live.py: upload or record audio, the model
encodes -> continuous latent -> decodes, and you hear the reconstruction.

Audio is processed in independent 3s windows (data.segment_seconds), exactly
like reconstruct_live.py, because the encoder only ever saw 3s segments and uses
global attention — a single pass over a longer clip is out-of-distribution. Mild
seam artifacts at window boundaries are expected.

The model is ~2.5M params, so CPU fp32 is fine (no GPU needed on the Space).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

# --- config -----------------------------------------------------------------
REPO_URL    = "https://github.com/Aryan3212/continuous-latent-autoencoder.git"
REPO_BRANCH = "simplification"
REPO_DIR    = "/home/user/clae_code"        # writable on Spaces; falls back below
CONFIG_PATH = "configs/large_2kh.yaml"
HF_REPO     = "aryan3212/clae-bengali-encoder"
# Hardcoded for now; this checkpoint must be compatible with CONFIG_PATH.
CKPTS = {
    "last": "last.pt",
}

# The model code (config.py, models/, reconstruct_audio.py) is not pip-installable
# (scripts/ is excluded from packaging), so clone it at startup and import — same
# idiom as scripts/kaggle_eval.py.
if not pathlib.Path(REPO_DIR).exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH,
                        REPO_URL, REPO_DIR], check=True)
    except Exception:
        REPO_DIR = "./clae_code"
        if not pathlib.Path(REPO_DIR).exists():
            subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH,
                            REPO_URL, REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)                  # config, models, losses (repo root)
sys.path.insert(0, str(pathlib.Path(REPO_DIR, "scripts")))  # reconstruct_audio lives here
os.chdir(REPO_DIR)                            # config `_base_:` paths are relative to cwd

from config import load_config                              # noqa: E402
from reconstruct_audio import load_model, reconstruct       # noqa: E402
from huggingface_hub import hf_hub_download                 # noqa: E402

# --- load once at startup ----------------------------------------------------
DEVICE = torch.device("cpu")
cfg = load_config(str(pathlib.Path(REPO_DIR, CONFIG_PATH)))
SR = cfg.data.sample_rate
CHUNK = int(round(cfg.data.segment_seconds * SR))
MODELS = {
    label: load_model(cfg, hf_hub_download(repo_id=HF_REPO, filename=fname,
                                            token=os.environ.get("HF_TOKEN")), DEVICE)
    for label, fname in CKPTS.items()
}
print(f"loaded CLAE x{len(MODELS)}: sr={SR}, segment={cfg.data.segment_seconds}s, "
      f"chunk={CHUNK}, checkpoints={list(CKPTS)}")


def _to_mono_16k(sr: int, wav: np.ndarray) -> np.ndarray:
    """Gradio (sr, int16/float array, mono or stereo) -> float32 mono @ SR."""
    wav = np.asarray(wav)
    if wav.ndim == 2:                       # (S, C) -> mono
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if np.issubdtype(np.asarray(wav).dtype, np.floating) and np.abs(wav).max() > 1.0:
        wav = wav / 32768.0                 # was int16 cast to float
    elif wav.dtype.kind in "iu":
        wav = wav / np.iinfo(wav.dtype).max
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav.astype(np.float32)


LABELS = list(CKPTS)                          # [early, latest], display order


@torch.no_grad()
def run(audio):
    blank = (None,) * len(LABELS) + ("Upload or record some audio first.",)
    if audio is None:
        return blank
    sr, wav = audio
    wav = _to_mono_16k(sr, wav)
    if wav.size == 0:
        return (None,) * len(LABELS) + ("Empty audio.",)
    x = torch.from_numpy(wav).view(1, 1, -1)
    if x.size(-1) < CHUNK:                  # pad short clips up to one window
        x = F.pad(x, (0, CHUNK - x.size(-1)))
    outs, n_frames, latent_dim = [], 0, 0
    for label in LABELS:
        x_hat, (n_frames, latent_dim) = reconstruct(MODELS[label], x, CHUNK)
        outs.append((SR, np.clip(x_hat[0, 0].cpu().numpy(), -1.0, 1.0)))
    dur = wav.size / SR
    info = (f"{dur:.2f}s → {n_frames} latent frames × {latent_dim} dims "
            f"(~{SR / max(1, n_frames):.0f} samples/frame, ~12.5 Hz). "
            f"Encoded/decoded in independent {cfg.data.segment_seconds:g}s windows. "
            f"Same input through {len(LABELS)} checkpoint(s): {', '.join(LABELS)}.")
    return (*outs, info)


DESCRIPTION = """
# CLAE — Bengali Speech Autoencoder

A continuous-latent autoencoder: waveform → latent → waveform. Upload a clip or
record yourself, and hear what survives the encode→decode round-trip using the
`last.pt` checkpoint and the `large_2kh.yaml` configuration. Bengali speech works
best (that's the training data); reconstruction is **intelligible but robotic** —
that's the current model, not a bug.

Processed in independent 3-second windows (the encoder was trained on 3 s segments),
so you may hear mild artifacts at window seams on longer clips.
"""

demo = gr.Interface(
    fn=run,
    inputs=gr.Audio(sources=["upload", "microphone"], type="numpy", label="Input audio"),
    outputs=[gr.Audio(label=f"Reconstruction — {label}", type="numpy") for label in LABELS]
            + [gr.Textbox(label="Latent info")],
    title="CLAE Bengali Speech Autoencoder",
    description=DESCRIPTION,
    flagging_mode="never",
)

if __name__ == "__main__":
    # ssr_mode=False: Gradio 5 enables SSR by default, which on HF Spaces can break
    # the prediction API route ("No API found" on Submit). Disable it.
    demo.launch()
