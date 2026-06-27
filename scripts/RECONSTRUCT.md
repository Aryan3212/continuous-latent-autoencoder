# Listening to the autoencoder — `reconstruct_live.py`

Run the trained continuous-latent autoencoder on your own audio (record from the
mic, or pass files) and hear the reconstruction. Tested on an M1 MacBook Air, CPU.

## One-time setup

The repo's `pyproject.toml` pins torch to a CUDA index, so build a small
**CPU-only** venv instead of using `uv run`. From the repo root:

```bash
uv venv --no-config --python 3.11 .venv-recon
uv pip install --no-config --python .venv-recon/bin/python \
    torch torchaudio numpy pydantic pyyaml soundfile sounddevice huggingface_hub
```

The model is ~3M params, so CPU fp32 is instant — no GPU or quantization needed.

## Run

**Mic mode** — Enter to start recording, Enter again to stop; it reconstructs and
plays the result, then loops (Ctrl-C to quit):

```bash
.venv-recon/bin/python scripts/reconstruct_live.py \
    --config configs/kaggle_3m_gan.yaml \
    --hf_repo aryan3212/clae-bengali-encoder
```

> macOS asks for microphone permission on the first recording — allow it (or
> System Settings → Privacy & Security → Microphone → your terminal app),
> otherwise sounddevice raises a PortAudio error.

**File mode** — reconstruct one or more files (mp3 / flac / wav / …). With
`--out_dir` it also writes `<name>_orig.wav` / `<name>_recon.wav` pairs:

```bash
.venv-recon/bin/python scripts/reconstruct_live.py \
    --config configs/kaggle_3m_gan.yaml --hf_repo aryan3212/clae-bengali-encoder \
    clip.mp3 voice.flac --out_dir recon_out
```

## Options

| Flag | Meaning |
|------|---------|
| `--hf_repo REPO` | download `last.pt` from a HF repo (cached after first use) |
| `--ckpt PATH` | use a local checkpoint instead of `--hf_repo` |
| `--config PATH` | architecture config — must match training (`configs/kaggle_3m_gan.yaml`) |
| `--chunk_seconds N` | window length (default `3.0` = `data.segment_seconds`) |
| `--out_dir DIR` | save orig/recon wav pairs |
| `--no_play` | reconstruct only, don't play audio |
| `--device mps` | run on the M1 GPU (default: CPU) |

Audio is processed in independent 3-second windows (matching training) and
concatenated, so expect mild artifacts at the window seams.

> Always invoke `.venv-recon/bin/python` directly — `uv run` would pull the CUDA
> torch wheel and fail on Apple Silicon.
