# continuous-latent-autoencoder

This commit should be signed!
Deterministic continuous-latent speech foundation autoencoder:

- Waveform-in (16kHz), strided Conv1D frontend → ~12.5 Hz tokens (hop 1280 samples)
- Encoder produces deterministic continuous latents `z ∈ R^{B×d×T'}` (default `d=16`, i.e. 200 dims/sec at 12.5 Hz)
- Joint objectives (Exp0): waveform reconstruction (Multi-Res STFT) + LeJEPA-style predictive loss + SIGReg
- Decoder is trained to reconstruct waveform and (later) tolerate latent noise

## Quick start (Exp0) with `uv`

1) Create a `uv` environment (recommended Python is 3.11/3.12 for PyTorch wheels):

```bash
uv venv --python 3.11
```

2) Install deps (either):

```bash
uv sync
```

or:

```bash
uv pip install -r requirements.txt
```

3) Prepare a JSONL manifest for audio (16kHz recommended):

Each line:

```json
{"audio_filepath": "/abs/path/to/file.wav"}
```

4) Run:

```bash
uv run python train.py --config configs/exp0.yaml data.train_manifest=/path/train.jsonl data.val_manifest=/path/val.jsonl
```

Artifacts:

- Checkpoints in `runs/<run_id>/checkpoints/`
- Logs in `runs/<run_id>/logs/` (and W&B if enabled)

## Notes

- This repo intentionally starts with a stable, low-compute Exp0 (no GAN, no mixture).
- Exp1+ (mixture, primary classification, latent-noise decoding, GAN) are implemented as config toggles but should be enabled only after Exp0 is stable.

## Folder guide

- `configs/`: experiment configs (Exp0, CALM-like preset, etc.)
- `data/`: dataset and manifest utilities
- `eval/`: probes + evaluation entrypoints (`eval_asr.py`, `eval_recon.py`, `run_all.py`)
- `losses/`: loss functions (multi-res STFT, etc.)
- `models/`: core model components (frontend, encoder, decoder, sigreg, discriminators)
- `optim/`: optimizers + LR schedulers (ScaledAdam, Eden/Eden2)
- `scripts/`: one-off utilities + smoke scripts
- `utils/`: misc helpers (config, logging, checkpointing)
- `paper-summaries/`, `papers/`: references
- `RAE/`, `lejepa/`, `mHC-manifold-constrained-hyper-connections/`, `icefall/`: vendored references (not installed)
- `runs/`: training outputs (checkpoints/logs)

## Running things

Train (Exp0):

```bash
uv run python train.py --config configs/exp0.yaml data.train_manifest=/path/train.jsonl data.val_manifest=/path/val.jsonl
```

CALM-like preset:

```bash
uv run python train.py --config configs/calm_like_exp0.yaml data.train_manifest=/path/train.jsonl
```

Compute latent stats (for decoder normalization):

```bash
uv run python scripts/compute_latent_stats.py --config configs/exp0.yaml --ckpt /path/to/ckpt.pt --out runs/latent_stats.pt
```

Recon evaluation:

```bash
uv run python -m eval.eval_recon --config configs/exp0.yaml --ckpt /path/to/ckpt.pt --manifest /path/to/manifest.jsonl --out runs/recon.json
```

Run all eval (recon + probes + baselines if available):

```bash
uv run python -m eval.run_all --config configs/exp0.yaml --ckpt /path/to/ckpt.pt --manifest /path/to/manifest.jsonl --out_dir runs/eval
```

Smoke tests:

```bash
PYTHONPATH=. uv run --no-project python scripts/smoke_encoder_mhc.py
PYTHONPATH=. uv run --no-project python scripts/smoke_gan_step.py
PYTHONPATH=. uv run --no-project python tests/test_scaled_adam_parity.py
```
