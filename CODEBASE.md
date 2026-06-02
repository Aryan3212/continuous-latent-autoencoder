# Codebase summary (quick reference)

## Core documentation

- `README.md`: Project overview and setup.
- `COMMANDS.md`: Detailed summary of commands for training, evaluation, and data prep.
- `CHANGELOG.md`: Log of major repository changes and decisions.
- `HISTORICAL_CHANGES.md`: Detailed audit of the project's commit history, outcomes, and hypotheses.

## Core pipeline (Exp0)

- Frontend: `models/frontend_conv.py` (strided Conv1D → ~12.5Hz tokens)
- Encoder: `models/encoder.py` (Conformer + rotary) + mHC (`models/mhc.py`)
- Decoder: `models/decoder_generator.py` (ConvTranspose stack + optional latent normalization)
- Projector: `models/projector.py` (LeJEPA MLP head, BatchNorm)
- Losses: `losses/multires_stft.py` + LeJEPA + SIGReg (`models/sigreg.py`)

## Exp0 sizing (configs/exp0.yaml, ~6.0M total)

Encoder is the largest block, then decoder, projector, frontend. `scripts/get_param_count.py` prints the full four-block breakdown.

- Frontend: channels `[64,128,160,192,192]`, strides product 1280 → 12.5 Hz at 16 kHz
- Encoder: d_model 192, 4× Conformer (FFN 576, kernel 31, rotary, mHA-4), mHC on layer 2 only
- Decoder: channels 320, 5× upsample `[4,4,4,4,5]`, 2× FiLM ResBlock per stage (dilations `[1,3,9]`, film_hidden 64)
- Projector: 192→896→896→64 MLP w/ BatchNorm (2 hidden blocks)

## Training entrypoints

- Main training loop: `train.py`
- Configs: `configs/exp0.yaml`, `configs/calm_like_exp0.yaml`

## GAN (optional)

- Discriminators: `models/discriminators.py` (MPD/MSD)
- Train wiring: `train.py` under `gan.enabled`

## Evaluation

- ASR probe: `eval/eval_asr.py` (frame-level encoder outputs, dry-run supported)
- Emotion/Gender: `eval/eval_emotion.py`, `eval/eval_gender.py`
- Recon eval: `eval/eval_recon.py`
- Unified entrypoint: `eval/run_all.py`

## Utilities & scripts

- Smoke tests: `scripts/smoke_encoder_mhc.py`, `scripts/smoke_gan_step.py`
- Tests: `tests/test_sigreg.py`, `tests/test_multires_stft.py`

## Vendored references (not installed)

- `icefall/`, `RAE/`, `lejepa/`, `mHC-manifold-constrained-hyper-connections/`
