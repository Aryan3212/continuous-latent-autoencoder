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
- Losses: `losses/multires_stft.py` + LeJEPA + SIGReg (`models/sigreg.py`)

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
