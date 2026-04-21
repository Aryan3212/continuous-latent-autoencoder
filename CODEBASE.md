# Codebase summary (quick reference)

## Core documentation

- `README.md`: Project overview and setup.
- `CHANGELOG.md`: Log of major repository changes and decisions.
- `HISTORICAL_CHANGES.md`: Detailed audit of the project's commit history, outcomes, and hypotheses.

## Core pipeline (Exp0)

- Frontend: `models/frontend_conv.py` (strided Conv1D → ~12.5Hz tokens)
- Encoder: `models/encoder.py` (Zipformer2-derived) + mHC (`models/mhc.py`)
- Bottleneck: `models/encoder.py` (deterministic, LayerNorm/RMSNorm)
- Decoder: `models/decoder_generator.py` (ConvTranspose stack + optional latent normalization)
- Losses: `losses/multires_stft.py` + LeJEPA + SIGReg (`models/sigreg.py`)

## Training entrypoints

- Main training loop: `train.py`
- Configs: `configs/exp0.yaml`, `configs/calm_like_exp0.yaml`
- Optimizer: `optim/scaled_adam.py` (Zipformer-style ScaledAdam)
- LR schedulers: `optim/lr_schedulers.py` (Eden/Eden2)

## GAN (optional)

- Discriminators: `models/discriminators.py` (MPD/MSD)
- Train wiring: `train.py` under `gan.enabled`

## Evaluation

- ASR probe: `eval/eval_asr.py` (frame-level encoder outputs, dry-run supported)
- Emotion/Gender: `eval/eval_emotion.py`, `eval/eval_gender.py`
- Recon eval: `eval/eval_recon.py`
- Unified entrypoint: `eval/run_all.py`

## Utilities & scripts

- Latent stats: `scripts/compute_latent_stats.py`
- Smoke tests: `scripts/smoke_encoder_mhc.py`, `scripts/smoke_gan_step.py`
- Tests: `tests/test_scaled_adam_parity.py`, `tests/test_sigreg.py`, `tests/test_decoder_rae.py`, `tests/test_multires_stft.py`

## Vendored references (not installed)

- `icefall/`, `RAE/`, `lejepa/`, `mHC-manifold-constrained-hyper-connections/`
