# Iteration 0: Skeleton implementation status

Date: 2026-01-27

Goal: Implement the repo skeleton for a deterministic continuous-latent waveform autoencoder with Exp0→Exp2 training objectives, plus evaluation probes and curriculum mining utilities.

## What’s implemented

### Core training (Exp0 / Exp1 / Exp2)

- Waveform input: `x ∈ R^{B×1×T}` at 16kHz with fixed-length random crops (`data/dataset.py`).
- Strided Conv1D frontend (12.5 Hz tokens): `models/frontend_conv.py`.
  - Strides `[5,4,4,4,4]` (product 1280) and per-layer `(Conv1d → GroupNorm → GELU)`.
- Encoder (Conformer-lite): `models/encoder.py` (`Encoder`, `ConformerLiteBlock`).
- Deterministic bottleneck `z ∈ R^{B×d×T'}` with channel-wise norm: `models/encoder.py` (`Bottleneck`).
- JEPA predictor: `models/predictor.py` (MLP predictor for Exp0/Exp1).
- SIGReg (sketched isotropy regularizer): `models/sigreg.py`.
- Decoder generator (vocoder-ish, no GAN): `models/decoder_generator.py` (ConvTranspose upsampler + FiLM-conditioned resblocks).
- Loss: Multi-Res STFT reconstruction (mag + log-mag): `losses/multires_stft.py`.
- Optimizer: ScaledAdam (simplified/approx): `optim/scaled_adam.py`.
- Training loop: `train.py`:
  - Exp0: clean recon (STFT) + JEPA (masked feature view → clean) + SIGReg.
  - Exp1: optional mixture view:
    - mix construction with primary assignment + optional role swap: `data/augment.py` (`maybe_mix_pair`, `swap_prob`).
    - JEPA mix term: predictor(z_mix) → z_primary.
    - optional mixture reconstruction: decode from `z_mix` to `x_primary` (scheduled via `loss.mix_recon.start_step`).
    - optional primary classification loss (`L_primary`) via cosine-sim logits between pooled embeddings.
  - Exp2: decoder latent noise injection (`latent_noise.*`) applied to decoder input only (encoder losses remain on clean z).

### Configs

- Main experiment configs:
  - `configs/exp0.yaml`
  - `configs/exp1_mix.yaml`
  - `configs/exp2_latent_noise.yaml`
  - `configs/exp3_gan.yaml` (placeholder; GAN not wired yet)
- Ablations (as config toggles): `configs/ablations/*.yaml`
  - no SIGReg, no JEPA, stop-grad, mix variants, latent sigma sweeps, latent dim sweeps, AdamW vs ScaledAdam.

### Checkpointing, metadata, logging

- Checkpoints:
  - `runs/<run_id>/checkpoints/last.pt` (every `save_interval_steps`)
  - `best_jepa.pt` (best `val_jepa` when validation is enabled)
  - `best_asr.pt` and `best_composite.pt` (when probes are enabled and `run_eval_on_save` is set)
- Run metadata:
  - `runs/<run_id>/config.yaml` (resolved config)
  - `runs/<run_id>/run_meta.yaml` includes git hash + manifest sha256 hashes (`utils/checkpoint.py`).
- Logging:
  - JSONL step logs in `runs/<run_id>/logs/train.jsonl`
  - W&B init is optional via config (`utils/logging.py`).

### Frozen-encoder probes (intermediate eval)

Implemented as scripts under `eval/` and orchestrated via `eval/run_probes.py`:

- ASR probe: `eval/eval_asr.py`
  - Freezes encoder and trains a simple char-level CTC head; reports WER (train/dev).
  - Expects manifest rows to include a transcript field (default key `text`).
- Emotion probe: `eval/eval_emotion.py`
  - Pooled embedding (mean+std) → MLP classifier; reports accuracy + macro-F1.
  - Expects label field (default `emotion`).
- Gender probe: `eval/eval_gender.py`
  - Pooled embedding → MLP classifier; reports accuracy.
  - Expects label field (default `gender`).

Shared utilities:

- `eval/common.py` loads a frozen encoder from checkpoint and iterates embeddings/features deterministically (no random crop).

### Curriculum mining utilities (seed → filter unlabeled → continue)

Under `data/`:

- VAD-like segmentation (energy threshold): `data/vad_segment.py` (writes `start`/`duration` segments into a new manifest).
- Quality filtering (clip frac, silence frac, SNR proxy): `data/quality_filter.py`.
- Seed embedding index (diagonal mean/var): `data/embed_index.py`.
- Candidate mining by Mahalanobis distance to seed distribution: `data/mine_unlabeled.py`.

Manifest support:

- `data/dataset.py` supports optional `start`/`duration` fields and deterministic crops for eval (`random_crop=False` in eval).

### Scripts

All scripts use `uv run`:

- `scripts/run_exp0.sh`, `scripts/run_exp1.sh`, `scripts/run_exp2.sh`, `scripts/run_exp3.sh`
- `scripts/run_round1_mining.sh` (VAD → filter → seed index → mine)
- `scripts/run_round1_train.sh`

## Known gaps / placeholders (expected before paper details)

- `models/discriminators.py` is a placeholder discriminator; GAN training is not implemented yet.
- Exact architectural/algorithmic fidelity for Conformer-lite, ScaledAdam, SIGReg, and decoder topology is tracked in `UNCERTAINTIES.md`.

## Environment notes

- This container currently has `python3 3.14.2` and no `torch` installed; PyTorch wheels may not exist for 3.14 yet.
- Recommended: use `uv venv --python 3.11` (or 3.12), then `uv sync`.
- Issues are tracked in `CHANGELOG.md`.

## How to run (minimal)

1) Create env + install:

```bash
uv venv --python 3.11
uv sync
```

2) Train Exp0:

```bash
uv run python train.py --config configs/exp0.yaml data.train_manifest=/path/train.jsonl data.val_manifest=/path/val.jsonl
```

3) Enable probes (optional):

- Set `eval.enabled=true`, set `eval.asr.*`, `eval.emotion.*`, `eval.gender.*`, and run with `--run_eval_on_save`.

## Files you should review next

- `UNCERTAINTIES.md` (provide the missing paper/spec details and I’ll align implementations)
- `train.py` (loss weights/schedules; probe triggering; mix schedules)
- `configs/exp0.yaml` / `configs/exp1_mix.yaml` / `configs/exp2_latent_noise.yaml`
