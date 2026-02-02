# Plan: Training hyperparameters (CALM-like)

Target: update configs to match the Continuous Audio Language Models (CALM) paper’s training hyperparameter *style* (sample lengths, latent dimension, latent normalization conventions), but **use Zipformer’s ScaledAdam + its LR scheduling philosophy** (not CALM’s AdamW LR).

## References to pin

- Paper summary: `paper-summaries/continuaudiollm.md`
- Paper PDF (hyperparameter tables):
  - `papers/continuaudiollm.pdf`:
    - Table 13 (VAE hyperparameters): frame rate 12.5Hz for speech, latent dim 32, KL weight 0.01, LR 8e-4, cosine schedule.
    - Table 14 (model training hyperparameters): AdamW β1=0.9 β2=0.95, cosine LR schedule, learning rates in the 5e-5–2e-4 range, long audio sample lengths (we only borrow the LR/schedule/sample-length “shape”; optimizer will be ScaledAdam).
  - Zipformer ScaledAdam reference:
    - `icefall/egs/librispeech/ASR/zipformer/optim.py:257` (`class ScaledAdam`)
  - Zipformer LR scheduler reference:
    - `icefall/egs/librispeech/ASR/zipformer/train.py:1343` (ScaledAdam + `Eden` scheduler)
    - `icefall/egs/librispeech/ASR/zipformer/optim.py:841` (`class Eden`) — suggests `base_lr = 0.04` for ScaledAdam.

## Step-by-step plan

1. Decide which CALM regime maps to our training loop
  - We are not training CALM’s autoregressive transformer; we are training an autoencoder + LeJEPA regularization.
  - We still adopt:
     - segment lengths (e.g., 12s) and latent dimensionality (e.g., 32),
     - latent normalization and decoder-side noise injection (RAE-inspired),
     - segment lengths where feasible.
2. Update `configs/*.yaml`
   - Optimizer defaults:
     - use `optim.kind: scaled_adam` (Zipformer-style) for CALM-like runs.
     - do **not** copy CALM’s AdamW learning-rate values directly onto ScaledAdam; instead:
       - after porting Zipformer ScaledAdam faithfully, adopt a Zipformer-like base LR and clipping config (see `icefall/.../optim.py:841` and `icefall/.../train.py:1343`).
   - Learning rate + schedule:
     - add an LR scheduler hook to this repo (Zipformer uses `Eden`; CALM uses cosine for AdamW).
     - choose one scheduler as the “default” for ScaledAdam runs (recommended: `Eden` for Zipformer parity).
   - Segment seconds:
     - add a “long-context” config variant (e.g., 30s) and keep a “dev” config (e.g., 12s) for iteration speed.
   - Latent dimension:
     - consider setting bottleneck dim to 32 for speech parity (currently default is 16 in README; confirm what you want).
3. Add a config preset
   - `configs/calm_like_exp0.yaml` (or similar) that pins:
     - ScaledAdam (Zipformer-style),
     - CALM-inspired segment length + latent dim (scaled to available GPU),
     - latent normalization toggle (if implemented).
4. Validation
   - Smoke run config parsing and a single training step on CPU.
5. Update `UNCERTAINTIES.md`
  - Replace “training hyperparameters uncertain” with “CALM-like preset” and list the pinned values.
