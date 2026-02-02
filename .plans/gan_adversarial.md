# Plan: Add adversarial (GAN) training for reconstruction

Target: implement the “Reconstruction + GAN discriminator loss” described in your research note, i.e. train the decoder (generator) with an adversarial objective in addition to the reconstruction loss **in the same joint training run** (with safety knobs to delay/disable GAN if needed).

## References to pin

- Your doc requirement: “GAN-based discriminator loss + noise-augmented decoder”.
- External reference to choose (needed): specify whether you want a HiFi-GAN style multi-period + multi-scale discriminator and feature matching, or a simpler single discriminator.

## Current state (repo)

- `models/discriminators.py:6` is explicitly a placeholder and is not wired into `train.py`.
- `configs/exp3_gan.yaml:1` is a placeholder and `gan.enabled` is false.

## Step-by-step plan

1. Choose discriminator family (pin a reference repo/paper)
   - Option A (recommended for audio): Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD) + feature matching.
   - Option B (minimal): keep a single 1D conv discriminator (fast to implement, lower quality).
2. Implement discriminators
   - Replace `models/PlaceholderDiscriminator` with the chosen discriminator(s).
   - Expose config knobs for periods/scales and channel sizes.
3. Implement GAN losses
   - Add generator adversarial loss, discriminator loss, and optional feature matching loss.
   - Decide GAN loss type (hinge vs least-squares) and document it.
4. Wire into training (`train.py`)
   - When `gan.enabled`:
     - update discriminator with real vs reconstructed waveform,
     - update generator with reconstruction + adversarial (+ feature matching),
     - keep SIGReg / LeJEPA objective on encoder outputs as designed.
   - Add safety knobs:
     - `gan.start_step` (delay GAN until recon stabilizes, default 0 to match “simultaneous” training)
     - `gan.d_steps_per_g_step` (e.g., 1)
5. Validation
   - Smoke test: 1–2 steps on CPU to ensure both optimizers step and grads flow.
   - Add a small script/config to overfit a tiny clip to confirm the GAN loss decreases.
6. Update `UNCERTAINTIES.md`
   - Mark GAN training as implemented and cite the chosen reference.
