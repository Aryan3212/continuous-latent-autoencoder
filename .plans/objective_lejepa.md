# Plan: Training objective = LeJEPA (prediction/invariance loss + SIGReg)

Target: implement your **joint, simultaneous** training objective:

- LeJEPA-style representation learning (prediction/invariance + SIGReg) on **per-frame tokens**
- Reconstruction loss on the same latents
- GAN adversarial loss for realism
- Mix objective: random primary assignment (A/B) so that `Enc(x_mix)` matches `Enc(x_primary)` and optionally `Dec(Enc(x_mix))` reconstructs `x_primary`

…while matching LeJEPA’s Algorithm 1/2 definitions where applicable.

## References to pin

- LeJEPA PDF (objective definitions):
  - `papers/lejepa.pdf`: Algorithm 1 (SIGReg) and Algorithm 2 (LeJEPA).
- LeJEPA repo snippet:
  - `lejepa/MINIMAL.md:57` (`class SIGReg`) and its usage in the minimal training loop.

## What “correct” means (local spec to write into `UNCERTAINTIES.md`)

1. Loss form matches LeJEPA:
   - `L = (1 - λ) * L_pred + λ * L_sigreg`
2. No stop-gradient, no teacher/EMA network.
3. SIGReg uses characteristic-function matching with random projections and fixed quadrature points, and can be DDP-synced by seeding projections with `global_step`.
4. Mix objective:
   - sample `primary ∈ {A,B}` randomly (or via `swap_prob`), mix by SNR,
   - enforce token-level alignment between `Enc(x_mix)` and `Enc(x_primary)`,
   - optionally enforce reconstruction `Dec(Enc(x_mix)) -> x_primary` (separation-like).

## Step-by-step implementation plan

1. Implement LeJEPA SIGReg (Algorithm 1)
   - Replace `models/sigreg.py` with a CF-based implementation:
     - sample `A ∈ R^{K×M}` with normalized columns,
     - compute `x_t = (x @ A) * t`,
     - compute empirical characteristic function `ecf = mean(exp(i * x_t))`,
     - compare to Gaussian CF `exp(-0.5 t^2)` with weighted L2 (as in Alg 1),
     - integrate over `t` (trapz).
   - Add config knobs: `num_slices=M`, `t_min`, `t_max`, `t_knots`, `resample_policy`, `seed_with_step`.
2. Implement LeJEPA prediction/invariance loss wiring (Algorithm 2)
   - Decide how to construct “views” for audio:
     - `global_views`: clean views,
     - `all_views`: specaugmented (feature-masked) views; apply specaug after frontend (as we already do).
     - apply mixing on waveform first, then optionally apply specaug on the mixed view too.
   - Embedding level: **per-frame tokens** (treat `B*T'` as samples for SIGReg; LeJEPA loss computed tokenwise).
   - Implement:
     - `centers = mean(emb(global_views))` across global views,
     - `L_pred = mean((centers - emb(all_views))^2)`.
   - Apply SIGReg:
     - `L_sigreg = mean(SIGReg(emb_v))` across views (as Alg 2 indicates).
3. Remove/retire incompatible knobs
   - Disable `stop_grad_target` (must be false).
   - Decide whether `models/predictor.py` remains:
     - if LeJEPA loss is purely center-matching, predictor becomes unnecessary.
4. Add joint losses
   - Reconstruction: multi-res STFT (and any waveform-domain auxiliary terms).
   - Decoder noise: apply noise to latents going into decoder only (keep semantics on clean latents).
   - GAN: add discriminator loss and generator adversarial loss (see `/.plans/gan_adversarial.md`).
   - Mix:
     - `Enc(x_mix)` supervision against `Enc(x_primary)` (token-level),
     - optional `Dec(Enc(x_mix)) -> x_primary` reconstruction (only for mixed samples).
4. Validation
   - Unit tests:
     - SIGReg is near-zero for Gaussian embeddings and higher for collapsed embeddings.
     - LeJEPA loss decreases when views are identical.
   - Smoke run: one forward/backward step with 2 views.
5. Update `UNCERTAINTIES.md`
   - Replace the “SIGReg uncertain points” and “evaluation probe wiring” notes with pinned Algorithm 1/2 references and the chosen view construction.
