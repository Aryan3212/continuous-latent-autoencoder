# Utterance-level SIGReg: what changed and why

## What changed

**Before (frame-level SIGReg).** The encoder emits `z` of shape `(V·B, 192, T)`.
We flattened time and batch into `(V·B·T, 192)` and fed *that* to SIGReg.
Invariance (JEPA) was computed on time-averaged vectors `(V·B, 192)`.

**Problem.** SIGReg saw ~75× more samples per utterance, but most came from the
same clip and were highly correlated. The time-averaged (utterance) subspace had
no isotropy constraint, so the encoder was free to push most structure into AC
components that wash out when you average over time. Result:
`z_rank_res ≈ 155` (frames spread fine), `z_rank_utt ≈ 14` (utterance vectors
live in a 14-dim subspace out of 192).

**Now (utterance-level SIGReg).** SIGReg sees time-averaged vectors
`(V·B, 192)` — the same space JEPA operates on. Applied per view, averaged
across views. Matches the reference `pool_mode="mean"` (Algorithm 2) exactly.

**Effect.** Utterance vectors must look isotropic Gaussian in 192-D. The
encoder can no longer hide invariances in AC-only dims. `z_rank_utt` should
climb toward 50–100+.

## On the observation: `z_rank_utt` peaked at 26 then decayed to 14

That's the symptom of the old setup. Early in training the encoder was not yet
confident about views, so pooled vectors had residual noise spread across dims
— a transient rank bump. As it learned to align views tightly (`l_jepa`
dropped 0.01 → 0.0016), it collapsed utterance representations onto fewer axes.
With no regularizer on the pooled subspace, drift was monotonic downward. The
new term penalizes exactly this.

`z_rank_res = 155` plateauing does **not** make vectors noisy — it means frames
span ~155 effective dims out of 192, which is healthy. "Noisy" would be
`z_var_min → 0` or unstable training; neither was happening.

## Metric glossary

**`z_a_rms`, `z_mask_rms`** — RMS of the encoder output across the entire
batch, one per view: `sqrt(mean(z²))`. Should sit near 1.0 if SIGReg is
tracking N(0,1). Collapse → 0, explosion → large.

**`z_std`, `z_mean`** — global mean/std of all `192·B·T` scalars in `z`.
Captures whether the distribution is centered and unit-scaled. `z_std ≈ 1.0`
corroborates `z_a_rms`.

**`z_var_min / med / max`** — per-dimension variance across the batch, min /
median / max over the 192 dims. *Key collapse signal.* If `z_var_min` drifts
toward 0, that dimension is collapsing. At `0.22` we're alive but one dim is
compressed.

**`z_rank` (participation ratio)** — `(Σλ)² / Σλ²` over eigenvalues of the
batch covariance. "How many dims does the representation effectively use."
Max = 192. 156 is high (good spread).

**`z_rank_utt`** — participation ratio computed on *time-averaged* vectors
`(B, 192)`. What the pooled utterance embedding spans. This is what downstream
classifiers see when they pool. At 14 this was the bottleneck.

**`z_rank_res`** — participation ratio on frame residuals
`z - mean_t(z)` (frame-level variation after removing the utterance mean).
What's left after time-pooling. 155 = per-frame variation is rich.

**Intuition:** `z_rank ≈ z_rank_utt + z_rank_res` roughly. All 156 effective
dims of `z` were expressing *frame-to-frame variation within a clip*, with
only 14 dims encoding *clip-to-clip differences*. That's the inverse of what
you want for speech SSL — clip identity (speaker, phonetics, prosody) should
span many dims.

**`sigreg_view` / `l_sig`** — the SIGReg loss itself. Epps-Pulley distance
between the empirical characteristic function of the sliced projections and
that of N(0,1), integrated over `t ∈ [-5, 5]`, scaled by N. Lower = more
Gaussian. With utt-SIGReg it will jump up initially (new, harder constraint)
then decrease.

**`jepa_diff_rms`** — `sqrt(mean((center - view)²))`. Raw RMS of the
per-sample, per-dim invariance error. Independent of how `l_jepa` is averaged.
1.14 means center and views differ by ~1.14 per dim on average.

**`jepa_to_norm_ratio`** — `jepa_diff_rms / z_a_rms`. Invariance error scaled
by embedding size. 1.13 means the JEPA gap is roughly the same magnitude as
the embeddings themselves. In isolation this is alarming: it says `l_jepa =
0.0016` was low only because embeddings were low-norm in the directions where
views disagree — views disagree *fully* in those dirs. Under utt-SIGReg this
ratio should drop because the utt space is now larger, making per-dim
disagreement less dominating.

## Expected trajectory next run

- `l_sig` starts ~5, decays toward ~1
- `l_jepa` rises from 0.0016 toward 0.01–0.03 (correct regime)
- `z_rank_utt` climbs 14 → 50–100
- `z_rank_res` may dip 155 → ~100 (frame isotropy no longer directly enforced,
  still maintained indirectly via the encoder's unit-variance pressure)
- `jepa_to_norm_ratio` drops below 1 once views share more than a 14-D subspace
