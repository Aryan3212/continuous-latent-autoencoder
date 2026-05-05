# Reading the Training Logs

How to interpret every metric emitted by `train.py` per `log_interval_steps`. Healthy ranges, collapse signatures, what to do when something looks wrong.

All metrics are logged to JSONL (`run_dir/log.jsonl`) and to W&B if enabled. Cadence is the value of `train.log_interval_steps` in the config (currently 10).

---

## Quick triage — the dashboard view

Look at these five first. If any is in the "collapse" column for >5k steps, the run is probably broken, regardless of what the loss looks like.

| Metric              | Healthy                          | Warning                           | Collapse                          |
| ------------------- | -------------------------------- | --------------------------------- | --------------------------------- |
| `z_rank_utt`        | rising past 40 → toward 80+      | flat at 20-40                     | stuck < 15 / 192                  |
| `z_rank` (frame)    | > 60                             | 30-60                             | < 25                              |
| `jepa_to_norm_ratio`| 0.15 - 0.5 and stable            | < 0.1 or trending to 0            | < 0.05 (views identical)          |
| `z_var_max/z_var_min` ratio | < 10                     | 10-30                             | > 30 (60× spread = bad)           |
| `l_stft` (val)      | < 1.5 and decreasing             | 1.5-2.0 plateau                   | > 2.0 plateau                     |

`l_jepa_mask` alone tells you almost nothing — see [The JEPA loss is misleading](#the-jepa-loss-is-misleading-on-its-own).

---

## Loss components

### `loss`
Sum of `stft_w·l_stft + wav_l1_w·l_wav + jepa_w·l_jepa + sig_w·l_sig + gan_w·(l_g_adv + l_fm)` plus mix/primary terms if enabled. Total optimization objective. Use the components, not this, for diagnosis — a flat `loss` can hide a falling `l_stft` and a rising `l_jepa`.

### `l_stft`, `stft_sc`, `stft_mag`, `stft_log`
Multi-resolution STFT reconstruction loss (`losses/multires_stft.py:122-127`):
- `l_stft` — average across `[stft_sc, stft_mag, stft_log]`
- `stft_sc` — spectral convergence: relative magnitude error, scale-invariant
- `stft_mag` — L1 on linear magnitude
- `stft_log` — L1 on log magnitude (more sensitive to low-energy bins)

**Healthy:** monotonic decrease through 5k-20k steps to ~1.0-1.5, then slow improvement. **Plateau > 2.0 by step 30k** = decoder can't reconstruct, almost always means the latent doesn't carry enough info (encoder-side problem) OR the decoder is too small / lr is wrong.

### `l_wav`
L1 distance on the raw waveform. Tracks `l_stft` directly. Healthy: drops below 0.05 by 10k steps. Use as a sanity-check for `l_stft`.

### `l_jepa`, `l_jepa_mask`
JEPA invariance — squared distance between view embeddings and their mean (`train.py:_lejepa_invariance`). **`l_jepa_mask` ≈ 0 does NOT mean success.** See [next section](#the-jepa-loss-is-misleading-on-its-own).

Healthy `l_jepa_mask`: drifts down from ~2.0 at step 0 to 0.1-0.5 over 30k+ steps **while `jepa_to_norm_ratio` stays > 0.1**. If `l_jepa_mask` falls fast (< 0.05 by 5k) and `jepa_to_norm_ratio` falls with it (< 0.1), that is collapse, not learning.

### `l_sig`, `l_sig_utt`, `l_sig_frm`, `sigreg_view`
SIGReg test statistic (Epps-Pulley, `models/sigreg.py`) measuring deviation from N(0, I). Average of utterance-pooled and frame-flat branches (`train.py:611-615`).

- `l_sig_utt` — utterance-pooled SIGReg (one vector per clip per view)
- `l_sig_frm` — frame-flat SIGReg (~40 vectors per clip per view at 12.5 Hz, 3-s crop)
- `l_sig` = 0.5 × (utt + frm)
- `sigreg_view` is identical to `l_sig` (legacy alias)

**Healthy:** `l_sig_utt` ≈ 0.02-0.1, `l_sig_frm` ≈ 0.01-0.05. Both stable.

**Warning:** rising `l_sig_utt` while embeddings drift away from N(0, I) — usually fine if rank metrics are good.

**Bad:** `l_sig_utt` near 0 with `z_rank_utt` very low → SIGReg has crushed the utt subspace into a tight isotropic ball with few effective dimensions (the LeJEPA paper's `lamb=0.02` weighting is what prevents this; current `sig_w=2.0` is ~100× stronger).

### `l_jepa_mix`, `l_stft_mix`, `mixed_frac`, `snr_db_mean`
Mixed-utterance auxiliary objectives. Only present if `mix.enabled`. `mixed_frac` is the fraction of the batch that received a mix; `snr_db_mean` is the SNR of the mix.

### `l_g_adv`, `l_fm`, `l_d`, `gan_w`
GAN losses (only when `gan_enabled` and `step >= gan_start`). `l_g_adv` = generator hinge loss, `l_fm` = feature matching, `l_d` = discriminator hinge loss, `gan_w` = adaptive VQGAN-style weight clamped to [0, 10] (`train.py:691-693`). Healthy: `l_d` oscillates around 1.0-2.0; `l_g_adv` decreases slowly. Watch for `l_d` → 0 (discriminator wins, generator stuck).

### `l_primary`
Optional primary objective. Only present if `primary_enabled`.

---

## The JEPA loss is misleading on its own

The invariance term in this repo is computed AFTER pooling each view's frames over time (`_pool_utt`, `train.py:65`). A run can drive `l_jepa_mask` near zero in two very different ways:

1. **Genuine invariance:** the encoder produces similar representations across views, *frame by frame*. `jepa_diff_rms` is small *because* per-frame embeddings agree. Good.
2. **Pool-shortcut collapse:** per-frame embeddings differ a lot, but their *time-averages* coincide because the encoder has dumped phonetic content into mean-zero residuals. `jepa_to_norm_ratio` stays ≥ 0.5 (views differ in raw space) but `l_jepa_mask` is tiny because the means align. **This is the failure mode of LeJEPA Algorithm 2 applied to speech.**

The way to tell them apart is `jepa_to_norm_ratio`:

```
jepa_to_norm_ratio = jepa_diff_rms / z_a_rms
```

| Pattern                                        | Diagnosis                                  |
| ---------------------------------------------- | ------------------------------------------ |
| `l_jepa_mask` low + `jepa_to_norm_ratio` < 0.1 | True invariance (good)                     |
| `l_jepa_mask` low + `jepa_to_norm_ratio` > 0.5 | Pool-shortcut collapse (bad)               |
| `l_jepa_mask` high + `jepa_to_norm_ratio` ~ 1  | Untrained / hasn't started learning yet    |
| `l_jepa_mask` rising                           | Augmentation too aggressive, or LR spike   |

In the 60k-step audit run: `l_jepa_mask=0.014`, `jepa_to_norm_ratio=0.71` → textbook pool-shortcut.

---

## Rank diagnostics — the most informative collapse signals

All three are *participation ratios* of latent covariance eigenvalues: PR = `(Σλ)² / Σλ²`. Range [1, D] where D=192. PR = D means all dims used equally; PR = 1 means one dim carries everything.

### `z_rank` (frame-flat PR, `train.py:715-719`)
PR over `(B*T, D)` — frame embeddings flattened across batch. Captures total latent diversity, both within-utterance (phonetic) and across-utterance (speaker/channel).
- **Good:** > 60, rising
- **Warning:** 30-60, flat
- **Bad:** < 25 → encoder is producing a low-rank latent space

### `z_rank_utt` (utterance-pooled PR, `train.py:722-726`)
PR over `(B, D)` after time-mean-pool — between-utterance diversity. **This is the most diagnostic single number for the LeJEPA shortcut.** When the encoder collapses utterance-level info to a few "global style" dims (speaker timbre, channel, loudness), this drops below 15.
- **Good:** > 40
- **Warning:** 20-40
- **Bad:** < 15 → utterance-level collapse

### `z_rank_res` (frame-residual PR, `train.py:729-733`)
PR over frame-minus-utt-mean — strictly within-utterance variation, removes the speaker baseline. Tracks whether *time-varying* (phonetic) content is preserved. The STFT decoder pressure tends to keep this healthy even when `z_rank_utt` collapses.
- **Good:** > 50
- **Warning:** 30-50
- **Bad:** < 25 → the decoder isn't getting frame-level signal either

**The signature ratio.** Healthy run: `z_rank_res` ≈ `z_rank` ≈ `z_rank_utt + small`. Pool-shortcut collapse: `z_rank_res` >> `z_rank_utt` (e.g. 100 vs 9). Total collapse: all three < 15.

---

## View comparison — `z_a_rms`, `z_mask_rms`, `jepa_diff_rms`, `jepa_to_norm_ratio`

Computed on view-0 (`z_a`) vs view-1 (`z_mask`). Both have shape `(B, D, T)`.

- `z_a_rms`, `z_mask_rms` — root-mean-square embedding norm per view. Healthy: 0.5-2.0, similar across views. If `z_a_rms` → 0 the encoder is dying; if it explodes the SIGReg target is being violated.
- `jepa_diff_rms` — RMS of `(z_a - z_mask)`. Per-frame view divergence in absolute terms.
- `jepa_to_norm_ratio` — `jepa_diff_rms / z_a_rms`. Scale-free per-frame divergence. **The single most important number for telling shortcut from real invariance** (see prior section).

---

## Per-dim variance — `z_var_min`, `z_var_med`, `z_var_max`

Min/median/max of per-dim variance across the SIGReg sample (`models/sigreg.py`). Tells you whether SIGReg is enforcing *uniform* isotropy or letting some dims dominate.

- **Healthy:** `z_var_max / z_var_min` < 10. Median ~ 0.7-1.3.
- **Warning:** ratio 10-30. Some dims are absorbing all the gradient.
- **Bad:** ratio > 30 (60× in the audit run) → SIGReg is ineffective at controlling spread; a few dims are dominating, the rest are near-flat. This is consistent with rank collapse.

---

## mHC diagnostics — `mhc/layer_{i}_alpha`, `mhc/layer_{i}_S_row_entropy`

Logged once per encoder layer that uses mHC (i.e. layers ≥ `mhc.start_layer` at `mhc.period` cadence). Other layers are `nn.Identity()` and are skipped (`models/encoder.py:75-101`).

### `mhc/layer_{i}_alpha`
`sigmoid(H_res_alpha_logit)` — learned mixing coefficient between identity and the Sinkhorn-normalized stream-mixing matrix S (`models/mhc.py:103-105`):
```
h_res = (1 - α) · I + α · S
```
- α = 0 → MHC bypassed (identity residual mixing, like a regular Zipformer)
- α = 1 → full Sinkhorn mixing
- Initialized at `alpha_init=0.01` (≈ bypass)

**Healthy:** alpha drifts from 0.01 toward 0.1-0.5 over warmup. Layers learn how much cross-stream mixing helps.
**Warning:** alpha stays at 0.01 ± 0.001 → MHC isn't being used; consider whether to bother with it.
**Bad:** alpha → 1 with degraded performance → too much cross-talk.

### `mhc/layer_{i}_S_row_entropy`
Entropy of the Sinkhorn doubly-stochastic matrix S, averaged over rows. Range:
- ~0 → S is identity (no inter-stream mixing) — even if alpha is high, mHC is effectively a no-op
- `log(num_streams)` ≈ 0.69 for `num_streams=2` → uniform mixing (each stream pulls equally from all others)

**Reading combined with alpha:**
| α     | S_row_entropy | Interpretation                                  |
| ----- | ------------- | ----------------------------------------------- |
| ~0    | any           | MHC bypassed                                    |
| > 0.1 | ~0            | MHC enabled but Sinkhorn collapsed to identity  |
| > 0.1 | ~ log(streams)| Active stream mixing                            |

If the mHC concern is "is mHC contributing to representation collapse?", the answer is *only* the third row above can plausibly contribute; the first two mean mHC is inert.

---

## SIGReg moments — `z_mean`, `z_std`

Mean and std of `z_a` across all dims and positions. SIGReg targets N(0, I) so:
- **Healthy:** `z_mean` ≈ 0 (|·| < 0.1), `z_std` ≈ 1 (range 0.7-1.3)
- **Bad:** `z_mean` drifting away from 0 with `l_sig_*` not rising → SIGReg has been bypassed somehow

---

## System / housekeeping

- `step`, `epoch` — from `scheduler.epoch` (via `optim/lr_schedulers.py`). Use `epoch` to compare to k2-ssl-style 90-epoch budgets.
- `sigma` — current latent-noise sigma applied to `z_a` before decoding. Schedule defined in config.
- `vram_gb` — peak CUDA memory in GB (use to spot leaks: should be flat across log intervals).
- `loss_stft/{ds_name}`, `loss_wav/{ds_name}` — per-dataset breakdown when training on a multi-source mix. Useful for spotting a dataset that the encoder fails to fit.

---

## Failure-pattern playbook

Each row: a pattern observable in the logs → most likely cause → first thing to check.

| Pattern                                                                    | Cause                                              | First check                                       |
| -------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| `l_jepa_mask` < 0.05 within 5k steps + `jepa_to_norm_ratio` > 0.5          | Pool-shortcut collapse (no frame-level objective) | Add frame-level invariance term                  |
| `z_rank_utt` < 15, `z_rank_res` > 50                                       | LeJEPA pooled-invariance shortcut                  | Same — frame-level term                           |
| `z_rank_utt` and `z_rank_res` both < 15                                    | Total representation collapse                       | LR spike? `z_a_rms` → 0? augmentation too strong? |
| `z_var_max / z_var_min` > 30                                               | A few dims dominating; SIGReg not regulating spread| Lower `sig_w`; or apply VICReg-style hard variance bound |
| `l_stft` plateaus > 2.0 for >20k steps                                     | Decoder under-fitting OR encoder dropping content   | Train decoder alone first; check `z_rank_res`     |
| `l_jepa_mask` rises mid-training                                           | Augmentation intensity ramped too high; LR too high| Check aug schedule; lower LR                      |
| Train WER == Dev WER ≈ 1.0 on probe                                        | Features carry no phonetic signal at all           | Check `z_rank_utt` — almost certainly very low    |
| Train WER << Dev WER on probe                                              | Genuine generalization gap, NOT collapse           | More probe data; bigger probe head; longer train  |
| `mhc/layer_*_alpha` all stuck at 0.01                                      | mHC never engaged                                   | Either fine (mHC inert) or LR too low for it     |
| `z_a_rms` → 0                                                              | Encoder dying (often from exploded grad earlier)    | Check grad-norm logs; LR; gradient clipping       |
| `vram_gb` slowly rising across log intervals                               | Memory leak (cached graphs, unfreed tensors)        | Look for `.detach()` misses; profile             |

---

## What "good" looks like — a healthy run profile

Approximate trajectory of a run that learns phonetic content:

```
step      l_stft   l_jepa_mask   jepa_to_norm   z_rank   z_rank_utt   z_var_max/min
   0       3.5        2.1            1.0           5         3            1
1000       2.4        1.1            0.65         15         8            3
5000       1.8        0.7            0.40         35        18            5
20000      1.4        0.4            0.25         60        35            7
60000      1.1        0.25           0.20         85        55            8
```

Key invariants of the trajectory:
- **All rank metrics are monotonically rising** until they saturate.
- **`jepa_to_norm_ratio` stabilizes around 0.15-0.3, not zero.** Views differ per-frame but agree under the encoder's invariance.
- **`l_jepa_mask` and `l_stft` decrease together.** If `l_jepa_mask` drops while `l_stft` plateaus, the encoder is taking a shortcut.
- **`z_var_max/z_var_min` stays single-digit.** Means SIGReg is doing its job uniformly across dims.

---

## What "collapse" looks like — the audit-run profile

```
step      l_stft   l_jepa_mask   jepa_to_norm   z_rank   z_rank_utt   z_var_max/min
   0       3.5        2.1            1.0           5         3            1
2000       2.7        0.8            1.10         25        12           20
5000       2.5        0.05           0.85         28        11           35
20000      2.1        0.03           0.78         26        10           50
60000      2.05       0.014          0.71         25         9.9          60
```

The pathognomonic features:
- `l_jepa_mask` collapses to near zero very fast (< 5k steps).
- `jepa_to_norm_ratio` does not collapse with it — views differ ~70% of norm even at 60k steps.
- `z_rank_utt` collapses early and stays flat at ~10.
- `z_rank` and `z_rank_res` are higher than `z_rank_utt` (frame info preserved by STFT decoder pressure).
- `z_var_max / z_var_min` blows up (60×).
- `l_stft` plateaus high.
- Train WER ≈ Dev WER ≈ 1.0 on the ASR probe.

When you see this signature, **more steps will not help**. The objective itself rewards staying. The fixes are objective-level (add frame-level invariance, lower SIGReg weight, optionally add an EMA target). See `docs/UTTERANCE_SIGREG_NOTES.md` for the prior analysis.
