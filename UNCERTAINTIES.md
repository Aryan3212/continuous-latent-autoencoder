# Implementation uncertainties (need paper/spec confirmation)

This file tracks parts I implemented based on the plan text alone (without paper/code references), and I’m not fully confident they match your intended algorithms/architectures. When you provide the exact references/specs, I’ll update these components.

Planning docs live in `/.plans/uncertainties_index.md`.

## Encoder / Zipformer + mHC

- Resolved: `models/encoder.py` now uses a Zipformer2-derived encoder layer stack (ported from `icefall/egs/librispeech/ASR/zipformer/zipformer.py` + `icefall/egs/librispeech/ASR/zipformer/scaling.py` into `models/zipformer.py` and `models/zipformer_scaling.py`).
- mHC integration is implemented in `models/mhc.py` and applied from `start_layer=2` every `period=3` layers, using `num_streams=2`, `sinkhorn_iters=10`, `tau=0.05` (configurable in `configs/exp0.yaml`).

## ScaledAdam

- Resolved: `optim/scaled_adam.py` is now a direct port of Zipformer ScaledAdam (icefall `egs/librispeech/ASR/zipformer/optim.py`) including size updates and clipping.

## SIGReg

- Resolved: `models/sigreg.py` now implements LeJEPA Algorithm 1 using Epps–Pulley + SlicingUnivariateTest (see `lejepa/lejepa/univariate/epps_pulley.py` and `lejepa/lejepa/multivariate/slicing.py`), applied to per-frame latents.

## Decoder (RAE-inspired)

- Resolved: Decoder now supports RAE-style latent normalization (optional) and latent-stats loading via `decoder.latent_stats_path`, with a stats computation script (`scripts/compute_latent_stats.py`). Noise control remains in `latent_noise` (now supports fixed noise_tau).

## Adversarial (GAN) reconstruction

- Resolved: Implemented HiFi-GAN style MPD/MSD discriminators with adversarial + feature matching losses, wired into `train.py` behind `gan.enabled`.

## Multi-Res STFT loss details

- Resolved: Multi-res STFT loss now includes spectral convergence + mag/log-mag L1 with configurable hop/window, window type, and eps.

## Training hyperparameters (CALM-like, but ScaledAdam)

- Resolved: CALM-like preset captured in `configs/calm_like_exp0.yaml` (segment length 12s, latent dim 32) with Zipformer-style ScaledAdam + Eden LR scheduler.

## Evaluation probes

- Resolved: `eval/eval_asr.py` now runs end-to-end on encoder outputs (frame-level) with a dry-run mode to verify shapes.
