# Issue log (repo-wide)

Date format: `YYYY-MM-DD`

## 2026-02-01

- Created this file because repo instructions reference `agents.md` as the issue log, but it did not exist yet.
- Research-note audit: current codebase does not yet implement Zipformer+mHC encoder, LeJEPA Algorithm 2 wiring, LeJEPA SIGReg (Algorithm 1), RAE-inspired decoder mechanics, or GAN training; these gaps are now captured in `.plans/*` and `UNCERTAINTIES.md`.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).

## 2026-02-02

- Added `REUSE.md` to document vendored repos and clarify that code is currently referenced (not yet ported) into core modules.
- Noted a design mismatch for downstream evaluation: current emotion/gender probes use pooled embeddings, but the desired direction is sequence heads over frame-level tokens.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).
- Ported Zipformer ScaledAdam into `optim/scaled_adam.py` and added a parity test (`tests/test_scaled_adam_parity.py`).
- Rewired LeJEPA objective (center-matching) and SIGReg Algorithm 1, with tests (`tests/test_sigreg.py`).
- SIGReg now uses Epps–Pulley + sliced univariate test per vendored LeJEPA references.
- Added Eden/Eden2 LR schedulers and a CALM-like config preset (`configs/calm_like_exp0.yaml`).
- Added RAE-style latent normalization support in the decoder and a latent-stats script (`scripts/compute_latent_stats.py`).
- Added MPD/MSD discriminators and GAN losses, wired into `train.py` behind `gan.enabled`.
- Added spectral convergence term and configurability to multi-res STFT loss, with tests.
- Updated ASR probe to support end-to-end encoder features with a dry-run mode.
- Added reconstruction evaluation and a run-all benchmark entrypoint with optional baselines.

## 2026-02-11

- Fixed numerical instability in STFT Spectral Convergence loss by increasing `logmag_eps` from 1e-7 to 1e-3. This prevents division-by-zero explosions when reconstructing silence.
- Documented research diagnostic guidelines in `AGENTS.md`.
