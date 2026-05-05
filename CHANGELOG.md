# Issue log (repo-wide)

Date format: `YYYY-MM-DD`

## 2026-02-01

- Created this file because repo instructions reference `agents.md` as the issue log, but it did not exist yet.
- Research-note audit: current codebase does not yet implement Zipformer+mHC encoder, LeJEPA Algorithm 2 wiring, LeJEPA SIGReg (Algorithm 1), RAE-inspired decoder mechanics, or GAN training; these gaps are now captured in `.plans/*` and `UNCERTAINTIES.md`.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).

## 2026-04-30

- Created `COMMANDS.md` as a quick reference for training, evaluation, and data preparation commands.
- Updated `CODEBASE.md` to include `COMMANDS.md` in core documentation.

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

## 2026-04-17

-   **Stabilized STFT Spectral Convergence (SC) Loss**: Modified `MultiResSTFTLoss` to use the unmasked ground truth magnitude in the denominator when calculating SC on masked regions. This prevents the loss from exploding (previously reaching ~56) when the masked segment of the audio is quiet.
-   **Normalized Masked Losses**: Updated `train.py` to divide masked STFT and L1 losses by `mask_frac`, ensuring the loss scale remains consistent with unmasked validation reconstruction (~3-4).
-   **Fixed Primary Loss Stagnation**: Added a temperature scale (default 0.07-0.1) to the cosine similarity logits in `_primary_logits`. This sharpens the distribution, allowing the `l_primary` classification loss to decrease from its random-chance plateau (~0.69).
-   **Updated Config**: Added `loss.primary.temp` to `configs/exp0.yaml` for easier tuning of the primary component similarity task.

## 2026-04-21

-   **Historical Commit Analysis & Documentation**: Conducted a comprehensive audit of all 23 repository commits. For each commit, analyzed changes, expected outcomes, and underlying research hypotheses.
-   **Created HISTORICAL_CHANGES.md**: Compiled the audit findings into a structured historical reference document.
-   **Git History Rewrite**: Systematically updated all commit messages and descriptions in the repository's history to reflect the refined technical understanding, outcomes, and hypotheses.
-   **Improved Repository Observability**: Standardized commit nomenclature (feat/fix/perf/refactor) and provided detailed context in the commit bodies to improve future maintainability.

