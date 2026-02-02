# Master plan: resolve `UNCERTAINTIES.md` using paper summaries + reference repos

This plan is meant to be executed item-by-item; each item ends by updating `UNCERTAINTIES.md` to remove/replace the uncertainty with a pinned reference and validated implementation.

## Inputs (what to trust)

- Paper summaries: `paper-summaries/zipformer.md`, `paper-summaries/lejepa.md`, `paper-summaries/rae.md`, `paper-summaries/manifold-hyper-connections.md`, `paper-summaries/fastconformer.md`
- Reference code locations (pinned commits + paths): `EXTERNAL_CODE_REFERENCES.md`

## Global workflow (repeat per uncertainty)

1. **Pin a reference** (paper section and/or reference repo file+line).
2. **Write a “local spec” paragraph** in `UNCERTAINTIES.md` that states the exact intended behavior/choices.
3. **Implement to spec** (surgical change; prefer config-driven toggles when behavior may be revisited).
4. **Add a minimal validation**:
   - parity test (when reference code is available and importable), or
   - invariants/sanity tests (when reference code is not available).
5. **Close the item**:
   - remove the uncertainty entry, or
   - replace it with a confirmed statement + reference links/paths.

## Item plans (mapped to this repo; updated target spec)

Per your latest spec:

- Encoder: Zipformer-derived (`icefall/`) with mHC between intermediate layers (start at layer 2–3, continue to end).
- Decoder: RAE-inspired (`RAE/`) ideas (latent normalization + noise-robust decoding).
- Objective: LeJEPA (Algorithm 2) + SIGReg (Algorithm 1) from `papers/lejepa.pdf` and `lejepa/`.
- Training hyperparameters: CALM-like settings from `papers/continuaudiollm.pdf` (AdamW betas, cosine schedule, segment lengths, latent normalization conventions).
- Training hyperparameters: CALM-like settings from `papers/continuaudiollm.pdf` (LR scale, cosine schedule, segment lengths, latent normalization conventions), but using Zipformer’s ScaledAdam.

### 1) Encoder: Zipformer + mHC

- Theory inputs:
  - `paper-summaries/zipformer.md`
  - `paper-summaries/manifold-hyper-connections.md`
- Reference code:
  - Zipformer: `icefall/egs/librispeech/ASR/zipformer/zipformer.py`
  - mHC: `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py`
- Plan: follow `/.plans/encoder_zipformer_mhc.md`

### 2) ScaledAdam (Zipformer reference)

- Theory input: `paper-summaries/zipformer.md`
- Reference code: `icefall/egs/librispeech/ASR/zipformer/optim.py` (see `EXTERNAL_CODE_REFERENCES.md`)
- Plan:
  1. Decide whether we want **full icefall ScaledAdam fidelity** or a **minimal subset** (document it).
  2. Implement the missing behaviors (at minimum: per-dimension RMS scaling + scale-learning path + clipping).
  3. Add a deterministic one-step parity test against the reference `ScaledAdam` update (seeded tensors).
  4. Update `optim/scaled_adam.py` docstring to cite the pinned reference path + commit.
  5. Update `UNCERTAINTIES.md` to mark ScaledAdam as “aligned with icefall (commit …)”.

### 3) Objective + SIGReg (LeJEPA reference)

- Theory input: `paper-summaries/lejepa.md`
- Reference:
  - `papers/lejepa.pdf` Algorithm 1 (SIGReg), Algorithm 2 (LeJEPA)
  - `lejepa/MINIMAL.md`
- Plan: follow `/.plans/objective_lejepa.md` and `/.plans/sigreg.md`

### 4) Training hyperparameters (CALM-like)

- Theory input: `paper-summaries/continuaudiollm.md`
- Reference: `papers/continuaudiollm.pdf` tables 13/14
- Plan: follow `/.plans/hparams_calm_like.md`

### 5) Decoder (RAE-inspired), Multi-Res STFT, Eval probes

- Theory input:
  - Decoder: `paper-summaries/rae.md`
  - STFT loss / probes: treat as “local spec” unless you pin a reference
- Plan:
  - Decoder: follow `/.plans/decoder_rae_inspired.md`
  - STFT loss: follow `/.plans/multires_stft_loss.md`
  - Probes: follow `/.plans/eval_probes.md`
  - Benchmarks/baselines: follow `/.plans/eval_benchmarks.md`

## Non-UNCERTAINTIES references (future work, not required to close current items)

- RAE reference repo: `RAE/` (`paper-summaries/rae.md`, `EXTERNAL_CODE_REFERENCES.md`)
- mHC reference repo: `mHC-manifold-constrained-hyper-connections/` (`paper-summaries/manifold-hyper-connections.md`, `EXTERNAL_CODE_REFERENCES.md`)
