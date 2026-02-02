# Code reuse / reference map (vendored repos)

This project vendors several upstream research repos in subdirectories (for reference and for selective code porting). This document records what we have **actually reused** vs what is currently **only referenced**.

If/when we port code into this repo’s own modules, update this file with:

- upstream repo + commit hash
- upstream file path(s)
- local destination file path(s)
- what was changed during the port (API/deps/shapes)

## Current state (as of 2026-02-02)

No upstream code has been ported verbatim into this repo’s core modules yet; the vendored repos are currently used as **reference implementations** only. The authoritative list of “where the reference code lives” is in `EXTERNAL_CODE_REFERENCES.md:1`.

## Vendored repos

### `icefall/` (Zipformer + ScaledAdam)

- Purpose: reference implementation for Zipformer encoder and Zipformer’s ScaledAdam/Eden training stack.
- Reference pointers: `EXTERNAL_CODE_REFERENCES.md:1`
- Intended ports (planned, not done yet):
  - Zipformer-derived encoder into `models/` (see `.plans/encoder_zipformer_mhc.md:1`)
  - ScaledAdam (+ Eden) into `optim/` / training loop (see `.plans/scaled_adam.md:1`)

### `lejepa/` (LeJEPA + SIGReg)

- Purpose: reference implementation/snippets for SIGReg and LeJEPA objective wiring.
- Reference pointers: `EXTERNAL_CODE_REFERENCES.md:1`
- Intended ports (planned, not done yet):
  - LeJEPA Algorithm 1 (SIGReg) into `models/sigreg.py`
  - LeJEPA Algorithm 2 objective wiring into `train.py`
  - See `.plans/objective_lejepa.md:1`

### `RAE/` (Representation Autoencoder)

- Purpose: reference for decoder-side robustness mechanics (latent normalization + latent noise) and decoder design ideas.
- Reference pointers: `EXTERNAL_CODE_REFERENCES.md:1`
- Intended ports (planned, not done yet):
  - RAE-inspired decoder mechanics into `models/decoder_generator.py`
  - See `.plans/decoder_rae_inspired.md:1`

### `mHC-manifold-constrained-hyper-connections/` (mHC)

- Purpose: reference implementation for Sinkhorn-projected doubly-stochastic residual mixing (mHC) to integrate into the encoder.
- Reference pointers: `EXTERNAL_CODE_REFERENCES.md:1`
- Intended ports (planned, not done yet):
  - `sinkhorn_log` + minimal mHC module into this repo
  - See `.plans/encoder_zipformer_mhc.md:1`

