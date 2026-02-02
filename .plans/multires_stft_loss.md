# Plan: Multi-Res STFT loss details (`losses/multires_stft.py`)

## Current implementation snapshot

- Computes STFT magnitudes for several FFT sizes.
- Loss = L1(mag) + L1(log-magnitude).

## Open questions to confirm (spec/paper)

- FFT/hop/window settings:
  - hop ratio and window ratio per resolution,
  - `center=True` vs `center=False`,
  - window type and normalization.
- Which terms:
  - spectral convergence term,
  - magnitude loss L1 vs L2,
  - log-magnitude clamp/eps.
- Whether to apply loss on raw waveform or pre-emphasized / normalized audio.

## Step-by-step implementation plan

1. Pin the target “multi-res STFT loss” definition (paper or reference repo).
2. Update `MultiResSTFTConfig` to represent the true knobs:
   - per-resolution hop/window (if needed),
   - include/exclude spectral convergence,
   - per-term weights.
3. Implement the missing terms (if any) and ensure numerical stability.
4. Validation:
   - Unit test: identical inputs yield zero loss.
   - Unit test: small perturbation yields small but non-zero loss with correct gradients.
5. Update `UNCERTAINTIES.md` with the confirmed settings and the reference.

## Exit criteria

- Loss matches the chosen reference and has basic correctness tests.

