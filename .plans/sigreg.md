# Plan: SIGReg (`models/sigreg.py`)

## Current implementation snapshot

- Flattens `(B,d,T')` into `(N,d)`, subtracts mean.
- Projects via random unit-norm columns `R ∈ R^{d×k}`.
- Penalizes `(cov(y) - I)^2` and mean^2.

## Target definition (per LeJEPA)

The LeJEPA paper defines SIGReg as a sketched univariate goodness-of-fit test towards an isotropic Gaussian, via random projections and characteristic-function matching (Algorithm 1 in `papers/lejepa.pdf`), and uses it inside LeJEPA (Algorithm 2).

## Open questions to confirm (spec/paper)

- Exact “SIGReg” definition as used in LeJEPA:
  - statistic (covariance vs correlation),
  - sketching method (random orthogonal vs Rademacher vs structured),
  - normalization (by `N` vs `N-1`, feature scaling),
  - whether to stop-grad anywhere.
- Reference starting point in vendored repo:
  - snippet implementation: `lejepa/MINIMAL.md:57` (pinned commit in `EXTERNAL_CODE_REFERENCES.md`)
  - library composition example: `lejepa/README.md:101` (`SlicingUnivariateTest` + univariate test)
- Which activations to regularize:
  - only `z_clean`,
  - also predictions,
  - pooled/global embeddings,
  - at which layers (every layer vs final).
- How often to refresh the projection matrix (fixed vs periodic resample).

## Step-by-step implementation plan

1. Pin the SIGReg reference:
   - `papers/lejepa.pdf` Algorithm 1 and the vendored snippet in `lejepa/MINIMAL.md:57`.
2. Update `SIGRegConfig` to reflect the real knobs (and keep backwards-compatible defaults where possible):
   - projection type,
   - resample interval,
   - normalization choices.
3. Implement the exact estimator:
   - implement CF/ECF-based statistic with random projections,
   - match quadrature points/windowing to the reference,
   - ensure numerical stability and DDP-compatibility (seeded projections).
4. Validation:
   - Unit test that loss is near-zero for synthetic “white” embeddings.
   - Unit test that loss increases for collapsed embeddings (constant vectors).
5. Update `UNCERTAINTIES.md` with the confirmed definition + where it’s applied in training.

## Exit criteria

- SIGReg matches the chosen LeJEPA spec and has sanity-check tests.
