# Plan: Encoder = Zipformer + manifold hyperconnections (mHC)

Target: replace the current `models/encoder.py` “Conformer-lite” with a **Zipformer-derived encoder (reduced depth for efficiency)**, and add mHC-style hyperconnections starting around layer 2–3 and continuing to the end (configurable).

## References to pin (code + paper)

- Zipformer implementation reference:
  - `icefall/egs/librispeech/ASR/zipformer/zipformer.py:53` (`class Zipformer2`)
  - `icefall/egs/librispeech/ASR/zipformer/scaling.py:425` (`BiasNorm`) and `Swoosh*`
- mHC implementation reference:
  - `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py:45` (`sinkhorn_log`)
  - `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py:1` (update equation described in README)

## What “correct” means (local spec to write into `UNCERTAINTIES.md`)

1. Encoder block family is Zipformer-like (not Conformer-like):
   - uses Zipformer’s normalization/activation choices (BiasNorm, Swoosh*) where applicable,
   - uses Zipformer’s attention/conv/FFN structure (as implemented in icefall) or a clearly documented subset.
2. mHC is applied “between intermediate layers”:
   - enable hyperconnections starting at `mhc_start_layer ∈ {2,3}`,
   - apply for every layer thereafter or every `mhc_period` layers (you choose; default: every layer from start).
3. The integration preserves the mHC equation:
   - `x_{l+1} = H_res x_l + H_post^T F(H_pre x_l, W_l)` with `H_res` doubly-stochastic via Sinkhorn.

## Interaction with per-frame token training

- The encoder must output per-frame tokens at ~12.5Hz (or the chosen frame rate) for:
  - token-level LeJEPA/SIGReg,
  - token-level mix alignment (`Enc(x_mix)` ≈ `Enc(x_primary)`).

## Step-by-step implementation plan

1. Decide the Zipformer surface area we import
   - Option A (highest fidelity): port a minimal Zipformer2 stack from `icefall/.../zipformer.py` and its required helpers from `scaling.py` into `models/zipformer_*.py`, removing external deps (`k2`, `lhotse`, etc.) as needed.
   - Option B (lower effort): keep our current attention/conv primitives but adopt Zipformer’s key building blocks (BiasNorm/Swoosh, multi-rate optional later), and explicitly call it “Zipformer-inspired” in code/docs.
2. Add an “mHC wrapper” module in our codebase
   - Extract the minimum needed from `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py`:
     - `sinkhorn_log`
     - a small `MHC` module that learns `H_res_logits`, `H_pre_logits`, `H_post_logits` and applies them to a `(B,S,T,D)` or `(B*S,T,D)` representation.
   - Make it independent of `einops` if possible (or add `einops` to deps if already used elsewhere).
3. Integrate mHC into the encoder
   - Add config knobs:
     - `mhc.enabled`, `mhc.num_streams`, `mhc.start_layer`, `mhc.period`, `mhc.sinkhorn_iters`, `mhc.tau`, and whether streams are expanded once vs always.
   - Implementation strategy (recommended):
     - keep the Zipformer layer as the branch `F`,
     - for layers `< start_layer`: run standard single-stream residual (S=1),
     - for layers `>= start_layer`: expand to `S` streams and run `MHC(branch=zipformer_layer)` each layer (or every `period`).
4. Validation
   - Unit test: shape preservation for `(B,T,D)` and for masked/padded inputs.
   - Unit test: `H_res` is approximately doubly-stochastic (row/col sums ~1) and non-negative.
   - Smoke test: forward/backward pass with small sizes runs on CPU.
5. Update `UNCERTAINTIES.md`
   - Rename the encoder item to “Zipformer + mHC”.
   - Record the exact reference paths and the mHC insertion policy (`start_layer`, `period`, `num_streams`).

## Known risks / decisions to make early

- Full Zipformer2 parity may be heavy because the icefall implementation depends on a larger ASR stack; the plan expects we either (A) port the needed parts or (B) clearly scope down and document the subset.
- mHC is multi-stream by design; we need to decide how to represent streams for time-series tensors and how to keep compute reasonable.
- “Reduced depth” should be explicit in config (e.g., fewer Zipformer layers/stacks) and captured in `UNCERTAINTIES.md` once chosen.
