# Plan: ScaledAdam (`optim/scaled_adam.py`)

## Current implementation snapshot

- Per-tensor parameter RMS `p_rms = rms(p)` is used to normalize gradients (`g = grad / p_rms`) and re-scale the final update by `p_rms`.
- Looks like “scale-invariant AdamW-ish”, but not verified against Zipformer/icefall.

## Open questions to confirm (spec/paper)

- What exact reference is intended:
  - `icefall/zipformer`’s `ScaledAdam`? (reference: `icefall/egs/librispeech/ASR/zipformer/optim.py:257`, pinned commit in `EXTERNAL_CODE_REFERENCES.md`)
  - Another “ScaledAdam” variant (per-dimension scaling, clipping schedule, etc.)?
- Is scaling per-tensor, per-channel, or per-parameter-element?
- Any special handling:
  - gradient clipping / “clipping heuristics”
  - warmup / learning-rate schedule coupling
  - parameter constraints (e.g., “max change” per step)
  - bias corrections and eps placement

## Step-by-step implementation plan

1. Identify the reference implementation and pin it:
   - paper/spec citation OR upstream code permalink + commit hash (see `EXTERNAL_CODE_REFERENCES.md`).
2. Add a small “optimizer parity harness” (pure-PyTorch) that:
   - initializes a fixed set of tensors + grads (seeded),
   - runs one step of the reference algorithm and the local one,
   - compares parameter deltas (within tolerance).
3. Update `optim/scaled_adam.py` to match reference behavior:
   - scaling granularity,
   - clipping / schedules,
   - moment updates and eps placement.
4. Validation:
   - Unit test for one-step delta parity with the harness.
   - Smoke test training step on a tiny model (forward/backward/step) to ensure stability.
5. Update `UNCERTAINTIES.md` with the confirmed reference and key hyperparameters.

## Exit criteria

- Local `ScaledAdam` matches the chosen reference on a deterministic parity test.
