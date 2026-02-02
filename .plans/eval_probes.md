# Plan: Evaluation probes / CTC probe (`eval/eval_asr.py`)

## Current implementation snapshot

- Current probe expects “precomputed features” from `iter_frame_features`.
- Trains a linear head with `CTCLoss`, greedy-decodes, reports WER via `jiwer`.
- Not wired directly to encoder internals; kept minimal.

## Open questions to confirm (spec/paper)

- What representation the probe should use:
  - encoder output `hE` (frame-level),
  - latent `z`,
  - pooled embedding.
- Data + tokenizer assumptions:
  - char-level vs BPE vs wordpieces,
  - text normalization rules (case, punctuation),
  - manifest format fields.
- Evaluation protocol:
  - freeze encoder fully vs allow BN/LN stats,
  - training steps/optimizer/schedule,
  - dev/test splits and reporting.

## Step-by-step implementation plan

1. Decide/confirm the probe’s input representation and document it in the script help.
2. Wire the probe to run the model end-to-end:
   - load checkpoint,
   - compute features from raw audio through the encoder (or through the latent path),
   - cache features optionally.
3. Make tokenization explicit:
   - implement char-level baseline (current),
   - optionally add a “plug-in” BPE tokenizer path if needed.
4. Validation:
   - smoke run on a tiny manifest (1–2 examples) to ensure the pipeline works.
   - add a “dry-run” mode that only loads + forwards to catch shape issues.
5. Update `UNCERTAINTIES.md` with the confirmed probe protocol.

## Exit criteria

- `eval/eval_asr.py` trains/evaluates a CTC head directly on model outputs with a documented protocol.

