# Plan: Evaluation + benchmarking (reconstruction + semantics)

Target: match the evaluation section of your research note:

- Reconstruction metrics: PESQ, mel-distance (and/or LSD), etc.
- Downstream classification: emotion/gender/ASR-style probes (already partially present).
- Baselines: discrete codecs (e.g., EnCodec) and semantic SSL models (e.g., HuBERT).
- Prism-style spectral analysis (needs a pinned reference for “Prism”).

## Current repo state

- Probes exist: `eval/eval_asr.py`, `eval/eval_emotion.py`, `eval/eval_gender.py`.
- No PESQ / EnCodec / HuBERT baseline scripts are implemented in this repo.

## Step-by-step plan

1. Pin evaluation definitions
   - Decide which exact metrics to report and which implementations to use (PESQ has licensing constraints; alternatives may be needed).
2. Add reconstruction metrics
   - Add an `eval/eval_recon.py` that computes:
     - multi-res STFT distance / log-spectral distance,
     - mel-spectral distance (if we add mel code).
3. Add baseline runners
   - EnCodec baseline: encode/decode audio and compute the same recon metrics.
   - HuBERT baseline: extract features and run the same classification probes.
4. Add Prism analysis
   - Once “Prism” reference is pinned, implement the required spectral representation analysis and log plots/summary.
5. Document a single “benchmark entrypoint”
   - Provide `eval/run_all.py` that runs recon + probes + baselines on a manifest and writes a JSON report.

