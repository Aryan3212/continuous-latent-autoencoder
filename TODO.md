# TODO — Proper Full Training Run + "Win a Benchmark" Plan

The current model tests the thesis "JEPA+SIGReg alone captures semantic structure." Result: it captures *identity*, not *content/prosody*. Mimi adds reconstruction and beats us on content/emotion. So the next run should test whether **reconstruction + capacity + data** closes the gap while keeping the speaker-verification edge.

## 1. Model capacity (biggest lever)
- [ ] `encoder.d_model` 64 → **256** (try 128 first for a fast ablation).
- [ ] `encoder.n_layers` 3 → **6–8**; `num_heads` 2 → **4–8**.
- [ ] Target ~10–30M params (still small, but a real model). Keep it the smallest thing that wins a benchmark — efficiency is the story.

## 2. Turn on the full loss suite
- [ ] `loss.stft_weight` 0.005 → **~1.0** (multi-res STFT is currently negligible).
- [ ] `loss.wav_l1_weight` 0 → **0.1–1.0** (waveform L1, currently off).
- [ ] Keep `jepa.weight` and `sigreg.weight`, then **rebalance** so recon and JEPA are comparable in magnitude (log the per-term losses to tune).

## 3. Data (scale up)
- [ ] Current ckpt trained on `data/manifests/combined_*.jsonl` — confirm size; scale to the full set.
- [ ] Hold out **kathbath** + a speaker-disjoint slice for eval (never in pretrain).

## 5. Training hygiene
- [ ] Longer schedule; cosine LR + warmup; larger batch (grad-accum if VRAM-bound — note: dev box is a 6 GB GTX 1660, so the real run needs a bigger GPU).
- [ ] Log per-loss-term curves (jepa / sigreg / stft / wav) to W&B to catch one term dominating.
- [ ] Checkpoint + run the probe suite every N steps (see §7).

## 6. Ablations to actually answer the thesis question
Run small/short versions of each, compare on the probe suite:
- [ ] **B — +reconstruction**: add STFT+wav L1. Does emotion/content improve? Does speaker hold?
- [ ] **D — B+C**: the real run.
Hypothesis: B/D lift emotion + ASR toward Mimi while A/baseline keep the speaker edge. If reconstruction *hurts* speaker EER, that's a finding too.

## 7. The benchmark to win (priority)
Speaker verification is the live candidate — ours already beat WavLM/MMS/Mimi on EER.
- [ ] **Rigorous speaker-verification eval** (`eval/eval_speaker_verif.py`): 100–300 speakers, proper target/non-target trials, EER + minDCF, mean and mean+std pooling, all 5 models. (Built — run it.)
- [ ] Secondary candidates if speaker holds: **gender** (cheap, near-ceiling), **age** (Common Voice tags), **accent/dialect** (IndicVoices region labels) — all identity/stationary, our strength.

## 8.5 VisReg
- [ ] Test with visreg (visual/visibility regularization) — add to training loop and eval suite.

## 9. Eval harness status (done, reusable)
`eval/repr_bench.py` (shared) + `eval_speaker_eer.py`, `eval_speaker_id.py`,
`eval_emotion.py`, `eval_emotion_temporal.py`, `eval_emotion_transformer.py`,
`eval_repr_cluster.py`, `eval_speaker_verif.py`. Results in `runs/eval/`.
Embeddings cached under `runs/eval/embeddings/` (pool-aware).
