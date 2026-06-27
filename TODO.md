# TODO — Proper Full Training Run + "Win a Benchmark" Plan

> ⚠️ **The evaluated checkpoint is the wrong (small) one.** `aryan3212/my-model`
> = `local_6gb` config = **0.43M params**. The intended ~5M model is `exp0`
> (encoder d_model=192, 2.88M). exp0's checkpoint is NOT on this machine or HF —
> locate/upload it and re-run the eval suite with `--ckpt`. Config ladder:
> local_6gb 0.29M → local_13gb 1.47M → exp0 3.39M (front+enc).

Context from the current checkpoint (`aryan3212/my-model`, step 154k):
- **0.4M params** total (encoder 0.25M; d_model=64, 3 layers, 2 heads). ~200× smaller than WavLM-base+ (95M) / Mimi. **This is the local_6gb dev model, not exp0.**
- Losses were **JEPA (3.0) + SIGReg (0.2)** dominant; **STFT recon = 0.005 (≈off), waveform L1 = 0 (off)**.
- 2.5 s segments, 16 kHz, ~12.5 Hz encoder frame rate.
- Eval verdict (see `runs/eval/SUMMARY.md`): strong on **speaker/gender** (stationary identity), weak on **emotion** (dynamic prosody), inconclusive on ASR (probe data-starved). **Beats all baselines on speaker-verification EER** — the candidate win.

The current model tests the thesis "JEPA+SIGReg alone captures semantic structure." Result: it captures *identity*, not *content/prosody*. Mimi adds reconstruction and beats us on content/emotion. So the next run should test whether **reconstruction + capacity + data** closes the gap while keeping the speaker-verification edge.

## 1. Model capacity (biggest lever)
- [ ] `encoder.d_model` 64 → **256** (try 128 first for a fast ablation).
- [ ] `encoder.n_layers` 3 → **6–8**; `num_heads` 2 → **4–8**.
- [ ] Target ~10–30M params (still small, but a real model). Keep it the smallest thing that wins a benchmark — efficiency is the story.

## 2. Turn on the full loss suite
- [ ] `loss.stft_weight` 0.005 → **~1.0** (multi-res STFT is currently negligible).
- [ ] `loss.wav_l1_weight` 0 → **0.1–1.0** (waveform L1, currently off).
- [ ] Keep `jepa.weight` and `sigreg.weight`, then **rebalance** so recon and JEPA are comparable in magnitude (log the per-term losses to tune).
- [ ] The decoder exists + has weights but was trained against ~0 recon weight — it's effectively untrained. With recon on, it actually learns.

## 3. Data (scale up)
- [ ] Build packed manifests from the full downloaded corpus via `scripts/housekeeping.py build` (OpenSLR53 ✓, regspeech12 ✓, indicvoices ~, + shrutilipi/subak/common_voice/bengaliai as they land).
- [ ] Current ckpt trained on `data/manifests/combined_*.jsonl` — confirm size; scale to the full set.
- [ ] Hold out **kathbath** + a speaker-disjoint slice for eval (never in pretrain).

## 4. Frame rate (deliberate choice)
- [ ] 12.5 Hz is fine for paralinguistics (speaker/gender/emotion) and is an efficiency feature — keep it if that's the thesis.
- [ ] If you want content/ASR too, reduce frontend stride to hit **25–50 Hz** (Mimi=12.5 Hz but distills WavLM for content; we have no content supervision, so higher rate may help). Treat as an ablation, not a default.

## 5. Training hygiene
- [ ] Longer schedule; cosine LR + warmup; larger batch (grad-accum if VRAM-bound — note: dev box is a 6 GB GTX 1660, so the real run needs a bigger GPU).
- [ ] Log per-loss-term curves (jepa / sigreg / stft / wav) to W&B to catch one term dominating.
- [ ] Checkpoint + run the probe suite every N steps (see §7).

## 6. Ablations to actually answer the thesis question
Run small/short versions of each, compare on the probe suite:
- [ ] **A — current**: JEPA+SIGReg only (baseline = existing ckpt).
- [ ] **B — +reconstruction**: add STFT+wav L1. Does emotion/content improve? Does speaker hold?
- [ ] **C — +capacity** (d_model 256). Marginal value of params.
- [ ] **D — B+C**: the real run.
Hypothesis: B/D lift emotion + ASR toward Mimi while A/baseline keep the speaker edge. If reconstruction *hurts* speaker EER, that's a finding too.

## 7. The benchmark to win (priority)
Speaker verification is the live candidate — ours already beat WavLM/MMS/Mimi on EER.
- [ ] **Rigorous speaker-verification eval** (`eval/eval_speaker_verif.py`): 100–300 speakers, proper target/non-target trials, EER + minDCF, mean and mean+std pooling, all 5 models. (Built — run it.)
- [ ] If ours wins at scale: headline = **"0.4M-param / 64-dim latent beats 95M WavLM-base+ on Bangla speaker verification."** Efficiency + identity story.
- [ ] Secondary candidates if speaker holds: **gender** (cheap, near-ceiling), **age** (Common Voice tags), **accent/dialect** (IndicVoices region labels) — all identity/stationary, our strength.
- [ ] Avoid leading with emotion/ASR until reconstruction run proves it moved.

## 7b. Deferred eval — Whisper adapter probe (Tier 3, offline)
Strongest *linguistic-content-of-the-encoder* eval; NOT for the 51-min Kaggle window.
- [ ] Train a small adapter: `encoder frames (B,d,T @ 12.5Hz) → projection/upsample → Whisper decoder → Bangla text`. Freeze the encoder and Whisper; train only the adapter.
- [ ] Handle the frame-rate mismatch (12.5Hz → Whisper's expected rate) inside the adapter.
- [ ] Report WER/CER on a speaker-disjoint Bangla holdout. Threshold of interest: WER < 50% = encoder carries real linguistic content.
- [ ] Distinct from the decoder-intelligibility oracle (Whisper end-to-end ASR on *decoded audio*) — that tests the decoder, this tests the encoder. Both deferred out of the 51-min run.

## 8. Eval harness status (done, reusable)
`eval/repr_bench.py` (shared) + `eval_speaker_eer.py`, `eval_speaker_id.py`,
`eval_emotion.py`, `eval_emotion_temporal.py`, `eval_emotion_transformer.py`,
`eval_repr_cluster.py`, `eval_speaker_verif.py`. Results in `runs/eval/`.
Embeddings cached under `runs/eval/embeddings/` (pool-aware).
