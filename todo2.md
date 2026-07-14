# TODO2 — Evaluation, Benchmarking & Paper Preparation

Status: Planning phase. Training is running on `large_2kh.yaml` config.

---

## 1. Fix Temporal Probe (DONE)

`eval/eval_emotion_temporal.py` has been updated:
- Single fixed 4-speaker split → 4-fold GroupKFold CV (speaker-disjoint)
- Default `--max-utts 2100` → all 7000
- Default models now include `ours_random`
- Epochs aligned: 40 → 30 (matches transformer probe)
- Reports mean±std like the other probes

**Retest all three probes after training converges:**
```bash
uv run python -m eval.eval_emotion --models mimi
uv run python -m eval.eval_emotion_transformer --models mimi
uv run python -m eval.eval_emotion_temporal --models mimi
```

---

## 2. Model Size & Benchmark Selection

### Target competitor models (~95M params, all self-supervised)

| Model | Params | Frame Rate | Dim | Pre-training Objective |
|---|---|---|---|---|
| WavLM-Base-Plus | 95M | 49 Hz | 768 | Masked denoising + utterance mixing, 94k hrs |
| HuBERT-base | 95M | 49 Hz | 768 | Masked prediction of cluster IDs, 960h |
| wav2vec2-base | 94M | 49 Hz | 768 | Contrastive over quantized features, 960h |
| Mimi | 79M | 12.5 Hz | 512 | Adversarial codec + WavLM distillation |

**Minimum viable benchmark:** WavLM-Base-Plus, HuBERT-base, Mimi

### Need to add embedders for
- [ ] Whisper Tiny (39M) and Whisper Base (74M) — encoder hidden states
- [ ] wav2vec2-base (94M)
- [ ] HuBERT-base (95M)

Add to `eval/repr_bench.py` `MODEL_ORDER` and `build_embedder()`.

---

## 3. Evaluation Tasks

### Already implemented — run these
| Task | Script | Metric | Status |
|---|---|---|---|
| Speaker verification | `eval_speaker_verif.py` | EER, minDCF | Ready |
| Speaker ID | `eval_speaker_id.py` | Accuracy | Ready |
| Emotion (linear) | `eval_emotion.py` | Macro-F1 | Ready |
| Emotion (transformer) | `eval_emotion_transformer.py` | Macro-F1 | Ready |
| Emotion (temporal) | `eval_emotion_temporal.py` | Macro-F1 | Ready |
| Reconstruction | `eval_recon.py` | STFT loss, Wav L1 | Ready |

### Should add
- [ ] Gender classification — SUBESCO already has gender labels. Add `--label_key gender` support to `eval_cls_probe.py` or create dedicated script. Easy win, standard paralinguistic probe.
- [ ] CKA (representation similarity) — port from `kaggle_eval.py` to main eval suite. Shows architectural relationship to WavLM/HuBERT/Mimi.
- [ ] Effective rank tracking — add to final eval (already computed mid-training).

### Skip for now
- ASR — wrong paradigm (12.5Hz too coarse for CTC alignment)
- SUPERB full suite — mostly ASR-focused
- Speaker diarization — too complex
- Language ID — need multilingual data

---

## 4. Paper Narrative

> "We train a compact (20M param) self-supervised speech encoder using JEPA + SIGReg on 2k hours of Bengali audio. At 12.5Hz (4x lower frame rate than WavLM/HuBERT), it matches WavLM-level paralinguistic performance while being 5x smaller and producing 4x cheaper representations for downstream use."

### Key results to show
1. Speaker verification EER — our strength (already competitive)
2. Emotion recognition (all 3 probes) — match WavLM at 12.5Hz
3. Reconstruction quality — better codec than Mimi
4. CKA — show representation similarity to larger models
5. Effective rank — no collapse

### Don't include
- ASR results — will hurt the paper
- Training compute comparison — different paradigm, not apples-to-apples

---

## 5. Training Compute Reality Check

| Model | Total audio processed | GPU-hours |
|---|---|---|
| You (200k steps) | 21,333 hours | ~500 (1×4900) |
| wav2vec2-base | 640,000 hours | ~2,430 (64×V100) |
| WavLM-Base+ | 2,880,000 hours | ~3,040 (32×GPU) |

You process 30x less than wav2vec2 and 135x less than WavLM. Don't try to match their raw throughput — different paradigms. Focus on downstream performance per parameter.

---

## 6. Rank Collapse Fix (if needed)

Current issue: `z_rank_utt` at ~2.5 of 48 projector dims after 7.9k steps.

**If this persists in the large_2kh run:**
- [ ] Increase `sigreg.weight` from 0.05 → 0.5 or 1.0
- [ ] Reduce `jepa.weight` from 2.0 → 1.0 (rebalance)
- [ ] Monitor `z_rank`, `z_rank_utt`, `l_jepa`, `l_sig_frm` in W&B

---

## 7. Mimi Reconstruction Evaluation (DONE)

`scripts/eval_mimi_recon.py` created and tested:
- 50-batch results: stft_loss=0.624, wav_l1=0.010
- 100-file results from `mimi_recon_full_2/`: mean stft_loss ~0.47
- Mimi has no reconstruction loss — adversarial-only training
- Key finding: Mimi's encoder before quantization is 512-dim at 12.5Hz

---

## 8. Mimi Emotion Probe Results (all 3 probes, DONE)

| Probe | Mimi | wavlm | mms | ours |
|---|---|---|---|---|
| Linear (mean+std) | 61.7% | 60.6% | 70.1% | 37.3% |
| Transformer ([CLS]) | 61.4% | 62.7% | 60.2% | 38.2% |
| Temporal (attentive) | 53.4% (old split) | — | — | — |

**Note:** Temporal probe result is from old single-split design. Retest with new GroupKFold.

Mimi and WavLM are neck-and-neck. Transformer probe doesn't help Mimi more than linear — the bottleneck is the encoder, not the probe.

---

## 9. CALM Paper Relationship

User's model is based on [CALM (Continuous Audio Language Models)](https://arxiv.org/abs/2509.06926) but diverges significantly:

| Feature | CALM | Our Model |
|---|---|---|
| Goal | Audio generation (TTS) | Representation learning |
| VAE bottleneck | Yes (KL + reparameterization) | No (deterministic, SIGReg instead) |
| Training objective | Consistency modeling + flow | JEPA + SIGReg + reconstruction |
| Causal | Yes | No (bidirectional) |
| Frame rate | 12.5 Hz | 12.5 Hz (matches) |
| Latent dim | 32 | 256 |
| Total params | ~100M | ~20M |

The 12.5Hz frame rate comes from CALM. The encoder architecture (Conformer) and training philosophy (self-supervised representation learning) diverge from CALM's generative approach.

---

## 10. Immediate Next Steps

1. [ ] Wait for `large_2kh` training to reach ~50k steps
2. [ ] Check eval curves in W&B — are speaker/emotion metrics improving?
3. [ ] Check rank metrics — is `z_rank_utt` collapsing?
4. [ ] If rank collapsing, apply fix from §6
5. [ ] Add missing embedders to `repr_bench.py` (Whisper, wav2vec2, HuBERT)
6. [ ] Run full eval suite at 50k, 100k, 200k steps
7. [ ] Add gender classification probe
8. [ ] Port CKA from kaggle_eval.py
9. [ ] Write paper results section
