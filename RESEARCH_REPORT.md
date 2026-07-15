# CLAE: A 2.5M-Parameter Continuous-Latent Speech Autoencoder for Bengali

**Run:** `final_aryan` · config `configs/kaggle_3m_gan.yaml` · 2× Tesla T4 ·
~30k steps (resumed; max_steps configured 70k) · trained 2026-06-23 → 2026-06-25.
**Eval:** Common Voice 24 Bengali (`validated.tsv`), frozen-feature linear probes.

---

## 1. Headline result

> **A 2.5M-parameter, 12.5 Hz continuous-latent autoencoder matches or beats
> WavLM-base (95M, ~38× larger) on Bengali speaker / gender / age probing**,
> using a latent whose utterance-pooled representation occupies only ~6 effective
> dimensions. Its latent geometry aligns more with a supervised speaker-embedding
> model (ECAPA) than with a content-rich SSL model (WavLM) — the encoder is
> **identity / paralinguistics-centric**. The open weakness is decoder perceptual
> quality (audio is intelligible but robotic).

| Probe (balanced acc.) | chance | mel+PCA (0-param) | **CLAE (2.5M)** | WavLM-base (95M) | ECAPA-TDNN (6M, supervised) |
|---|---|---|---|---|---|
| Gender (binary) | 0.50 | 0.80 | **0.91** | 0.89 | 0.98 |
| Speaker (7-way)¹ | ~0.14 | 0.77 | **0.85** | 0.86 | 0.96 |
| Age (multi-class) | ~0.20 | 0.29 | **0.39** | 0.28 | 0.47 |
| z_rank (effective dim of pooled emb.) | — | 3.5 | 6.2 | 13.9 | 65.0 |

¹ Underpowered: only 7 speakers had ≥4 clips in the sampled set — indicative, not solid (see §6).

**CKA (latent-geometry similarity to CLAE):** `clae~wavlm = 0.11`, `clae~ecapa = 0.23`.
CLAE is closer to a speaker-embedding space than to WavLM's SSL space.

---

## 2. Model architecture

Waveform (16 kHz) → conv frontend → Conformer encoder → continuous latent `z`
→ conv/FiLM decoder → waveform. `z` is what the decoder **and** the probes consume;
the self-supervised losses act on a separate projector output `p` so loss-space and
representation-space are decoupled.

| Block | Spec | Params (approx.) |
|---|---|---|
| **Frontend** (`ConvFrontend`) | 5× strided Conv1d (k=[10,8,8,4,4], s=[5,4,4,4,4], ch→128), GroupNorm+GELU, groups=8. Stride product **1280 → ~12.5 Hz** tokens. | — |
| **Encoder** (`Encoder`+Conformer) | d_model **128**, **5** layers, **4** heads, FFN 320, conv kernel 31, RoPE MHSA, dropout 0.1. | ~1.45M |
| **MHC** (`MHCWrapper`) | Manifold Hyper-Connections on layers ≥2: 2 streams, Sinkhorn (10 iters), τ=0.05, α_init=0.01, identity-mix. **Value unproven — pending ablation.** | small |
| **Projector** | per-frame MLP 128→512→512→**48**, BatchNorm. JEPA + SIGReg act here. | — |
| **Decoder** (`WaveformDecoder`) | ConvTranspose upsample (strides [4,4,4,4,5]), FiLM ResBlocks (dilations [1,3,9]), 256 ch, film_hidden 128. | ~0.79M |
| **Discriminator** (`MultiPeriodDiscriminator`) | HiFi-GAN MPD, periods [2,3,5,7,11], slim ch [24,48,64,96]. Train-only. | ~0.5M |

**Generator total ≈ 2.5–2.8M params.**

---

## 3. Training data

Bengali speech, packed to 16 kHz mono and built into train/val JSONL manifests via
`scripts/housekeeping.py`:

- **Common Voice 24 Bengali** (`sajidullah03/common-voice-24-bn`) — crowd-sourced
  read speech, the demographically-labelled source (also our eval set).
- **regspeech12** (`mdrezuwanhassan/regspeech12`) — regional Bengali speech.

Segments: **3 s @ 16 kHz**. The encoder
only ever saw 3 s windows, so eval encodes/decodes in independent 3 s windows too.

**Augmentation** (train only): additive noise (SNR 0–25 dB), low-pass (2.7–8 kHz —
deliberately preserving the timbre band the gender/age probes need), gain (0.6–1.5×),
clipping (p=0.3); plus JEPA waveform chunk-masking (target ratio 0.25, spans 2–8 frames).

---

## 4. Loss functions

Objective: `0.1·L_recon + 6·L_jepa + 0.05·L_sigreg (+ adv + FM from step 20k)`,
where `L_recon` is `L_stft` or `L_mel` selected by `loss.recon_type` (the two are
ablated against each other).

- **Multi-resolution STFT** (`L_stft`, w=0.1): spectral convergence + magnitude +
  log-magnitude over FFT sizes [256,512,1024,2048]. Deliberately light —
  reconstruction is a regularizer here, not the primary objective.
- **Mel-spectrogram** (`L_mel`): mel-scaled magnitude / log-magnitude comparison
  (`loss.recon_type: mel`). Inherently magnitude/log-magnitude weighted, so it
  leans toward the speech-relevant part of the spectrum.
- **V-JEPA** (`L_jepa`, w=6): dense global/local prediction in projector space
  (`l_global + l_predict + context_weight·l_context`). The dominant training signal.
- **SIGReg** (`L_sigreg`, w=0.05): Epps–Pulley sliced Gaussianity test — anti-collapse
  pressure. *(See §7: an earlier run collapsed because SIGReg acted only in projector
  space; this run's healthy z_rank shows the fix held.)*
- **Adversarial + feature matching** (HiFi-GAN MPD, LSGAN), from step 20k, with
  **VQGAN-style adaptive weighting** (`lam_adv`, clamped to ≤1) and FM weight 2.

**Final training metrics (~step 30k):**

| Metric | Value | Reading |
|---|---|---|
| `l_jepa` | 0.025 | converged (global 6e-4, predict 0.012, context 0.013) |
| `l_stft` / `stft_log` | 2.07 / 1.09 | recon plateaued at a coarse level (light weight) |
| `l_wav` | 0.065 | — (untrained term) |
| `l_adv` / `l_disc` / `l_fm` | 1.25 / 2.50 / 0.008 | GAN active, balanced; `lam_adv` 0.92 |
| **`z_rank` (frame)** | **34.3 / 128** | **healthy — not collapsed** |
| `z_rank_utt` | 6.72 | pooled rep is low-rank (matches eval z_rank 6.2) |
| `sim/pos_frame_mse` vs `neg_frame_mse` | 0.010 vs 2.38 | strong positive/negative frame separation |
| `sim/pos_utt_mse` vs `neg_utt_mse` | 0.005 vs 1.84 | strong utterance-level contrast |

---

## 5. Results & interpretation

**Encoder is healthy and efficient.** On all three identity/paralinguistic probes
CLAE clears the mel+PCA acoustic floor and **matches or beats WavLM-base** — a model
~38× its size — winning on gender (0.91 vs 0.89) and age (0.39 vs 0.28) and tying on
speaker (0.85 vs 0.86). ECAPA, a *supervised* speaker model, is the ceiling on these
tasks as expected; CLAE closing most of that gap unsupervised at 2.5M params is the
efficiency story.

**Two-scale rank structure.** Frame-level `z_rank = 34/128` (training) shows the
per-frame latent is genuinely rich and **not collapsed** (cf. §7). But the
*utterance-pooled* effective dimension is only ~6 (eval 6.2 ≈ training `z_rank_utt`
6.72): pooling concentrates onto a few dominant factors — speaker identity, gender,
and pitch/vocal-tract cues that age correlates with. This is the profile of a
representation optimized by JEPA+SIGReg for stable global structure rather than
fine-grained linguistic content.

---

## 6. What's weak / unverified

- **Mimi baseline missing this round.** Failed on a 16 kHz vs 24 kHz sample-rate
  mismatch (Mimi is a 24 kHz codec); resampling fix applied, **re-run pending**. Mimi
  is the most apt comparison (also 12.5 Hz).
- **Linguistic content untested.** No ASR/phoneme eval was run — the probes here are
  all paralinguistic/identity. The encoder's linguistic content is unmeasured.

## 7. What we have *not* tried (open threads)

- **Linguistic-content eval (highest value).** Whisper-adapter probe (CLAE frames →
  adapter → Whisper decoder → Bengali text, WER/CER) to test whether the latent
  carries linguistic content. Deferred to offline. CTC probe
  abandoned — the 12.5 Hz frame rate vs ~8–15 Bengali chars/s makes many samples
  CTC-infeasible.
- **mHC ablation.** Manifold Hyper-Connections are *in* this model but their value is
  unproven — no on/off comparison has been run. Could be deleted.
- **Reconstruction-focused run.** STFT weight is only 0.1 and wav-L1 is off; the
  decoder was effectively trained against weak recon pressure. A run with stronger
  recon + the GAN from earlier is the obvious path to fix audio quality.
- **Frame rate ablation.** 12.5 Hz is great for paralinguistics and is the efficiency
  story, but likely caps linguistic content; 25–50 Hz untested.
- **Capacity / data scale-up.** d_model 128 and a 2-source corpus; effect of larger
  encoder or the full adapter corpus (IndicVoices/Shrutilipi/Kathbath) untested.
- **UMAP/t-SNE phoneme-coloured latent viz** (needs MFA labels) — not yet run.

## 8. One-line summary for external reporting

> *A 2.5M-parameter, 12.5 Hz continuous-latent autoencoder trained on Bengali
> (Common Voice + regspeech) with a JEPA + SIGReg + light-reconstruction objective
> matches 95M-parameter WavLM-base on Bengali speaker/gender/age probing while its
> utterance-pooled latent uses only ~6 effective dimensions — an identity-centric
> representation (latent geometry closer to a speaker model than to WavLM) and a
> strong efficiency candidate for paralinguistic tasks. Reconstruction quality and
> linguistic-content evaluation remain open.*
