# Continuous Latent Autoencoder for Bengali Speech Representation Learning

## Executive summary

This thesis work trains a **23.8M-parameter continuous-latent autoencoder
(CLAE)** for 16 kHz Bengali speech.  Its main product is a frozen, frame-level
speech representation for downstream analysis—not a deployment audio codec.
The training objective combines clean-target mel reconstruction with
global/local representation consistency and isotropic-distribution
regularisation.  The current `large_2kh` run is intended to test whether this
compact 12.5 Hz latent sequence retains useful speaker, emotion, age, and
linguistic-content information.  Architecture and objective below are derived
from [`configs/large_2kh.yaml`](configs/large_2kh.yaml) and the training path in
[`train.py`](train.py).

## Model and objective

For a 16 kHz waveform, the encoder emits a continuous 256-dimensional latent
sequence at 12.5 frames/s (a total frontend stride of 1,280 samples).  The
decoder reconstructs the waveform from this latent.  A separate per-frame
projector is used only for representation-learning losses: the decoder and
downstream frozen-feature probes consume the encoder latent `z`, not projector
output `p` ([`models/autoencoder.py`](models/autoencoder.py),
[`models/projector.py`](models/projector.py)).  This separation avoids making
the downstream representation identical to the space directly constrained by
the auxiliary objectives.

The configured active objective is:

```text
L_total = 1.0 L_mel + 0.3 L_JEPA + 0.7 L_VISReg
```

- **`L_mel`** compares the reconstructed waveform with the clean target in an
  80-bin log-mel domain (FFT/window 1024, hop 256).  Although multi-resolution
  STFT loss is implemented, it is not the configured reconstruction loss.
- **`L_JEPA`** uses two global and four local augmented views.  It computes a
  per-frame centre from the global projector features, then applies mean-square
  error to keep both global and local projector features close to that centre.
- **`L_VISReg`** operates on pooled frame/view projector features.  Using 256
  random projections, it encourages zero-centred, unit-scale, isotropic
  Gaussian-like feature distributions, which is intended to discourage
  degenerate representation geometry.

The decoder input is additionally regularised by latent span masking and
Gaussian noise.  GAN/adversarial and feature-matching terms are **disabled** in
this run ([`configs/large_2kh.yaml`](configs/large_2kh.yaml),
[`models/visreg.py`](models/visreg.py), [`train.py`](train.py)).

## Architecture

| Block | Configured design | Parameters |
|---|---|---:|
| Frontend | Five Conv1D stages; channels 128/256/384/512/512, kernels 10/8/8/4/4, strides 5/4/4/4/4; GroupNorm + GELU | 2,890,240 (12.1%) |
| Encoder | 8-layer FastConformer, `d=256`, 8 heads, FFN 1024, 9-tap convolution, dropout 0.1, squeeze-excitation | 12,403,220 (52.1%) |
| mHC | Two hidden streams with mixing at zero-indexed layers 2 and 5; uniform mean readout | included above |
| Projector | Per-frame BatchNorm/GELU MLP, 256 → 512 → 64 | 165,440 (0.7%) |
| Decoder | FiLM-conditioned residual waveform decoder: 768 initial channels; five upsampling stages with strides 4/4/4/4/5; two residual blocks/stage, dilations 1/3/9 | 8,344,225 (35.1%) |
| **Total** |  | **23,803,125 (~23.8M)** |

```text
Clean 16 kHz waveform x
        │  2 global views + 4 local views (augmentation / local masking)
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 5-stage Conv1D frontend: stride 5×4×4×4×4 = 1280                     │
│ 16,000 samples/s  ───────────────────────────────►  12.5 frames/s    │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 8-layer FastConformer + two-stream mHC mixing                         │
│ d=256, 8 heads, FFN=1024, 9-tap convolution, squeeze-excitation       │
└───────────────────────────────────────────────────────────────────────┘
        │ z: continuous encoder latent (256-D per frame)
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
Decoder branch (global view)       Projector: 256 → 512 → 64
span mask + Gaussian noise                 │ p
        │                                  ├── L_JEPA  (0.3): global/local MSE
        ▼                                  └── L_VISReg (0.7): 256 projections
FiLM residual waveform decoder
        │ x-hat
        ▼
Reconstructed 16 kHz waveform ──────────────── L_mel (1.0) ──► clean x

Configured active loss: L = 1.0 L_mel + 0.3 L_JEPA + 0.7 L_VISReg
Configured adversarial/GAN and feature-matching losses: OFF
```

## Data and training status

The training manifests contain four Bengali speech sources.  They are split
95/5 stratified with seed 42.

| Dataset | Utterances |
|---|---:|
| Common Voice Bengali | 1,052,178 |
| OpenSLR-53 | 218,703 |
| regspeech12 | 21,313 |
| shrutilipi | 17,882 |
| **Total** | **1,310,076** |

| Split | Utterances |
|---|---:|
| Train | 1,244,572 |
| Validation | 65,504 |

As of **40,250 of 100,000 optimizer steps**: batch size is 42 with four
accumulation steps (effective batch 168); training uses 3.0-second segments,
AdamW (learning rate 1e-3), 5,000-step warm-up, cosine decay, gradient clip
1.0, and BF16 mixed precision.  The run has processed 6,762,000 forward-pass
samples, equivalent to 20,286,000 seconds / **5,635 hours** of audio and
**5.43** dataset passes.  The planned 100,000-step budget is approximately
14,000 audio hours ([`configs/large_2kh.yaml`](configs/large_2kh.yaml)).

## Evaluation status and next milestone

Frozen-representation evaluation is in progress; results will be appended
separately.  The planned measures are SUBESCO speaker-disjoint emotion
classification, Bengali speaker identification and verification, Common Voice
Bengali age prediction, a fixed-budget Transformer-decoder Bengali ASR probe,
and PCA/UMAP views coloured by speaker or emotion.  These tasks evaluate the
latent representation rather than decoded-waveform quality
([`eval/repr_bench.py`](eval/repr_bench.py)).

The next milestone is to complete these held-out probes, record their metrics
alongside the reference encoders, and determine the representation's balance of
paralinguistic and linguistic information before selecting final thesis
ablations.

## Prompt for an architecture figure

> Create a clean landscape 16:9 academic neural-network architecture figure on
> a white background, using restrained navy, teal, and orange accents. Show:
> “16 kHz Bengali waveform” → five-stage strided Conv1D frontend (channels
> 128, 256, 384, 512, 512; total stride 1280) → “8-layer FastConformer +
> two-stream mHC” → “continuous 256-D latent, 12.5 Hz”. Split the latent into
> (1) a decoder branch with span mask and Gaussian noise, FiLM residual
> waveform decoder, reconstructed waveform, then “80-bin log-mel loss,
> weight 1.0” against the clean waveform; and (2) a projector branch “256 →
> 512 → 64, BatchNorm + GELU”, then “JEPA global-local consistency, weight
> 0.3” and “VISReg Gaussianisation, 256 projections, weight 0.7”. Include the
> formula “L = 1.0 L_mel + 0.3 L_JEPA + 0.7 L_VISReg” and a small footer
> “Adversarial/GAN and feature-matching losses disabled.” Use crisp readable
> labels, thin directional arrows, and no evaluation scores or SOTA claims.
