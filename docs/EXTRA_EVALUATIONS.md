# Extra evaluations for the TACL evidence package

This is the evaluation checklist for the six matched 30k configurations. It
separates required submission evidence from optional follow-ups. Running any
evaluation script still requires selecting the correct hardware/environment.
All conditions train through the packed TAR backend inherited from
`configs/large_2kh_packed.yaml`.

## Experiment matrix

| Condition | Config | Main question |
|---|---|---|
| Full `R + J + VISReg` | `configs/large_2kh_ablation_full_50k.yaml` | Matched reference |
| Reconstruction only `R` | `configs/large_2kh_ablation_recon_only_50k.yaml` | Does representation learning add transferable information? |
| Representation only `J + VISReg` | `configs/large_2kh_ablation_repr_only_50k.yaml` | What does reconstruction add or preserve? |
| No mHC | `configs/large_2kh_ablation_no_mhc_50k.yaml` | Does mHC improve quality, stability, or sample efficiency? |
| Full objective at 25 Hz | `configs/large_2kh_ablation_25hz_50k.yaml` | Does temporal resolution improve ASR? |
| No decoder corruption | `configs/large_2kh_ablation_no_decoder_corruption_50k.yaml` | Do decoder masking/noise improve reconstruction or transfer? |

All runs stop at 30k but intentionally retain the original 100k scheduler
horizon and use the same packed dataset backend. The primary downstream
evaluation point is the individual step-30k checkpoint for each condition; the
1k checkpoint cadence permits optional earlier or intermediate evaluations if a
curve is surprising.
Use the already-trained historical full checkpoint as the primary reference if its
resolved configuration exactly matches the full-reference config. Otherwise,
run the full reference again and document the mismatch.

## Required evaluations

### 1. Attention-ASR and the frame-rate hypothesis

- Evaluate every objective condition at 12.5 Hz and the dedicated 25 Hz model
  with the same frozen-feature attention-ASR probe.
- Use at least three probe seeds and report mean and standard deviation for WER
  and CER on the unchanged dev/test split.
- Add log-Mel controls downsampled to 12.5 Hz and 25 Hz so the effect of temporal
  resolution is separable from learned representation quality.
- Compare against pinned WavLM, Whisper, and Mimi adapters using the same text
  normalization, probe budget, and split.
- Report feature frame rate, dimension, encoder parameter count, and continuous
  scalars per second beside each result.

Primary comparison: 12.5 Hz full versus 25 Hz full. Secondary comparison:
25 Hz CLAE (`256 * 25 = 6,400` continuous scalars/s) versus 12.5 Hz Mimi
pre-quantization features (`512 * 12.5 = 6,400` continuous scalars/s). This is
a capacity comparison, not a bitrate comparison.

### 2. External, non-overlapping Bengali evaluation

- Use official IndicSUPERB/Kathbath splits for all applicable tasks: ASR,
  speaker identification, speaker verification, keyword spotting, and
  query-by-example.
- Confirm and record that evaluation utterances were not present in the SSL
  pretraining inventory.
- Keep existing OpenSLR-53 speaker and Common Voice age results, but label them
  explicitly as in-domain/transductive because those corpora contributed to
  SSL pretraining.
- Apply the identical frozen-encoder probe and preprocessing protocol to every
  supported baseline.

### 3. Reconstruction quality

On one fixed held-out, non-overlapping manifest, report:

- multi-resolution STFT distance
- waveform L1 (for continuity with existing results)
- SI-SDR
- STOI or ESTOI
- PESQ, or a clearly named replacement if licensing/runtime prevents PESQ
- UTMOS and/or DNSMOS with exact model version

Compare full, reconstruction-only, representation-only, 25 Hz, and
no-decoder-corruption CLAE. The no-mHC run is useful but secondary for
reconstruction. Add Mimi at exactly eight quantizers (paper-style 1.1 kbps);
an all-available-quantizer Mimi result may be reported separately as an upper
quality setting, never as the same operating point.

If feasible, run a small blinded listening study with randomized samples and
report participant count, number of clips, rating question, confidence
intervals, and anonymized condition order.

### 4. Speaker and paralinguistic probes

- Evaluate speaker ID, speaker verification, emotion, and age with at least
  three deterministic probe seeds.
- Preserve speaker-disjoint folds for emotion and age.
- Replace all-pairs-only speaker verification with a documented enrollment/test
  trial protocol. Keep the old all-pairs result only as a secondary diagnostic.
- Report EER and minDCF with speaker-bootstrap 95% confidence intervals.
- For classification, report macro-F1 plus a 95% interval; include accuracy only
  as a secondary metric when classes are imbalanced.

### 5. Representation diagnostics

For full, reconstruction-only, representation-only, no-mHC, and
no-decoder-corruption conditions, plot over training step:

- effective rank
- per-dimension standard-deviation distribution
- covariance off-diagonal magnitude and isotropy
- collapsed-dimension fraction
- clean/augmented positive versus negative similarity

Use identical axes and checkpoint steps. Interpret VISReg using projector-space
diagnostics and downstream usefulness using encoder-latent diagnostics; these
are different spaces and should not be conflated.

## Reproducibility and reporting requirements

- Pin checkpoint hashes, baseline repository revisions, feature layers, pooling,
  manifests, split hashes, seed lists, and probe hyperparameters.
- Report both the 15.29M-parameter feature extractor and the 23.80M-parameter
  complete training autoencoder where relevant.
- Report unique training hours, source-audio exposure hours, GPU-hours, number
  and type of GPUs, world size, wall time, and number of encoder views.
- Use the same downstream protocol for all six CLAE conditions. A result is not
  matched if the probe budget, seed set, data split, preprocessing, or selected
  checkpoint differs.

## Compute terminology

- **Unique training hours:** duration of distinct source recordings in the
  pretraining corpus after the declared filtering/deduplication policy. For this
  project the working figure is approximately 2,000 hours.
- **Exposure hours:** cumulative duration of source segments fed into training,
  counting repeats across epochs and steps. With one GPU and the reference
  settings, a 30k run processes
  `30,000 * 42 * 4 * 3 / 3,600 = 4,200` source-audio exposure hours. Multiply by
  world size for DDP. Do not multiply by the six augmented views; disclose those
  separately as a compute multiplier.
- **GPU-hours:** hardware usage: wall-clock training hours multiplied by the
  number of GPUs. It depends on model speed and hardware, not merely how much
  audio was sampled.

These quantities answer different questions: unique hours describe dataset
scale, exposure hours describe how often training sampled audio, and GPU-hours
describe computational cost.

## Optional follow-up

- A future video experiment can test whether the teacher-free Gaussian-regularized
  alignment recipe transfers to temporal visual data. Frame the current paper
  only as motivating that experiment, not as evidence that video transfer has
  already been demonstrated.
- A short no-regularizer run can visualize collapse, but it is not required for
  the selected six-run matrix.
