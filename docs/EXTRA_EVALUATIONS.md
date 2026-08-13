# Extra evaluations for the TACL evidence package

This is the evaluation checklist for the six-run 25k workflow. It
separates required submission evidence from optional follow-ups. Running any
evaluation script still requires selecting the correct hardware/environment.
All conditions train through the packed TAR backend inherited from
`configs/large_2kh_packed.yaml`.

## Experiment matrix

| Condition | Config | Main question |
|---|---|---|
| Packed base (`R + J + VISReg`) | `configs/large_2kh_packed_25k.yaml` | Original packed-data reference at the shared endpoint |
| Reconstruction only `R` | `configs/large_2kh_ablation_recon_only_50k.yaml` | Does representation learning add transferable information? |
| Representation only `J + VISReg` | `configs/large_2kh_ablation_repr_only_50k.yaml` | What does reconstruction add or preserve? |
| No mHC | `configs/large_2kh_ablation_no_mhc_50k.yaml` | Does mHC improve quality, stability, or sample efficiency? |
| Full objective at 25 Hz | `configs/large_2kh_ablation_25hz_50k.yaml` | Does temporal resolution improve ASR? |
| No decoder corruption | `configs/large_2kh_ablation_no_decoder_corruption_50k.yaml` | Do decoder masking/noise improve reconstruction or transfer? |

All runs stop at 25k but intentionally retain the original 100k scheduler
horizon and use the same packed dataset backend. The primary downstream
evaluation point is the individual step-25k checkpoint for each condition; the
five ablations save every 1k steps and the packed base saves every 5k, permitting
optional earlier or intermediate evaluations if a curve is surprising.
The packed-base checkpoint is the only full-objective reference in this
workflow; do not train the redundant matched-full config as a seventh run.

## Current external-evaluation boundary

Kathbath is a reasonable single Bengali OOD corpus for the first submission
iteration, provided it was excluded from the SSL inventory. The current adapter
downloads its Bengali `valid-*` shards and the generic manifest builder makes a
seeded train/validation split; this is not an official benchmark split. The
current speaker-verification entry point also uses OpenSLR-53 all-pairs trials,
not a Kathbath enrollment/test protocol. The fixed-checkpoint statistics
launcher now reports repeated-probe aggregates and paired predefined-trial
intervals for this diagnostic protocol, but that does not turn it into an
official Kathbath result.

Do not label a Kathbath result as official-split speaker verification until the
source split/trial definition is recorded and a dedicated evaluator implements
it. The automatic six-checkpoint launcher is therefore suitable for shared
reconstruction metrics and whichever ASR/classification probe manifests are
configured, but it does not by itself clear the external-evaluation submission
gate.

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

The six-run evaluator now computes MR-STFT, waveform L1, SI-SDR, STOI, ESTOI,
and wide-band PESQ per utterance, retaining aggregate coverage and failure
examples for later paired bootstrap analysis. `pystoi` and `pesq` remain
in the locked `reconstruction-metrics` project extra selected by the launcher;
PESQ is enabled on Linux because its compiled extension is platform-sensitive.
If that extra cannot be materialized, the affected task fails explicitly and
later conditions are still attempted. All conditions save the same
deterministic source prefix; paired audio is saved only when the optional
listening bundle is enabled.

The evaluation STFT recipe is fixed across CLAE and Mimi rather than inherited
from each training config. Strict paper-table coverage requires one successful
value for every requested metric and source item; a partial aggregate is only a
diagnostic and is not emitted as a complete comparison result. The launcher
requires the normal validation JSONL manifest, and all models and baselines use
the same ordered source examples from it.

Compare full, reconstruction-only, representation-only, 25 Hz, and
no-decoder-corruption CLAE. The no-mHC run is useful but secondary for
reconstruction. The automated baseline uses the immutable Mimi revision
`89091b3e466eb6a9d11e537bf26b144f194978f7`, requests exactly eight quantizers,
verifies the code tensor actually contains eight codebooks, and labels that
result as the paper-style 1.1 kbps point. An all-available-quantizer Mimi result
may be reported separately as an upper-quality setting, never as the same
operating point.

If feasible, run a small blinded listening study with randomized samples and
report participant count, number of clips, rating question, confidence
intervals, and anonymized condition order.

The evaluator prepares randomized, anonymized stimuli and a hidden key after
all reconstructions only when `ABLATION_EVAL_LISTENING=1`. Human recruitment,
rating collection, exclusions, and confidence intervals remain manual study
steps and must not be described as automatic evaluation. Leave this option off
when no human listening result will appear in the submission.

### 4. Speaker and paralinguistic probes

- Evaluate speaker ID, speaker verification, emotion, and age with at least
  three deterministic probe seeds.
- Preserve speaker-disjoint folds for emotion and age.
- Replace all-pairs-only speaker verification with a documented enrollment/test
  trial protocol. Keep the old all-pairs result only as a secondary diagnostic.
- Report EER and minDCF with speaker- or predefined-trial-bootstrap 95%
  confidence intervals, matching the declared verification protocol.
- For classification, report macro-F1 plus a 95% interval; include accuracy only
  as a secondary metric when classes are imbalanced.

### 5. Representation diagnostics

Each condition now also receives SUBESCO t-SNE and UMAP panels colored by
UTMOSv2 source-audio MOS, plus attentive-statistics and Transformer
temporal emotion heads. These tasks require an explicit SUBESCO directory and
write inside the condition's own `step_25000/` output so results cannot
overwrite another checkpoint.

Only the UTMOSv2 package source is commit-pinned. Because its helper downloads
default weights from an unversioned upstream `main` URL, every plot records the
exact loaded state-dictionary SHA-256 and cache reuse requires that identity.
Use `UTMOSV2_CHECKPOINT` to select a preserved local checkpoint.

`scripts/run_tacl_statistics.sh` orchestrates the available fixed-checkpoint
reconstruction, attention-ASR, OpenSLR speaker, Common Voice age, and SUBESCO
emotion tasks when their inputs are configured. It does not implement the
missing official Kathbath split/trial protocols.

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

- Use probe seeds 0, 1, and 2 with one frozen checkpoint and one fixed data
  selection/split. Treat these as probe-initialization and minibatch-order
  repeats; do not change the evaluated utterances with the probe seed.
- Report the mean and sample standard deviation over probe seeds separately
  from the paired-bootstrap confidence interval. Bootstrap saved predictions;
  do not train another seed for each resample.
- Draw each bootstrap sample jointly across systems using the same IDs. Use
  utterances for reconstruction and ASR, speaker clusters for speaker and
  paralinguistic classification, and speakers or predefined trials for
  verification.
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
settings, a 25k run processes
  `25,000 * 42 * 4 * 3 / 3,600 = 3,500` source-audio exposure hours. Multiply by
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
