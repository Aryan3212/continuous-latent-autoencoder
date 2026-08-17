# Codebase map

Agent-facing reference for the current implementation. Treat the code and this
file as authoritative. `CHANGELOG.md` is a human-only historical record; agents
should not consult it for implementation decisions.

## Purpose

A continuous-latent speech autoencoder for 16 kHz audio:

`waveform → convolutional frontend → Conformer encoder → continuous z → waveform decoder`

Training combines reconstruction, JEPA view consistency, and configurable
Gaussianisation, with an optional spectrogram-domain GAN.

## Model

`models/autoencoder.py` owns the checkpointed components and preserves their
top-level state-dict prefixes:

- `frontend` — `ConvFrontend`, `(B, 1, T) → (B, C, T')`; strided Conv1d,
  GroupNorm, and GELU. Its stride product is samples per encoder frame.
- `encoder` — `Encoder`, `(B, C, T') → (B, D, T')`; macaron FFNs, rotary
  attention, and convolution. `encoder_type` selects Conformer or FastConformer;
  FastConformer adds squeeze-excitation to the convolution branch.
- `projector` — `Projector`, `(B, D, T') → (B, P, T')`; per-frame MLP and
  BatchNorm for JEPA/Gaussianisation. The decoder and probes consume `z`.
- `decoder` — `WaveformDecoder`, `(B, D, T') → (B, 1, T)`; interpolation,
  asymmetrically padded same-length Conv1d, and latent-conditioned FiLM blocks.

Supporting modules:

- `models/conformer.py` contains the shared encoder primitives.
- `models/mhc.py` optionally adds Manifold Hyper-Connections. The reference
  retains widened `(S, B, T, D)` streams; this encoder uses a project-specific
  uniform mean readout to return `(B, T, D)` while preserving average scale.
- `models/sigreg.py` and `models/visreg.py` are alternative parameter-free
  regularizers selected by `loss.reg_type`.
- `models/discriminator.py` is the optional spectrogram-domain MPD.

Encoder internals use `(B, T, D)`; model boundaries are channels-first.

## Training and losses

`losses.py` provides STFT/mel reconstruction, the discriminator spectrogram,
and adversarial/feature-matching losses. STFT and complex math run in FP32 even
under FP16/BF16 network autocast.

View construction:

- Globals use `aug.waveform_aug_global`; locals may override it with
  `aug.waveform_aug_local` and add a waveform-aligned span mask.
- Local frontend frames can receive an independent span mask and Gaussian noise
  before the encoder.
- Decoder inputs receive their configured independent span mask and Gaussian
  noise regardless of whether global, local, or all views are selected.
- `loss.recon_views` selects global-view, local-view, or all-view reconstruction
  against the clean waveform; when multiple views are selected, their losses are
  averaged. The GAN uses the first selected reconstruction view.

Conceptual generator objective:

`recon_weight·L_recon + jepa_weight·L_jepa + reg_weight·L_reg + adv_weight·λ·L_adv + fm_weight·L_fm`

Adversarial terms exist only when `loss.adv.enabled`; adaptive weighting can
derive `λ` from reconstruction/adversarial gradients at the decoder output.
Under DDP, regularizer inputs and metrics are global while reported loss remains
world-size invariant.

## Runtime invariants

`train.py` is a CUDA entrypoint supporting one process or NCCL DDP via
`torchrun`. It owns checkpointing, W&B/JSONL logging, accumulation, AMP, the
optional discriminator optimizer, and profiling.

- Frontend and decoder stride products must match; the schema validates this.
- cuDNN benchmarking is enabled for fixed-shape training throughput; this is
  not a deterministic or reproducible runtime mode.
- LR is closed-form warmup plus cosine. Resume uses the completed step and the
  current schedule; changed LR inputs produce a warning instead of replaying
  or restoring scheduler state.
- Checkpoints restore model, optimizer, AMP scaler, global step, and optional
  discriminator state. The step-based scheduler is reconstructed from global
  step and the current config; it has no separately restored state. Packed
  checkpoints also store the active `data_epoch`. Resume restarts at the
  beginning of that deterministic packed epoch (same whole-shard worker
  assignment, selection, shuffle, and crops), not at an exact in-epoch sample
  position. Older checkpoints without this field retain the legacy global-step
  seed fallback. There is no EMA state in this codebase.
- AMP-overflow-skipped updates still advance the attempted-step counter.
- Every training log interval writes the same reduced row to JSONL and W&B. It
  includes objective/VISReg decomposition; globally gathered FP32 covariance
  diagnostics for clean encoder latents and the all-view projector population;
  pre-clip total/per-module gradient mean and maxima; boundary-step per-module
  update norms; clipping, AMP-skip, and non-finite counters; mHC layer internals;
  input/decoder health; throughput, source-audio exposure, and CUDA memory.
  Existing rank, decoder, and loss keys remain aliases. Packed runs additionally
  record `data_epoch` and each global rank/worker's shard IDs, shard count,
  assigned samples, and equal selected-sample quota. Static runtime metadata
  records effective batch construction, representation-view count, GPU models,
  and the fixed collapse/isotropy thresholds.
- There is no in-loop validation. `train.eval_interval_steps`,
  `train.val_batches`, and `eval.enabled` are currently unused;
  `data.val_manifest` is metadata for external evaluation.

## Configuration

`schema.py` is the Pydantic source of truth and rejects unknown keys. `config.py`
loads YAML and dotted overrides. Closed choices use `Literal` fields.
`DatasetConfig` in `data_loading.py` is a runtime dataclass, not YAML schema.

Full configs: `exp0.yaml`, `exp_3m.yaml`, `exp_3m_gan.yaml`, `large_2kh.yaml`,
`large_2kh_packed.yaml`, `local_6gb.yaml`, and `local_13gb.yaml`.
`kaggle_3m_gan.yaml` inherits from `exp_3m_gan.yaml` with Kaggle-specific
overrides; `large_2kh_packed.yaml` inherits from `large_2kh.yaml` with only
packed-data loader overrides.
`large_2kh_packed_25k.yaml` is the packed base configuration with only the
training endpoint reduced to 25k, retaining the original 100k LR horizon.

The six-run TACL workflow uses `large_2kh_packed_25k.yaml` as its full-objective
reference plus reconstruction-only, representation-only, mHC-off, 25 Hz, and
decoder-corruption-off configs inherited from `large_2kh_packed.yaml`. The five
ablations retain their historical `large_2kh_ablation_*_50k.yaml` names and run
IDs for resume compatibility. Every run shares the packed TAR backend, stops at
25k, and retains the underlying base config's 100k LR horizon.

## Data

`data_loading.py` contains the datasets, fixed collator, waveform augmentation,
shared span-mask construction, and waveform/feature mask application. The
default `data.backend=files` path is the existing map-style JSONL loader: rows
require `audio_filepath`, paths resolve per manifest against its directory or
its parent for `<root>/manifests/`, audio is loaded/resampled with torchaudio,
mixed to mono, then cropped and padded to fixed length. `data.backend=tar`
uses `PackedTarDataset`: it validates producer format v1 descriptors without
loading `index.jsonl`, deterministically assigns whole uncompressed TAR shards
uniquely to global `(rank, worker)` consumers, streams adjacent FLAC/JSON pairs,
selects an equal full-batch quota with an exact TAR-payload-byte shuffle buffer
(plus at most one oversized selected member), decodes each PCM16 FLAC through
one libFLAC handle, and applies the same crop/pad semantics. Its shared epoch
counter is visible to spawned persistent workers.

`scripts/prepare_audio_shards.py` is the optional CPU-side producer for the
packed streaming backend. It treats an existing combined training JSONL as
the sole inventory, performs the current load → channel-mean → default
torchaudio resample sequence on each complete utterance, and writes mono 16 kHz
PCM16 FLAC members into uncompressed TAR shards. It records a versioned
`shard_manifest.json` plus `index.jsonl`, records actual encode/decode
quantization error, supports safe resume, and verifies finished archives.
Its read-only `audit` subcommand either samples index rows from a bounded
number of random shards or streams the complete corpus with `--all`, compares
the loader-equivalent decoded TAR waveform against the still-mounted original
source after identical mono/resample canonicalization, and requires a new
log-file path for progress and exact sample/shard mismatch reporting.
Finite canonical peaks outside PCM16's range are held in a reversible
per-sample storage scale (recorded as `amplitude_restore_gain` alongside
canonical/storage peaks); `PackedTarDataset` restores it before normal
training preprocessing, so no loudness normalization or training-distribution
change is introduced. Legacy shards omit those optional fields and use gain 1;
the default file backend is unaffected. Non-finite/corrupt/missing sources still
fail rather than being silently dropped. `--resume` is safe start-or-resume for
an empty or matching interrupted output only; it migrates the original v1
interrupted failure-on-peak state without redoing finalized shards. The optional
`data.backend=tar` training path consumes its `shard_manifest.json`;
`data.backend=files` remains unchanged.

`scripts/housekeeping.py make-manifests` is also the combined data-preparation
path. It uses bounded thread pools for dataset downloads, OpenSLR shards,
adapter record collection, and manifest writes. ZIP/TAR files are deleted only
after verified extraction. HF parquet adapters atomically cache extracted row
metadata in `extracted/.records.jsonl` before deleting their source shards;
that cache is the idempotence boundary for later manifest rebuilds.

Both housekeeping and `train.py` load the repo-local `.env` with
`override=False`, so shell/platform environment variables win. Worker count is
controlled by `--workers`, `HOUSEKEEPING_WORKERS`, or the Make variable of the
same name.

## Entrypoints

- `scripts/housekeeping.py` — download data, make manifests, and publish/fetch checkpoints.
- `scripts/prepare_audio_shards.py` — optional manifest-to-uncompressed-TAR
  canonical audio producer and standalone structural verifier; paired with
  `data.backend=tar` for optional training, not dataset discovery.
- `scripts/plot_ablation_diagnostics.py` — strict JSONL post-processor for the
  five matched representation conditions. It merges rows by step, uses the
  common available steps and every plotted metric, keeps encoder-latent and
  projector panels separate, and writes provenance metadata.
- `scripts/run_ablation_suite.sh` — the single six-run workflow. It serially
  completes the five 25k ablations and packed-base reference, skips intact 25k
  checkpoints, resumes an interrupted run from its newest intact checkpoint,
  and invokes the evaluator once after all training is complete.
- `scripts/run_ablation_evals.sh` — serial evaluator for the same six step-25k
  checkpoints. It runs `eval.run_all` per intact checkpoint, then pinned Mimi
  at exactly 8 quantizers / nominal 1.1 kbps, optionally prepares a blinded
  fixed-source listening bundle when `ABLATION_EVAL_LISTENING=1`, and
  consolidates raw summaries plus a compact Markdown comparison. Failures are
  isolated so later conditions still run. The
  reconstruction source is the explicit normal validation JSONL manifest, and
  its `audio_filepath` entries must be accessible during evaluation.
- `scripts/run_tacl_statistics.sh` — fixed-checkpoint TACL evidence launcher.
  It validates all six step-25k checkpoints before writing outputs, keeps the
  data/split seed fixed, repeats stochastic probes at seeds 0/1/2, reuses
  frozen ASR and temporal-emotion features across seeds, and publishes paired
  bootstrap statistics only after every configured evaluation succeeds.
- `scripts/run_full_evaluation_suite.sh` — resume-aware seven-checkpoint
  evaluation launcher for the six 25k conditions plus the packed 210k
  `last.pt`. Its compact default uses one fixed seed, bounded Common Voice,
  speaker, and ASR samples, and evaluates FP32 reconstruction, exact-8-
  quantizer Mimi reconstruction, ASR, generic emotion, age, gender, speaker
  ID, and verification. Temporal-emotion variants are opt-in through
  `FULL_EVAL_TASKS`. A successful task writes a neighbouring `.ok` marker, so
  interrupted or failed tasks retry without repeating completed work. Shared
  external baseline tables are evaluated once against the packed 210k run;
  their Mimi representation features remain continuous pre-quantization
  latents, separate from the bitrate-controlled reconstruction baseline.
- `eval/bootstrap_statistics.py` — result-only paired bootstrap analyzer. It
  aligns the same item IDs across systems, reports probe-seed mean/sample SD
  separately from sampling uncertainty, bootstraps utterances for
  reconstruction/ASR, speakers for classification, and speakers or predefined
  trials for verification, and never retrains a probe.
- `scripts/build_listening_study.py` — verifies that every condition saved the
  same originals, samples a deterministic shared set, copies anonymized
  stimuli and a ratings template into a public directory, and preserves the
  condition/source key separately. It prepares stimuli only; it does not
  collect or analyze human ratings.
- `scripts/summarize_ablation_evals.py` — result-only companion used by the
  ablation evaluator to combine per-condition summaries without rerunning them.
- `scripts/download_subesco.py` — materializes the processed
  `sajid73/SUBESCO-audio-dataset` Parquet release into local WAV files plus a
  label-preserving TSV at `datasets/SUBESCO/` for emotion evaluation.
- `Makefile` — setup, data preparation, training, and run cleanup.
- `eval/run_all.py` — reconstruction evaluation plus configured probes. Each
  checkpoint writes one self-contained `step_<N>/` directory and summary;
  `eval.enabled` and child probe flags are honored, missing manifests are
  reported as skips, and latent visualization is opt-in with `--visualize`.
  Reconstruction runs the CLAE forward pass and metrics in FP32; this avoids
  BF16 decoder-activation overflow on otherwise finite held-out inputs. It
  reports fixed-source per-item MR-STFT components, waveform L1,
  SI-SDR, STOI, ESTOI, and wide-band PESQ with coverage/failure records and
  optional paired audio. Report mode additionally runs both SUBESCO temporal
  emotion heads and a t-SNE/UMAP figure colored by UTMOSv2 MOS; an
  explicit SUBESCO root is required or these tasks are recorded as skipped.
  Reruns remove stale per-task artifacts, apply bounded timeouts, and return a
  failing status after writing the summary if a requested task is incomplete.
  Step-less legacy checkpoints require `--step`.
  CLAE and Mimi share the explicit evaluation STFT config from
  `eval/recon_metrics.py`; strict completeness requires every metric on every
  item. This runner does not orchestrate the standalone speaker, age,
  attention-ASR, or Kathbath evaluations.
- `eval/repr_bench.py` — shared frozen-feature adapter registry and versioned
  embedding cache. Supports CLAE, WavLM, Whisper-tiny, ECAPA, emotion2vec,
  Mimi, Higgs Audio V2, and XCodec2; codec adapters use continuous
  latent/quantizer-decoded vectors and never substitute discrete code IDs.
  emotion2vec is extracted through its official FunASR 50 Hz frame-feature API.
  The default comparison set is the smaller CLAE/random/WavLM/Mimi core; heavier
  adapters remain available through `--models`. The random CLAE control is
  deterministically seeded and checkpoint-identified in the cache; decoded
  audio is fingerprinted so changed clips invalidate embedding and UTMOS
  results. Multi-pool callers extract frame features once, then derive mean and
  mean+std embeddings. Remote adapter revisions are immutable Hub commits, and
  cache hits pair embeddings with the current dataset labels.
- `eval/eval_emotion.py`, `eval/eval_speaker_id.py`, `eval/eval_speaker_verif.py`,
  and `eval/eval_age.py` — downstream representation probes. Emotion, age,
  and Common Voice gender use speaker-disjoint group folds; closed-set speaker
  ID holds out utterances from the same enrolled speakers; speaker verification
  scores all utterance pairs. The demographic probe reads a selected `age` or
  `gender` column from local Common Voice Bengali `validated.tsv`.
  Probe outputs separate data selection, split, and probe seeds and retain
  item/speaker predictions for paired resampling. Verification stores compact
  predefined trial identities and scores in a compressed NPZ artifact.
- `eval/eval_asr_attn.py` — fixed-budget 2-layer Transformer-decoder ASR probe;
  it accepts the shared adapters and is the content metric for low-rate CLAE.
  CLAE and external adapters both use versioned, resumable per-utterance frame
  caches; checkpoint/config/adapter settings and source-audio identities are
  validated before reuse. CLAE cache identity hashes the fully resolved config,
  including inherited bases and overrides. Actual audio durations are verified
  before extraction so a cropped waveform is never paired with a full transcript.
  Feature extraction is batched for CLAE and reuses one frozen external adapter
  across train/dev. Training and greedy evaluation load/pad only the current
  `DataLoader` batch (`--num_workers 0` by default), keeping RAM proportional to
  batch size for every model. The head and replacement sampler use a recorded
  deterministic seed; head initialization is reset after extraction so cache
  hits and misses are identical.
- `eval/eval_emotion_temporal.py` / `eval/eval_emotion_transformer.py` —
  condition-safe attentive-statistics and Transformer temporal emotion probes;
  both accept explicit checkpoint, SUBESCO root, data/split/probe seeds, and
  output paths. Their validated frame cache excludes the probe seed so frozen
  extraction is shared across repeated probe runs.
- `eval/eval_repr_cluster.py` — condition-safe t-SNE/UMAP plots colored by
  UTMOSv2 predicted MOS of the source audio. UTMOSv2 package code is locked to
  commit `cc2700db57bb83ee13dc31ebe1b868c254e15d09`; the upstream default weight
  URL is not revisioned, so runtime metadata and cache identity include the
  exact loaded state-dictionary SHA-256, config, fold, and seed. An explicit
  local weight path can be supplied with `UTMOSV2_CHECKPOINT`.
- `eval/eval_repr_viz.py` / `eval/render_compact_scorecard.py` — PCA+UMAP
  attribute plots and Markdown scorecard aggregation.
- `eval/eval_mimi_recon.py` — pinned Mimi reconstruction baseline using the
  same fixed-source per-item metric pipeline as CLAE. It enforces exactly eight
  quantizers and fails rather than silently evaluating all codebooks.
- `eval/recon_metrics.py` — shared reconstruction metric and coverage layer.
  SI-SDR is internal; STOI/ESTOI and PESQ use the pinned
  `reconstruction-metrics` project extra (`pesq` is Linux-only) and expose
  dependency/per-item failures instead of hiding missing metrics.
- `scripts/reconstruct_audio.py` / `reconstruct_live.py` — reconstruction tools.
- `scripts/visualize_latents.py` — standalone latent visualization.

## Documentation roles

- `README.md` — human-facing setup and workflows.
- `SUPERVISOR_RESEARCH_REPORT.md` — concise supervisor-facing snapshot of the
  configured `large_2kh` architecture, objective, data scale, and evaluation plan.
- `SUPERVISOR_RESEARCH_REPORT_UPDATED.md` — updated supervisor-facing account
  of the 170k checkpoint, full downstream results, interpretation, and
  reproducibility caveats from the supplied experiment log.
- `CODEBASE.md` — current agent-facing map.
- `CHANGELOG.md` — human-only change history.
- `docs/ABLATION_LOGGING_REQUIREMENTS.md` — implemented logging and plotting contract
  for the matched ablations; its hardware smoke-run acceptance check remains to
  be performed in the target CUDA environment.
- `docs/EXTRA_EVALUATIONS.md` — matched-checkpoint evaluation and reporting checklist
  for the TACL evidence package.
- `docs/ABLATION_RUNS.md` — user-facing serial launch, resume, output-path, and
  multi-GPU instructions for the matched ablation suite.
