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
- Checkpoints restore model/optimizer/scaler, matching-config scheduler state
  when present, and optional discriminator state. Legacy checkpoints without
  serialized scheduler state retain the closed-form resume path; changed schedule
  inputs intentionally keep that current-config closed-form path. Future checkpoints record data epoch;
  packed resumes deliberately begin a fresh randomized packed epoch rather than
  attempting to restore a sample position. There is no EMA state in this codebase.
- AMP-overflow-skipped updates still advance the attempted-step counter.
- There is no in-loop validation. `train.eval_interval_steps`,
  `train.val_batches`, and `eval.enabled` are currently unused;
  `data.val_manifest` is metadata for external evaluation.

## Configuration

`schema.py` is the Pydantic source of truth and rejects unknown keys. `config.py`
loads YAML and dotted overrides. Closed choices use `Literal` fields.
`DatasetConfig` in `data_loading.py` is a runtime dataclass, not YAML schema.

Full configs: `exp0.yaml`, `exp_3m.yaml`, `exp_3m_gan.yaml`, `large_2kh.yaml`,
`local_6gb.yaml`, and `local_13gb.yaml`. `kaggle_3m_gan.yaml` inherits from
`exp_3m_gan.yaml` and contains Kaggle-specific overrides.

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
selects an equal full-batch quota with a byte-budgeted compressed shuffle
buffer (plus at most one oversized selected member), decodes PCM16 FLAC through libFLAC, and applies the same crop/pad
semantics. Its shared epoch counter is visible to spawned persistent workers.

`scripts/prepare_audio_shards.py` is an optional, CPU-side producer for a
future streaming data backend. It treats an existing combined training JSONL as
the sole inventory, performs the current load → channel-mean → default
torchaudio resample sequence on each complete utterance, and writes mono 16 kHz
PCM16 FLAC members into uncompressed TAR shards. It records a versioned
`shard_manifest.json` plus `index.jsonl`, records actual encode/decode
quantization error, supports safe resume, and verifies finished archives.
Finite canonical peaks outside PCM16's range are held in a reversible
per-sample storage scale (recorded as `amplitude_restore_gain` alongside
canonical/storage peaks); the planned packed loader restores it before normal
training preprocessing, so no loudness normalization or training-distribution
change is introduced. Non-finite/corrupt/missing sources still fail rather than
being silently dropped. `--resume` is safe start-or-resume for an empty or
matching interrupted output only; it migrates the original v1 interrupted
failure-on-peak state without redoing finalized shards. It does not alter the
current training loader or make `train.py` consume shards yet.

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
- `scripts/download_subesco.py` — materializes the processed
  `sajid73/SUBESCO-audio-dataset` Parquet release into local WAV files plus a
  label-preserving TSV at `datasets/SUBESCO/` for emotion evaluation.
- `Makefile` — setup, data preparation, training, and run cleanup.
- `eval/run_all.py` — reconstruction evaluation plus configured probes.
- `eval/repr_bench.py` — shared frozen-feature adapter registry and versioned
  embedding cache. Supports CLAE, WavLM, Whisper-tiny, ECAPA, emotion2vec,
  Mimi, Higgs Audio V2, and XCodec2; codec adapters use continuous
  latent/quantizer-decoded vectors and never substitute discrete code IDs.
  emotion2vec is extracted through its official FunASR 50 Hz frame-feature API.
- `eval/eval_emotion.py`, `eval/eval_speaker_id.py`, `eval/eval_speaker_verif.py`,
  and `eval/eval_age.py` — speaker-disjoint downstream probes. The age probe
  reads local Common Voice Bengali `validated.tsv` metadata.
- `eval/eval_asr_attn.py` — fixed-budget 2-layer Transformer-decoder ASR probe;
  it accepts the shared adapters and is the content metric for low-rate CLAE.
- `eval/eval_repr_viz.py` / `eval/render_compact_scorecard.py` — PCA+UMAP
  attribute plots and Markdown scorecard aggregation.
- `eval/eval_mimi_recon.py` — standalone Mimi reconstruction baseline using the
  same loss family as CLAE reconstruction evaluation.
- `scripts/reconstruct_audio.py` / `reconstruct_live.py` — reconstruction tools.
- `scripts/visualize_latents.py` — standalone latent visualization.

## Documentation roles

- `README.md` — human-facing setup and workflows.
- `SUPERVISOR_RESEARCH_REPORT.md` — concise supervisor-facing snapshot of the
  configured `large_2kh` architecture, objective, data scale, and evaluation plan.
- `CODEBASE.md` — current agent-facing map.
- `CHANGELOG.md` — human-only change history.
