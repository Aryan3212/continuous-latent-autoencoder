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
- Local decoder inputs can receive a second independent span mask; decoder
  inputs for every view can receive Gaussian noise.
- `loss.recon_views` selects global-view reconstruction or decoding every view
  against the clean waveform and averaging their losses. The GAN uses global
  view 0 in either mode.

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
- Checkpoints restore model/optimizer/scaler and optional discriminator state,
  but not sampler position or Python/NumPy/Torch RNG state.
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

`data_loading.py` contains the dataset, fixed collator, waveform augmentation,
shared span-mask construction, and waveform/feature mask application. JSONL
rows require `audio_filepath`. Relative paths resolve per manifest against its
directory or its parent for `<root>/manifests/`; multi-manifest lists retain
separate roots. Audio is loaded/resampled with torchaudio, mixed to mono, then
cropped and padded to the configured fixed length.

## Entrypoints

- `scripts/housekeeping.py` — download data, make manifests, and publish/fetch checkpoints.
- `Makefile` — setup, data preparation, training, and run cleanup.
- `eval/run_all.py` — reconstruction evaluation plus configured probes.
- `eval/run_probes.py` — ASR/classification probes and latent visualization.
- Other `eval/eval_*.py` files — standalone diagnostics and representation benchmarks.
- `scripts/reconstruct_audio.py` / `reconstruct_live.py` — reconstruction tools.
- `scripts/visualize_latents.py` — standalone latent visualization.

## Documentation roles

- `README.md` — human-facing setup and workflows.
- `CODEBASE.md` — current agent-facing map.
- `CHANGELOG.md` — human-only change history.
