# Codebase map (quick reference)

A deterministic continuous-latent speech autoencoder. Waveform in (16 kHz) →
strided conv frontend → Conformer encoder → continuous latents `z` → conv/FiLM
decoder back to waveform, trained jointly with reconstruction + JEPA + SIGReg.

## Core pipeline

All modules take their pydantic `*Cfg` from `schema.py` directly (no
mirror dataclasses). Internal tensor layout is `(B, T, D)`; external module
interfaces use channels-first `(B, D, T)` as noted.

- `models/frontend_conv.py` — `ConvFrontend`: strided Conv1D stack with
  GroupNorm+GELU. `(B, 1, T)` → `(B, C, T')`. Stride product 1280 → ~12.5 Hz
  tokens at 16 kHz.
- `models/encoder.py` + `models/conformer.py` — `Encoder`: 1×1 in-proj then a
  stack of `ConformerLayer`s (macaron FFN → MHSA+RoPE → conv → FFN → LN,
  `F.scaled_dot_product_attention`). `(B, C, T')` → `(B, D, T')`.
- `models/mhc.py` — `MHCWrapper` + `sinkhorn_log`: Manifold Hyper-Connections,
  applied on selected encoder layers (residual streams `(S, B, T, D)`). Kept
  for an upcoming on/off ablation (see `LAB_NOTEBOOK.md`).
- `models/projector.py` — `Projector`: per-frame MLP + BatchNorm1d head.
  `(B, D, T')` → `(B, P, T')`. Decouples loss-space (JEPA + SIGReg act here)
  from representation-space (`z`, which the decoder and probes consume).
- `models/decoder_generator.py` — `WaveformDecoder`: ConvTranspose upsample
  stack with FiLM ResBlocks conditioned on the latent. `(B, D, T')` → `(B, 1, T)`.
- `models/sigreg.py` — `SIGReg` (Epps–Pulley sliced univariate Gaussianity
  test). `forward` returns a bare scalar. Single-GPU only.
- `models/visreg.py` — `VISReg` (Vector-ISotropic Gaussianisation,
  https://haiyuwu.github.io/visreg/). Per-batch center/scale/shape losses driving
  the projector output to N(0, I). `forward` takes `(N, B, D)` and returns a
  scalar. Param-free, random projection resampled each call. Selectable in place
  of SIGReg via `loss.reg_type` (`"sigreg"` default | `"visreg"`).
- `models/discriminator.py` — `MultiPeriodDiscriminator` (HiFi-GAN MPD) for the
  optional adversarial loss. Operates on the **reconstruction-domain spectrogram**
  (mel or STFT magnitude, matching `loss.recon_type`) rather than the raw
  waveform — `in_channels` is set to the spectrogram's bin count. Slim channel
  widths by default (~2.7M params) to fit 6 GB; `disc_channels` is configurable.
  Built only when `loss.adv.enabled`.
- `losses.py` — `MultiResSTFTLoss` (multi-resolution STFT reconstruction:
  spectral convergence + magnitude + log-magnitude) and `MelLoss` (mel-spectrogram
  reconstruction, interchangeable via `loss.recon_type` for the STFT-vs-mel
  ablation) plus the HiFi-GAN GAN losses
  `discriminator_loss` / `generator_adv_loss` / `feature_matching_loss`. The
  adversarial losses are **mean-normalized across discriminator branches** (so the
  objective is topology-independent) and support `loss_type` `"lsgan"` (default)
  or `"hinge"`. `feature_matching_loss` is normalized by total feature-map count.
  All STFT / complex math (`_stft_mag`, `MelLoss._mel_mag`, `ReconSpectrogram`)
  casts input + window to **FP32** regardless of autocast, so the most
  dynamically ranged op stays FP32. The discriminator consumes the same mel/STFT
  spectrogram as `L_recon` (built via `ReconSpectrogram`), so the GAN adversarially
  refines the spectrogram domain, not the raw waveform.

## Training

- `train.py` — single entrypoint and training loop. Objective:
  `recon_weight·L_recon + jepa_weight·L_jepa + reg_weight·L_reg` (+ optional
  `adv_weight·L_adv + fm_weight·L_fm` when `loss.adv.enabled`), where `L_reg` is
  `L_sig` (SIGReg) or `L_vis` (VISReg) chosen by `loss.reg_type`, and `L_recon`
  is `MultiResSTFTLoss` or `MelLoss` selected by `loss.recon_type` (the two are
  ablated against each other).   The STFT log-mag metric is logged in BOTH recon
  modes but only after `loss.recon_log_start_step`. The GAN path adds a HiFi-GAN
  MPD discriminator with its
  own AdamW optimizer + GradScaler. The whole GAN path is skipped until
  `step >= adv_start_step` (fast pre-GAN phase); once active, per microbatch it
  runs a discriminator update (real vs detached fake) then adds the generator
  adversarial + feature-matching terms with D frozen (grad still flows to the
  decoder, but D grads aren't corrupted under grad accumulation). `L_fm` is
  additionally gated on `fm_start_step`. The adversarial objective is selected by
  `loss.adv.loss_type` (`lsgan` default, `hinge` available); it is
  mean-normalized across MPD branches.   `L_jepa` = MSE(globals, center) + MSE(locals, center)
  (uniform 1:1, no context weight). `L_reg` is frame-level Gaussianisation on the
  projector output — SIGReg (`L_sig`) or VISReg (`L_vis`), chosen by `loss.reg_type`.
  JEPA local views are now heavily augmented and masked at the `h0` level
  (pre-encoder) with an independent second token mask at the decoder, and the
  reconstruction loss is averaged over all V views (each noised/masked) against
  the clean `wav_a`; globals retain light augmentation so they stay a clean
  anchor, and the projector/JEPA path itself consumes all views unchanged.
  Validation computes `val_stft` only; the val dataloader is built once at
  startup.
- Configs: `configs/exp0.yaml` (cloud, ~6M params, d_model 192),
  `configs/exp_3m.yaml` (~3.1M, staging data), `configs/exp_3m_gan.yaml`
  (small + fast GAN variant: ~2.8M generator + ~0.5M MPD, slight recon, JEPA up,
  adversarial/FM from `adv_start_step`; AMP **bf16** — `run.amp=true`,
  `run.amp_dtype=bf16` — to avoid the LSGAN FP16-overflow NaN on Ampere+ like the
  4090; override `run.amp=false` for fp32 on the 6 GB GTX 1660), and
  `configs/local_6gb.yaml` (tiny). All are valid full configs, not overrides.

## Config system

`schema.py` (pydantic, `extra="forbid"`) is the single source of truth.
`config.py`: `load_config(path)` parses YAML → `Config`;
`apply_overrides(cfg, ["a.b=c", ...])` applies dotted CLI overrides. Both
return the pydantic model; call sites use attribute access. `DatasetConfig`
in `data_loading.py` is a runtime object, intentionally not YAML-mirrored.

`RunCfg.amp_dtype` selects the AMP autocast precision (`fp16` default, `bf16`
recommended on Ampere+ like the 4090 — FP32-like exponent range so LSGAN's
squared logits can't overflow to NaN under AMP). The STFT / complex path is
always forced to FP32 regardless of `amp_dtype` (precision boundary in `losses.py`).

## Evaluation

Frozen-encoder probes + reconstruction metrics.

- `eval/eval_asr.py` — CTC ASR probe (small head over frame features), WER.
- `eval/eval_asr_attn.py` — attention seq2seq ASR probe (autoregressive decoder, no CTC T>=L constraint); diagnostic vs eval_asr.py to separate CTC frame-rate limits from representation quality.
- `eval/eval_cls_probe.py` — pooled-embedding MLP probe (`--label_key` selects emotion/gender).
- `eval/eval_recon.py` — waveform reconstruction metrics (STFT etc.).
- `eval/common.py` — shared frame-feature extraction.
- `eval/run_probes.py` — orchestrates the three probes (direct schema access).
- `eval/run_all.py` — CLI: recon + all enabled probes in one call.

## Data prep + HF I/O (`scripts/housekeeping.py`)

One self-contained file (no package) holding the data pipeline + checkpoint
artifact CLI, driven by `python scripts/housekeeping.py <cmd>` and the
`Makefile`. The adapter pattern lives inside it: per-source `DatasetAdapter`
subclasses (openslr53, common_voice_bn, bengaliai_speech, regspeech12,
indicvoices, subak_ko, shrutilipi, kathbath — each `download()` +
`iter_records()`), registered in the `REGISTRY` dict.

Subcommands:
- `download` — fetch raw archives from HF/Kaggle/OpenSLR into `DATA_ROOT`.
- `make-manifests` — iterate adapter records, split train/val, write JSONL
  manifests. Accepts `--map NAME=PATH` (Kaggle) or `--data-root` + `--datasets`
  (local workflow). On first run, parquet-based adapters (indicvoices, subak_ko,
  shrutilipi, kathbath) extract inline audio bytes to an `extracted/` subdir;
  subsequent runs of indicvoices walk `extracted/` + a `.metadata.jsonl` sidecar
  instead of re-parsing the parquet files.
- `publish-checkpoint` — upload `last.pt` + model card + config to HF model repo.
- `fetch-checkpoint` — download the latest published checkpoint from HF (for
  multi-session resume).

Credentials are read from the environment at point of use (`os.environ["HF_TOKEN"]`
etc.) — there is no committed creds file. They live in the gitignored `.env`
(template: `.env.example`). Only `HF_TOKEN` is needed for training; `KAGGLE_*`
(regspeech12, bengaliai_speech) and `MDC_API_KEY` (common_voice_bn) only for
`make download-data` on a prep instance. The `bengaliai_speech` competition also
requires accepting its rules once at kaggle.com/competitions/bengaliai-speech.

## Data loading + augmentation

One module, `data_loading.py` (repo root, not a package):
- `AudioDataset` (JSONL manifests; relative `audio_filepath` resolved via
  `resolve_manifest_root`: manifest dir or one level up for the packed
  `<root>/manifests/` layout), `collate_fixed`, `DatasetConfig`.
  Audio loading uses the CPU-only TorchCodec wheel (via `torchaudio.load`), so
  CUDA training does not impose a CUDA NPP runtime dependency on dataloader
  workers.
- augmentation for JEPA views, now asymmetric: globals get light waveform
  augmentation only, while locals get heavy waveform augmentation plus a
  token-level frame mask applied to the frontend output `h0` (before the
  encoder, so masked frames are zeroed and never seen) and, at decode time, a
  second independent random token mask plus light Gaussian noise on `z`. All
  V=num_globals+num_locals views are decoded and averaged for reconstruction
  against the clean target; the projector/JEPA path uses all views unchanged.
  New helpers: `apply_token_chunk_mask_h0`, `apply_token_chunk_mask_z` (plus the
  prior `apply_waveform_augment`, `make_frame_chunk_masks`,
  `apply_waveform_chunk_mask`).

Fetched/built datasets live under `$DATA_ROOT` (default the gitignored
`datasets/` at the repo root), never in the repo source tree.

## Scripts (`scripts/`)

- `housekeeping.py` — the data/artifact CLI (see "Data prep + HF I/O" above).
- `reconstruct_audio.py` — encode/decode audio files through a checkpoint;
  writes `_orig`/`_recon` WAV pairs + per-file STFT/L1 numbers.
- `visualize_latents.py` — PCA/UMAP latent-space plot.
- `fill_durations.py` — backfill `duration` into manifests that have `null`.

(`train.py` prints a per-block trainable-parameter breakdown at startup.)

## Reference implementations (`reference-implementations/`)

Slim, in-tree: single-file refs (`lejepa_*`, `mhc_*`, `zipformer2_*`) +
`*-REIMPLEMENTATION_NOTES.md` + a README. Full vendored upstream repos
(`vjepa2`, `RAE-main`, `le-wm`) were moved out of the repo to
`../reference-implementations-archive`.

## Other docs

- `README.md` — setup, running things, cloud one-command flow, credentials.
- `CHANGELOG.md` — dated log of major changes.
- `LAB_NOTEBOOK.md` — experiment log + open research decisions.
