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
- `models/discriminator.py` — `MultiPeriodDiscriminator` (HiFi-GAN MPD) for the
  optional adversarial loss. Slim channel widths by default (~2.7M params) to fit
  6 GB; `disc_channels` is configurable. Built only when `loss.adv.enabled`.
- `losses.py` — `MultiResSTFTLoss` (multi-resolution STFT reconstruction:
  spectral convergence + log-magnitude) plus the HiFi-GAN GAN losses
  `discriminator_loss` / `generator_adv_loss` / `feature_matching_loss` (LSGAN).

## Training

- `train.py` — single entrypoint and training loop. Objective:
  `stft_weight·L_stft + jepa_weight·L_jepa + sigreg_weight·L_sig` (+ optional
  L1 wav term, + optional `adv_weight·L_adv + fm_weight·L_fm` when
  `loss.adv.enabled`). The GAN path adds a HiFi-GAN MPD discriminator with its
  own AdamW optimizer + GradScaler. The whole GAN path is skipped until
  `step >= adv_start_step` (fast pre-GAN phase); once active, per microbatch it
  runs a discriminator update (real vs detached fake) then adds the generator
  adversarial + feature-matching terms with D frozen (grad still flows to the
  decoder, but D grads aren't corrupted under grad accumulation). `L_fm` is
  additionally gated on `fm_start_step`. `L_jepa` is the V-JEPA global/local
  dense loss
  (`l_global` + `l_predict` + `context_weight·l_context`); `l_global ≡ 0` when
  `num_globals == 1`. `L_sig` is frame-level SIGReg on the projector output.
  Validation computes `val_stft` only; the val dataloader is built once at
  startup.
- Configs: `configs/exp0.yaml` (cloud, ~6M params, d_model 192),
  `configs/exp_3m.yaml` (~3.1M, staging data), `configs/exp_3m_gan.yaml`
  (small + fast GAN variant: ~2.78M generator + ~0.5M MPD, slight recon, JEPA up,
  adversarial/FM from step 20k; fp32, batch 10, 30k steps for the 6 GB card), and
  `configs/local_6gb.yaml` (tiny). All are valid full configs, not overrides.

## Config system

`schema.py` (pydantic, `extra="forbid"`) is the single source of truth.
`config.py`: `load_config(path)` parses YAML → `Config`;
`apply_overrides(cfg, ["a.b=c", ...])` applies dotted CLI overrides. Both
return the pydantic model; call sites use attribute access. `DatasetConfig`
in `data_loading.py` is a runtime object, intentionally not YAML-mirrored.

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

One self-contained file (no package) holding the whole data/artifact pipeline,
driven by `python scripts/housekeeping.py <cmd>` and the `Makefile`. The adapter
pattern lives inside it: per-source `DatasetAdapter` subclasses (openslr53,
common_voice_bn, bengaliai_speech, regspeech12, indicvoices, subak_ko,
shrutilipi, kathbath — each `download()` + `iter_records()`), registered in the
`REGISTRY` dict. The
`pack_to_dir` step writes 16 kHz mono FLAC under `audio/<dataset>/` plus four
JSONL manifests (train/val/asr_probe_{train,val}); `push_to_hub`/`fetch_dataset`
move the packed layout to/from HF Hub; `publish_checkpoint` uploads a trained
`last.pt` + model card. Subcommands: download, build, audit, push, fetch,
pack-and-push, publish-checkpoint. Credentials are read from the environment at
point of use (`os.environ["HF_TOKEN"]` etc.) —
there is no committed creds file. They live in the gitignored `.env` (template:
`.env.example`); on a fresh GPU VM, `./setup.sh` sources `.env` and runs deps →
fetch → train in one shot (`--no-train` to stop after the data fetch). Running
`make` targets standalone (e.g. on a prep instance) requires sourcing `.env`
first: `set -a && . ./.env && set +a`. Only `HF_TOKEN` is needed for training;
`KAGGLE_*` (regspeech12, bengaliai_speech) and `MDC_API_KEY` (common_voice_bn)
only for `make pack-and-push`/`download` on a prep instance. The
`bengaliai_speech` competition also requires accepting its rules once at
kaggle.com/competitions/bengaliai-speech.

## Data loading + augmentation

One module, `data_loading.py` (repo root, not a package):
- `AudioDataset` (JSONL manifests; relative `audio_filepath` resolved via
  `resolve_manifest_root`: manifest dir or one level up for the packed
  `<root>/manifests/` layout), `collate_fixed`, `DatasetConfig`.
- waveform augment (noise/lowpass/gain/clip) and frame/waveform chunk masking
  for JEPA local views (`apply_waveform_augment`, `make_frame_chunk_masks`,
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
