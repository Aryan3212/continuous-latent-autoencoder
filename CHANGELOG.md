# Issue log (repo-wide)

Date format: `YYYY-MM-DD`

## 2026-07-18

**Authenticated representation-model downloads**

- **`eval/repr_bench.py`**: representation adapters now load the repo-local
  `.env` and pass `HF_TOKEN` (or legacy lowercase `hf_token`) to Hugging Face
  model and processor downloads, without recording credentials in outputs.

## 2026-07-18

**Reproducible SUBESCO acquisition**

- **`scripts/download_subesco.py`**: added an idempotent downloader for the
  processed Hugging Face SUBESCO release. It materializes its Parquet audio
  rows as WAV clips under `datasets/SUBESCO/audio/` and writes the supplied
  transcription, speaker, gender, sentence, repetition, and emotion labels to
  `datasets/SUBESCO/metadata.tsv`.

## 2026-07-18

**Script workflow cleanup**

- Removed obsolete Kaggle wrapper/notebook-export evaluation scripts and the
  unused staging-upload helper. The supported checkpoint workflow is now the
  `housekeeping.py publish-checkpoint` / `fetch-checkpoint` CLI.
- Moved the Mimi reconstruction baseline from `scripts/` to `eval/` and
  ignored local checkpoint, listening-output, and separate model-checkout
  artifacts.

## 2026-07-18

**Compact frozen-representation benchmark for `large_2kh`**

- **`eval/repr_bench.py`**: introduced model metadata/adapters for the compact
  benchmark (CLAE controls, WavLM, Whisper-tiny, ECAPA, emotion2vec,
  USAD2-Small, Mimi, Higgs Audio V2, and XCodec2) and versioned embedding
  cache keys by model revision, feature layer, checkpoint identity, pooling,
  preprocessing, and utterance IDs.
- **`eval/eval_age.py`**, **`eval/eval_repr_viz.py`**, and
  **`eval/render_compact_scorecard.py`**: added a speaker-disjoint Common
  Voice Bengali age probe, PCA+UMAP plots coloured by speaker/emotion, and the
  compact Markdown results table.
- **`eval/eval_asr_attn.py`**: the fixed 2-layer Transformer decoder now
  accepts frozen `repr_bench` adapters, avoiding CTC's token-rate constraint
  for the 12.5 Hz CLAE representation.
- **Dependencies**: added SpeechBrain for ECAPA-TDNN.

## 2026-07-17

**Uniform decoder-latent corruption across reconstruction-view ablations**

- **`schema.py`**: `loss.recon_views` now accepts `global`, `local`, and `all`.
- **`train.py`**: selects the requested global, local, or all latent views for
  reconstruction, then applies `aug.decoder_input_mask` and
  `aug.decoder_input_noise` to every selected decoder input. The augmentations
  no longer depend on the selected view type.

## 2026-07-16 (b)

**Parallel, self-cleaning data preparation and automatic local env loading**

- **`scripts/housekeeping.py`**: added bounded parallel downloads (including
  OpenSLR shards), parallel adapter record collection, and atomic parallel
  train/val manifest publication via `--workers` / `HOUSEKEEPING_WORKERS`.
  Successful archive extraction now removes ZIP/TAR inputs. HF parquet sources
  publish an atomic extracted-record cache before deleting shards, preserving
  safe retries and fast subsequent manifest builds.
- **`train.py`, housekeeping, dependencies**: direct commands automatically
  load the gitignored repo `.env` without overriding already-exported platform
  values. Added `python-dotenv` as a direct dependency and documented HF, W&B,
  MDC, Kaggle, and worker settings in `.env.example`.
- **`Makefile`, `setup.sh`, docs**: exposed the worker count and made the setup
  path use the current combined data preparation flow before training.

## 2026-07-16 (a)

**Wire `aug.frame_mask` into the training path (P1)**

- **`schema.py`**: `FrameMaskCfg` (`mask_ratio`, default `enabled=False`) already
  existed but was never read by `train.py`.
- **`data_loading.py`**: `make_frame_masks` produces a `(B, T')` binary mask that
  zero-masks ENTIRE latent frames (Bernoulli per frame) on the post-frontend grid.
- **`train.py`**: now imports `make_frame_masks`, reads `cfg.aug.frame_mask`, and
  applies it to the LOCAL slices of `h0_cat` (the last `num_locals*B` rows) before
  the encoder — globals stay clean as the anchor. Enabling `frame_mask` now drops
  whole feature vectors from local-view context as documented.

## 2026-07-15 (f)

**Use CPU-only TorchCodec for dataset audio decoding**

- **`pyproject.toml`**: Torch and TorchAudio remain on the CUDA 12.8 index for
  training, while TorchCodec is sourced from PyTorch's CPU index. This avoids
  the CUDA TorchCodec wheel's NPP/NVRTC runtime requirement for an operation
  that only decodes audio on the CPU. Removed the unused `nvidia-npp-cu12`
  dependency; `nvidia-nvjpeg-cu12` remains for its separate GPU JPEG use.
- **`data_loading.py`**: removed the `ctypes.RTLD_GLOBAL` NPP pre-loader. Audio
  loading now relies on normal package resolution rather than application-level
  dynamic-linker manipulation.

## 2026-07-15 (e)

**IndicVoices: skip parquet re-parse on repeated `iter_records` calls**

- **`scripts/housekeeping.py`**: `IndicVoicesAdapter.iter_records` now writes a
  `.metadata.jsonl` sidecar alongside the extracted audio directory on first run.
  On subsequent calls it walks the extracted `.wav`/`.flac` files directly and
  reads text/speaker metadata from the sidecar, avoiding a full re-read of all
  parquet files (which previously happened on every `make-manifests` call).

## 2026-07-15 (d)

**FastConformer as a configurable drop-in alternative to the Conformer encoder**

- **`models/fastconformer.py`** (new): `FastConformerLayer` (= macaron Conformer
  block + Squeeze-and-Excitation gate after the conv module) and `SqueezeExcitation`.
  Reuses `ConformerLayer`'s `MultiHeadSelfAttentionRotary` / `FeedForward` /
  `ConvModule` (RoPE + SDPA attention unchanged). Same `(B,T,D)` in/out and
  constructor shape as `ConformerLayer` (plus `use_se`), so it is a drop-in.
- **`schema.py`**: `EncoderCfg` gains `encoder_type` (`"conformer"` default |
  `"fastconformer"`), `use_se` (default `true`, FastConformer-only), and
  `xscaling` (scale input embeddings by `sqrt(d_model)`, NeMo FastConformer).
- **`models/encoder.py`**: `_build_encoder_layer(cfg)` selects the block by
  `encoder_type`; the stack + MHC wrappers are type-agnostic. `xscaling` is
  applied once after `in_proj`. All `Encoder(...)` call sites are unaffected.
- **`configs/large_2kh.yaml`**: commented example showing the swap
  (`encoder_type: fastconformer` + `cnn_module_kernel: 9`, optionally
  `xscaling: true`). SE adds a few params; `use_se: false` keeps param count
  closest to the standard Conformer.

## 2026-07-15 (c)

**VISReg as a selectable alternative to SIGReg for the Gaussianisation loss**

- **`models/visreg.py`** (new): `VISReg` (Vector-ISotropic Gaussianisation,
  https://haiyuwu.github.io/visreg/). Paper-faithful implementation: per-batch
  center / scale / shape losses driving the projector output toward N(0, I). No
  learnable params; the random projection `W` is resampled each forward pass.
  Input is `(N, B, D)`.
- **`schema.py`**: `LossCfg` gains `reg_type` (`"sigreg"` default | `"visreg"`)
  and a `VISRegCfg` (`weight`, `num_projections`). Selecting a `reg_type` keeps
  the other block's config inert.
- **`train.py`**: the Gaussianisation loss is now built/swapped by `reg_type`.
  Both SIGReg and VISReg are gathered across ranks (the same `_gather_with_grad`
  + `×world_size` trick restores the single-GPU full-batch gradient). VISReg's
  global frame pool is fed as `(1, N_total, D)`. The active term is logged under
  `l_sig` (sigreg) or `l_vis` (visreg) so W&B / jsonl reflect whichever is used.
- **`configs/*.yaml`**: each loss section now declares `reg_type: sigreg` and a
  `visreg:` block (kaggle inherits from exp_3m_gan via `_base_`). Flip
  `loss.reg_type=visreg` (or set `loss.visreg.weight`) to switch.

## 2026-07-15 (b)

**GAN stability: precision boundary + topology-independent adversarial loss (keep LSGAN, no defensive asserts)**

- **`losses.py`**: adversarial `discriminator_loss` / `generator_adv_loss` are now
  **mean-normalized across discriminator branches** (per-period sub-discriminator)
  instead of summed, so the objective/gradient magnitude no longer depends on how
  many MPD periods exist (duplicating a scale can't silently 5x the loss). Added a
  `loss_type` arg (`"lsgan"` | `"hinge"`); LSGAN remains the default. Hinge is
  available but not used by default (per preference). `feature_matching_loss` was
  already normalized by total feature-map count.
- **`losses.py`**: STFT / complex path (`_stft_mag`, `MelLoss._mel_mag`,
  `ReconSpectrogram._spec_mag`) now casts input + window to **FP32** before
  `torch.stft`, regardless of the surrounding AMP autocast dtype. This is the
  precision boundary: the most dynamically ranged op stays FP32 (BF16/FP16 never
  reach `torch.stft`), while model params / optimizer states stay FP32 and only
  conv activations are autocast.
- **`schema.py`**: `RunCfg` gains `amp_dtype` (`"fp16"` default, `"bf16"`
  supported). Recommended `"bf16"` on Ampere+ (4090/A100): FP32-like exponent range
  means LSGAN's squared logits can't overflow to `inf`/NaN under AMP the way they
  can in FP16 — a root cause of the earlier GAN-NaN-then-everything-NaN failure.
- **`train.py`**: autocast now uses `dtype=amp_dtype`. GAN loss calls pass
  `acfg.loss_type`. No finiteness assertions added (per request) — the NaN is
  prevented at the source (precision + normalization) rather than detected.
- **`configs/exp_3m_gan.yaml`**: now runs AMP with `run.amp_dtype=bf16` (was
  `amp:false` fp32 for the GTX 1660). BF16's FP32-like exponent range stops the
  LSGAN squared-logit overflow that caused the earlier FP16-AMP GAN NaN. 1660
  users override `run.amp=false` for fp32.

## 2026-07-15

**GAN discriminator operates in the recon domain (mel/STFT), not waveform**

- **`models/discriminator.py`**: `MultiPeriodDiscriminator` / `DiscriminatorP`
  now take `in_channels` (default 1, backward-compatible) so the discriminator
  consumes the **mel or STFT magnitude spectrogram** — the same representation as
  the active reconstruction loss — instead of the raw waveform. Docstrings
  updated accordingly.
- **`losses.py`**: added `ReconSpectrogram` — waveform `(B,1,T)` → magnitude
  spectrogram `(B, F, T_frames)` in the active recon domain (mel via MelLoss's
  STFT→mel recipe, STFT via the first `fft_sizes` entry). Exposes `n_bins` to
  size the discriminator's `in_channels`.
- **`train.py`**: builds `disc_spec = ReconSpectrogram(...)` and passes
  `in_channels=disc_spec.n_bins` to the MPD. Both discriminator calls feed
  `disc_spec(wav_a)` / `disc_spec(x_hat)` (and the detached fake), so the
  adversarial signal pushes the decoder toward realistic mel/STFT features.
  Startup log now reports the discriminator's input domain + bin count.
- **`train.py`**: the discriminator/generator spectrograms are now computed ONCE
  per microbatch (`spec_real`/`spec_fake`) and reused for both the discriminator
  update and the generator/feature-matching block, instead of calling `disc_spec`
  twice per step. No wasted STFT/mel compute when the GAN is inactive
  (`step < adv_start`).

**Mel reconstruction loss + STFT-vs-mel ablation switch**

- **`schema.py`**: added `MelCfg` (mel-spectrogram loss params: n_mels, n_fft,
  hop/win, sample_rate, fmin/fmax, window, logmag_eps, sc/mag/logmag weights).
  `LossCfg` gains `recon_type` (`stft`|`mel`), `recon_weight` (renamed from
  `stft_weight`, now weights whichever recon loss is active), and
  `recon_log_start_step` (STFT log-mag metric only logged after this many steps,
  in both recon modes). `STFTCfg.sc_weight` default lowered to 0.1 so mag/logmag
  dominate for speech.
- **`losses.py`**: added `MelLoss` (mel-scaled magnitude / log-magnitude
  comparison, stats prefixed `mel_*`) — interchangeable with `MultiResSTFTLoss`
  in the training loop for the ablation. Mel loss is weighted toward mag/logmag
  by construction (SC off by default).
- **`train.py`**: `recon_fn` is selected by `recon_type`; STFT loss is always
  built separately so `stft_log` (and the full stft_* breakdown) is logged in
  BOTH recon modes but only after `recon_log_start_step`. Active recon scalar
  logged as `l_stft`/`l_mel`; mel mode additionally logs `mel_*` every step.
- **Configs**: all `loss` blocks renamed `stft_weight`→`recon_weight`, added
  `recon_type`/`recon_log_start_step` and a `mel` block, and set `sc_weight: 0.1`.
  `kaggle_3m_gan.yaml` inherits the updated loss via `_base_`.

## 2026-07-12

**Makefile/README simplification, checkpoint publish/fetch subcommands**

- **Makefile**: removed `fetch-data`, `build-data`, `pack-and-push`, `publish`,
  `evaluate`, `all` targets. Kept: `prepare`, `download-data`, `make-manifests`,
  `train`, `clean-runs`. `train` no longer depends on `fetch-data`;
  `MANIFEST_DIR` variable controls manifest path (default `staging/manifests/`).
- **`scripts/housekeeping.py`**: added `--data-root`/`--datasets` support to
  `make-manifests` subcommand (auto-discovers adapter raw dirs under DATA_ROOT).
  Restored `publish-checkpoint` and `fetch-checkpoint` subcommands so the
  Kaggle multi-session workflow can push/pull checkpoints to/from HF Hub.
- **README.md**: cut cloud-GPU setup, dataset preparation, packed-format, and
  cloud-flow sections. Kept folder guide (updated paths), quick start,
  running-things (eval commands), and architecture overview. Added the
  two-step manifest → train workflow, checkpoint resume, and publish/fetch
- documentation.

## 2026-06-22

**Adversarial + feature-matching losses (optional GAN path)**

- **New `loss.adv` config block** (`schema.py`, `AdvCfg`, default `enabled:
  false` so all existing configs/runs are byte-for-byte unchanged): adversarial
  + feature-matching weights, start steps (`adv_start_step`, `fm_start_step`),
  discriminator lr/betas, MPD `periods`, and slim `disc_channels`.
- **`models/discriminator.py`**: HiFi-GAN Multi-Period Discriminator. The
  textbook channel widths (32/128/512/1024) give a ~41M-param disc that OOMs the
  6 GB card next to the ~3M generator, so `disc_channels` defaults to a slim
  `[16,64,128,256]` (~2.7M).
- **`losses.py`**: `discriminator_loss` / `generator_adv_loss` /
  `feature_matching_loss` (LSGAN + L1 feature matching).
- **`train.py`**: two-optimizer GAN loop, gated on `loss.adv.enabled`. The whole
  GAN path (D build/update + generator adv/FM) is skipped until `step >=
  adv_start_step`, so the pre-GAN phase stays fast / low-VRAM. When active, per
  microbatch: D update on real vs detached fake, then generator adv/FM with D
  params frozen (grad flows through D to the decoder, but D grads aren't
  corrupted by the generator backward under grad accumulation). Second
  AdamW + GradScaler; discriminator/optimizer state added to checkpoints
  (backward-compatible — old checkpoints lack the keys). Logs `l_adv` / `l_fm`
  / `l_disc`. Also dropped the per-dataset `loss_stft/<ds>` / `loss_wav/<ds>`
  logging breakdown.
- **`configs/exp_3m_gan.yaml`**: small + fast GAN variant. Slight recon
  (STFT 1.0→0.1, wav_l1→0), JEPA weight 3→6, SIGReg unchanged (0.05),
  adversarial + feature matching BOTH from step 20000. Smaller model — encoder
  ~1.45M (n_layers 7→5, ff 384→320), decoder ~0.79M (channels 160→256, film
  64→128), ~0.5M MPD (`disc_channels [24,48,64,96]`); ~2.78M generator total.
  Fast 30k-step run. fp32 (`run.amp=false`), `batch_size: 10` / `grad_accum: 20`
  → peak ~4.5 GB (incl. desktop). Earlier batch sweep: 8→3.7 GB, 12→5.2 GB
  (tight), 16→OOM.

## 2026-06-17

**Simplification turn 5: two housekeeping bug/cruft fixes**

- **openslr53 re-download bug**: `download()` re-fetched and re-extracted all 16
  multi-GB part zips on every run (zips deleted after extract, so their absence
  was read as "not done"). Now writes a per-part `.part_<p>.done` marker after a
  successful extract and skips marked parts. Also dropped the silent
  try/except around download/extract — failures now raise (and leave no marker,
  so the next run retries just that part).
- **pack/audit double pass collapsed**: the separate parallel `audit_records`
  pre-pass (header-probe every file) followed by a serial transcode (full-read
  every file) is now one pass. `_transcode_one` reads each file once, filters on
  duration inline, and returns a `(record, status)` so dropped rows are tallied
  (`transcode_status_counts` in `build_meta.yaml`) instead of silently returned
  as `None`. `audit_records` is retained for the standalone `audit` subcommand.

**Simplification turn 4: generalize `CLAE`/`clae`, prune scripts + scaffolding**

- Renamed env vars: `CLAE_DATA_ROOT`→`DATA_ROOT`, `CLAE_HF_REPO`→`HF_DATASET_REPO`,
  `CLAE_CKPT_REPO`→`HF_MODEL_REPO` (Makefile, setup.sh, housekeeping.py, docs,
  .env.example). Generalized remaining `CLAE`/`clae` branding (Makefile/setup
  headers, `run-` run-name prefix, `pack_` tmp prefix, dataset-card title). Kept
  the actual Hub slugs `aryanrahman/clae-bengali{,-encoder}`.
- Deleted folders: `.static-analysis/`, `.plans/`, `docs/`, `tests/`.
- Deleted scripts: `check_rank.py` (not imported anywhere), `get_param_count.py`
  (folded into `train.py`, which now prints a per-block trainable-parameter
  breakdown at startup), `smoke_encoder_mhc.py`, `test_mhc.py`,
  `verify_experiment.py`. Kept `reconstruct_audio.py`, `visualize_latents.py`,
  `fill_durations.py`.
- README: dropped the Unit-tests / Smoke-tests / Static-analysis sections (all
  referenced deleted paths); folder guide + CODEBASE.md updated.

**Simplification turn 3: `data/` package → `data_loading.py`; data root → `datasets/`**

- Consolidated `data/dataset.py` + `data/augment.py` into a single root module
  `data_loading.py` (no package). Updated the 6 import sites (`train.py`,
  `eval/common.py`, `eval/eval_asr.py`, `eval/eval_recon.py`,
  `scripts/check_rank.py`, `scripts/visualize_latents.py`) to
  `from data_loading import ...`. pyproject: dropped `data*` from packages.find,
  added `py-modules = ["data_loading"]`.
- Default `CLAE_DATA_ROOT` is now `<repo>/datasets` (gitignored, created by
  `scripts/housekeeping.py` on demand) instead of `$HOME/data/clae`. Updated
  Makefile (`$(CURDIR)/datasets`), setup.sh (`$PWD/datasets`),
  `housekeeping._data_root` (`_REPO_ROOT/datasets`), `.env.example`, `.gitignore`,
  README, CODEBASE.md.
- Removed the stale `data/bengaliai_speech` symlink (left over from the
  swapped-out Kaggle dataset). The `data/bengaliai-speech/` dir + the root
  `bengaliai-speech.zip` are left in place for the user to delete.

**Simplification turn 2: `clae_data/` package collapsed into one script**

- Replaced the whole `clae_data/` package with a single self-contained file,
  `scripts/housekeeping.py`. The adapter pattern is preserved (one
  `DatasetAdapter` subclass per source + a `REGISTRY` dict, all in-file);
  schema, audit, pack, push, fetch, and publish-checkpoint are sections of the
  same file. Heavy deps stay lazily imported inside the functions that use them.
- Invocation changed: `python -m clae_data <cmd>` → `python scripts/housekeeping.py <cmd>`.
  Makefile targets (download-data, build-data, fetch-data, pack-and-push,
  publish) rewired. A `sys.path` bootstrap at the top lets it run by path from
  any cwd while still resolving `from utils.checkpoint import ...`.
- `publish-checkpoint` stays as a subcommand (model upload is housekeeping too).
- Docs updated: README "Adding a new dataset source" + folder guide, CODEBASE.md
  "Data prep + HF I/O" section + Scripts list, `configs/exp0.yaml` comment,
  `data/dataset.py` + `scripts/fill_durations.py` docstrings.

**Simplification turn 1: credentials are pure env vars**

- Deleted the credentials indirection: removed `clae_data/_creds.py`
  generation from `setup.sh`, the `_CREDS` dict + `_load_creds()` +
  `_ensure_kaggle_env()` shim in `cli.py`, and `clae_data/_creds.example.py`.
  All secrets are now read straight from the environment at point of use
  (`os.environ["HF_TOKEN"]`, `MDC_API_KEY`, `KAGGLE_*`) — a missing key is a
  hard `KeyError`, not a silent fallback to blanks. `setup.sh` sources `.env`;
  standalone `make` calls need `set -a && . ./.env && set +a` first.
- Repo IDs / data root: `cli.py` reads `CLAE_HF_REPO`, `CLAE_CKPT_REPO`,
  `CLAE_DATA_ROOT` from the env with literal defaults (these aren't secrets).
- Fixed latent bug: `MDC_API_KEY` (needed by `common_voice_bn`) was never
  written by `setup.sh`'s `_creds.py` heredoc nor listed in `.env.example`, so
  the Common Voice download crashed on import. Now in `.env.example`.
- Removed the dead `requires_credentials` adapter field — declared on every
  adapter but never read anywhere.
- `.gitignore`, `.env.example`, `README.md`, `CODEBASE.md` updated.

## 2026-06-16

**Data prep: Common Voice swap, HF subdir fixes, local pipeline make targets**

- Replaced the Kaggle `bengaliai_speech` adapter with `common_voice_bn`
  (Mozilla Common Voice Bengali via the `datacollective` SDK + `MDC_API_KEY`).
  CC0-licensed and needs no per-competition rules acceptance, which the Kaggle
  competition download kept 401-ing on. Registry, Makefile `DATASETS`, README,
  CODEBASE.md, and `_creds.example.py` updated; `kagglehub` dep dropped,
  `datacollective` added (`kaggle` kept for `regspeech12`).
- Fixed `shrutilipi` and `kathbath` adapters: their HF repos key the Bengali
  subset under `bengali/` (not `bn/`), so `allow_patterns` matched 0 files.
  Kathbath has no `test` split, so `valid-*` shards are used as the eval set.
- Added Makefile targets `download-data` (raw download), `build-data` (full
  local pipeline: download → transcode → manifests under `STAGING_DIR`, with a
  `LIMIT` knob for smoke tests).

## 2026-06-15

**Attention seq2seq ASR diagnostic probe (`eval/eval_asr_attn.py`)**

- New `eval/eval_asr_attn.py`: autoregressive Transformer decoder probe over
  the same frozen encoder features as `eval_asr.py`, but with **no CTC T>=L
  constraint**. Lets us attribute poor ASR CER to either (a) CTC's frame-rate
  alignment limit or (b) the encoder representation itself.
- Reuses `_filter_manifest_by_duration` and `_load_feats_and_text` from
  `eval_asr.py` verbatim for a fair feature comparison. Writes filtered
  manifests with an `.attn.` infix so they never clobber the CTC probe's
  outputs. Caches vocabulary to `<train_manifest>.charset_attn.json`
  (distinct from `.charset.json`).
- `AttnDecoderHead`: `Linear` in-proj + `SinusoidalPE` on both memory and
  target embeddings + `nn.TransformerDecoder` + output projection. Greedy
  autoregressive decode respects per-sample valid frame lengths via
  `memory_key_padding_mask`.
- Output JSON matches `eval_asr.py` schema where applicable; omits
  `upsample_factor`/`infeasible` keys (not relevant); adds `decoder` config
  block and `"ctc_free": true`.
- `CODEBASE.md` Evaluation section updated.

## 2026-06-12

**Fresh-VM bootstrap + packed-manifest path fix**

- **`setup.sh` (new)**: one-shot GPU-VM bootstrap — loads `.env` (template
  rewritten in `.env.example`), generates `clae_data/_creds.py`, then `make
  prepare` → `make fetch-data` → `make train`. Falls back to
  `WANDB_MODE=offline` when no W&B key is set; `--no-train` stops after the
  fetch. Only `HF_TOKEN` is required for training (Kaggle keys are
  prep-instance only).
- **Bug fix — packed-manifest path resolution**: `clae_data.pack` writes rows
  relative to the dataset root (`audio/<ds>/<id>.flac`) but manifests live in
  `<root>/manifests/`, while `AudioDataset` and `eval_asr` resolved relative
  paths against the manifest's own dir — so `make fetch-data && make train`
  failed on the first batch. New `resolve_manifest_root` in `data/dataset.py`
  probes the first relative row against both candidate roots and fails fast
  at construction; both consumers use it.
- **Repro pins**: dead `uv.lock` line removed from `.gitignore` (the lockfile
  is tracked; the rule would silently swallow it if ever untracked) and
  `.python-version` (3.11) added so `uv sync` picks the same interpreter
  everywhere.

## 2026-06-11

**Simplification pass: dead-code removal, config unification, doc consolidation**
(`simplification` branch). No model/loss math changed — external module
interfaces are unchanged.

- **Config unification**: deleted all mirror `@dataclass` configs (Frontend,
  Encoder, MHC, Decoder, Projector, SIGReg, MultiResSTFT, WaveAug,
  WaveChunkMask); modules now take the pydantic `*Cfg` classes from
  `utils/schema.py` directly — single source of truth. `DatasetConfig` in
  `data/dataset.py` kept (runtime object, not YAML-mirrored).
- **Encoder stack**: removed dead `key_padding_mask` plumbing and the
  Zipformer-era compat args (`pos_emb`/`chunk_size`/`attn_mask`); internal
  layout standardized to `(B, T, D)` (mHC residuals `(S, B, T, D)`). mHC kept
  for an upcoming on/off ablation.
- **`models/sigreg.py`**: removed DDP helpers (single-GPU project);
  `SIGReg.forward` now returns a bare scalar (the stats dict was always
  discarded).
- **`losses/multires_stft.py`**: removed unused `mask` / `target_mags` params.
- **`train.py`**: validation computes `val_stft` only (`val_sig` removed —
  not train-comparable, redundant with training-side gauges); val
  dataset/dataloader built once at startup; one-line guard noting
  `l_global ≡ 0` when `num_globals == 1`.
- **Removed the dead `use_latent` thread** end-to-end (`eval_asr`,
  `eval/common`, `run_probes`, `AsrCfg`, both configs).
- **`eval/run_probes.py`**: stale getattr-defensive block replaced with
  direct schema attribute access.
- **Deleted dead code**: `optim/` (empty package), `clae_data/wandb_setup.py`,
  `validate_record` in `clae_data/schema.py`, `sweep.yaml` + `sweep_fast.yaml`,
  `scripts/{memory_test,reconstruct_sample,count_params}.py`,
  `eval/extract_embeddings.py` + `iter_embeddings` in `eval/common.py`, the
  tracked `wandb/` dir and generated `.static-analysis` reports (configs kept).
  `.gitignore` additions: `wandb/`, `.cache_ggshield`, `.ruff_cache/`.
- **`reference-implementations/` slimmed**: full vendored repos (`vjepa2`,
  `RAE-main`, `le-wm`) moved out of the repo to
  `../reference-implementations-archive`; single-file refs +
  `*-REIMPLEMENTATION_NOTES` + a new README remain in-tree.
- **Doc consolidation**: deleted `COMMANDS.md` (folded into README's "Running
  things"), `SIMPLIFICATION_PLAN.md` (open decisions → `LAB_NOTEBOOK.md`),
  `DATASET_PIPELINE_PLAN.md` (key-purge procedure inlined into README's
  credentials notice; how-to already in README), and `HISTORICAL_CHANGES.md` +
  `RESEARCH_SUMMARY_2026_04_21.md` (superseded history, kept in git). Rewrote
  `CODEBASE.md` to match the current tree; surgical README fixes (folder guide,
  removed `calm_like_exp0.yaml` refs, corrected the "Notes" claim that removed
  experimental paths were still config toggles).
- **Probe / eval / logging cleanups**: merged the ~95%-identical
  `eval/eval_emotion.py` + `eval/eval_gender.py` into one
  `eval/eval_cls_probe.py` (required `--label_key`, `--hidden` arg, always
  reports accuracy + macro-F1; `run_probes.py` passes `--hidden 256/128` to
  preserve per-probe capacity). Root `Config` drops `extra="allow"` so
  root-level typos fail loudly (inherits `extra="forbid"`). Eval checkpoint
  loading is now filter-then-`strict=True` (`eval/common.py`,
  `eval/eval_recon.py`, `scripts/check_rank.py`, `scripts/visualize_latents.py`)
  — train checkpoints hold the full ModuleDict, so keys are filtered to the
  built submodules before a strict load. `.gitignore` `data/` narrowed to
  `data/manifests/` (the broad pattern was shadowing tracked `data/*.py`).
  `RotaryEmbedding._maybe_build` cache check now includes `dtype` (an
  autocast-fp16 cache was being served to later fp32 calls). `train.py`
  per-dataset log averaging switched to sum+count pairs so intermittent
  `loss_stft/<ds>`/`loss_wav/<ds>` keys report a true mean instead of being
  divided by the full microbatch/log-interval denominator.
- **UTF-8 pinned on text I/O**: `data/dataset.py` manifest reads,
  `utils/config.py` YAML loads, and `utils/checkpoint.py` run-metadata writes
  now pass `encoding="utf-8"` explicitly. Under a C/POSIX locale (WSL2/SSH)
  Python's default open() encoding is ASCII, which crashed a run at its first
  eval when the val manifest's Bengali text hit `_read_manifest`
  (UnicodeDecodeError on byte 0xe0). The configs' non-ASCII comment characters
  would have tripped the same way on config load.

## 2026-06-10

**Anti-collapse change set after the 8,280-step local_6gb run rank-collapsed**
(z_rank 3.62/64, z_rank_utt 0.54 — see `LAB_NOTEBOOK.md` for the diagnosis and
`docs/EXPERIMENT_PLAN_2026_06_10.md` for the next-run plan). Root cause:
SIGReg and JEPA both act on the 32-dim projector output, recon is weighted
30× below JEPA, and AdamW wd 0.01 deletes the unconstrained encoder dims —
"the projector absorbs the regularization". Edits land across several files in
parallel; exact details are authoritative in the code:

- **`train.py` / `utils/schema.py`**: two new SIGReg branches sharing the
  existing `SIGReg` instance — frame-level on encoder z
  (`loss.sigreg.z_weight`, logged as `l_sig_z`) and utterance-level on
  time-pooled projector output (`loss.sigreg.utt_weight`, logged as
  `l_sig_utt`). Eigenvalue clamp for the three `z_rank*` participation-ratio
  metrics (near-zero covariance let `eigvalsh` return negative eigenvalues,
  producing the impossible `z_rank_utt` 0.54).
- **`configs/local_6gb.yaml` (new) + `configs/exp0.yaml`**: projector
  `output_dim` raised to ≥ `d_model` (32→64 local, 64→192 exp0) with wider
  hidden; `weight_decay` 1e-2→1e-5; chunk-mask `target_ratio` 0.15→0.25;
  local config moves to 2.5 s segments at batch 64 (was 1.5 s × 96);
  `lowpass_min_freq` raised to 2700 Hz (stop training invariance to the
  timbre band gender/emotion probes need); gender/emotion probes enabled in
  `local_6gb.yaml`; ASR probe `steps` 1000→8000.
- **`eval/eval_asr.py`**: ×4 time-upsampling of 12.5 Hz features before the
  char-CTC head (12.5 Hz vs ~8–15 Bengali chars/sec made most samples
  CTC-infeasible; `zero_infinity=True` silently zeroed them — the probe
  measured its own handicap), duration filtering (start-crop vs
  full-transcript mismatch), infeasibility accounting, optional BiLSTM head.
- **`eval/common.py` + gender/emotion probes**: utterance-level probes
  hardened; pooled-embedding effective-rank gauge added to their output JSON.

Old checkpoints are incompatible (projector shape changed) — restart from
scratch. Note `docs/LOG_INTERPRETATION.md` healthy ranges were written for
the d_model-192 setup and the old `l_sig = 0.5×(utt+frm)` definition; they
are stale relative to this change set.

## 2026-02-01

- Created this file because repo instructions reference `agents.md` as the issue log, but it did not exist yet.
- Research-note audit: current codebase does not yet implement Zipformer+mHC encoder, LeJEPA Algorithm 2 wiring, LeJEPA SIGReg (Algorithm 1), RAE-inspired decoder mechanics, or GAN training; these gaps are now captured in `.plans/*` and `UNCERTAINTIES.md`.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).

## 2026-04-30

- Created `COMMANDS.md` as a quick reference for training, evaluation, and data preparation commands.
- Updated `CODEBASE.md` to include `COMMANDS.md` in core documentation.

## 2026-06-02

- **Param redistribution in `configs/exp0.yaml`** (Exp0 shrunk from ~14.6M → ~6.0M trainable). Targets encoder > decoder > projector > frontend, all four blocks within ~5% of (3M, 1.5M, 1M, 0.5M):
  - Frontend channels `[64,128,192,256,256]` → `[64,128,160,192,192]` (~724K → ~503K). Frontend output still 192, so encoder d_model matches.
  - Encoder `d_model` 256 → 192, `feedforward_dim` 768 → 576 (keeps 3× ratio). mHC config unchanged. (~5.1M → ~2.88M)
  - Decoder `channels` 512 → 320, `film_hidden` 128 → 64. Ups/ResBlock/dilation layout unchanged. (~3.95M → ~1.55M)
  - Projector `hidden_dim` 2048 → 896; `n_hidden_layers` and `output_dim` unchanged. (~4.86M → ~1.04M)
- **`scripts/get_param_count.py`**: now also counts and prints the projector line, so the full four-block breakdown is visible.
- No model-code changes — the four `*Config` dataclasses default to the new values only if the config is silent; `configs/exp0.yaml` sets them explicitly. Frontend stride product (1280 → 12.5 Hz) and mHC wiring unaffected.

## 2026-02-02

- Added `REUSE.md` to document vendored repos and clarify that code is currently referenced (not yet ported) into core modules.
- Noted a design mismatch for downstream evaluation: current emotion/gender probes use pooled embeddings, but the desired direction is sequence heads over frame-level tokens.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).
- Ported Zipformer ScaledAdam into `optim/scaled_adam.py` and added a parity test (`tests/test_scaled_adam_parity.py`).
- Rewired LeJEPA objective (center-matching) and SIGReg Algorithm 1, with tests (`tests/test_sigreg.py`).
- SIGReg now uses Epps–Pulley + sliced univariate test per vendored LeJEPA references.
- Added Eden/Eden2 LR schedulers and a CALM-like config preset (`configs/calm_like_exp0.yaml`).
- Added RAE-style latent normalization support in the decoder and a latent-stats script (`scripts/compute_latent_stats.py`).
- Added MPD/MSD discriminators and GAN losses, wired into `train.py` behind `gan.enabled`.
- Added spectral convergence term and configurability to multi-res STFT loss, with tests.
- Updated ASR probe to support end-to-end encoder features with a dry-run mode.
- Added reconstruction evaluation and a run-all benchmark entrypoint with optional baselines.

## 2026-04-17

-   **Stabilized STFT Spectral Convergence (SC) Loss**: Modified `MultiResSTFTLoss` to use the unmasked ground truth magnitude in the denominator when calculating SC on masked regions. This prevents the loss from exploding (previously reaching ~56) when the masked segment of the audio is quiet.
-   **Normalized Masked Losses**: Updated `train.py` to divide masked STFT and L1 losses by `mask_frac`, ensuring the loss scale remains consistent with unmasked validation reconstruction (~3-4).
-   **Fixed Primary Loss Stagnation**: Added a temperature scale (default 0.07-0.1) to the cosine similarity logits in `_primary_logits`. This sharpens the distribution, allowing the `l_primary` classification loss to decrease from its random-chance plateau (~0.69).
-   **Updated Config**: Added `loss.primary.temp` to `configs/exp0.yaml` for easier tuning of the primary component similarity task.

## 2026-05-22 (later)

**`simplification` branch — train.py audit pass (DDP/torch.compile prep)**

- **Removed GAN training path entirely** (`models/discriminators.py`,
  `scripts/smoke_gan_step.py`, `scripts/check_gan_params.py`,
  `gan:` block in `configs/exp0.yaml`, `GANCfg` in `utils/schema.py`,
  all GAN forward/backward/adaptive-weight logic in `train.py`). The two
  `torch.autograd.grad` calls were also DDP/compile-hostile. Recoverable from
  git history if revisited.
- **Removed inline CTC probe** (`eval/inline_probe.py`, `InlineProbeCfg`,
  `loss.inline_probe:` block, all call sites in `train.py`). Offline sanity
  check still available via `eval/eval_asr.py` (run as separate process).
- **Removed eval-on-save block** (subprocess probe orchestration, GPU shuffle
  to CPU and back, profiler pause/resume, `best_asr.pt`/`best_composite.pt`
  checkpoints, `run_eval_on_save` config, `--run_eval_on_save` CLI flag).
  Only `last.pt` is saved now.
- **Removed CodeCarbon emissions tracker** (`track_emissions` config,
  start/stop calls).
- **Simplified microbatch grad-accum**: dropped the list-of-microbatches
  pre-fetch pattern and the two-tier `mb_stats`/`accum_stats` accumulator.
  Now inline: pull → forward → backward → accumulate, single stats dict.
  Behaviour identical; ~80 fewer lines.
- **Diagnostics moved to log-step boundary**: three `torch.linalg.eigvalsh`
  calls + RMS detectors now compute only on the final microbatch of a log
  step, not every microbatch.
- **Dropped per-dataset loss breakdowns** (`meta["dataset"]` plumbing in the
  training step).
- **Tightened profiler block** into a single init + auto handler selection
  (no more eval-pause-resume since eval-on-save is gone).
- **ASR charset caching**: `eval/eval_asr.py` now writes/reads
  `<train_manifest>.charset.json` so the charset isn't rebuilt every run.
- **Deleted dead helpers**: `_pool_utt`, `_pool` (vulture-flagged).

Net: `train.py` 994 → ~470 lines. No model/loss math changed; only control
flow, removed features, and diagnostics placement.

## 2026-05-22 (dataset pipeline pass)

**`simplification` branch — `clae_data/` package replaces the prep scripts**

- **Introduced `clae_data/`** as a single unified surface for dataset prep.
  Replaces twelve scattered scripts (`prepare_*`, `create_*`,
  `audit_datasets`, `datasets_download`, `finalize_manifests`,
  `prepare_asr_manifest`). One CLI (`python -m clae_data`) with seven
  subcommands: `download`, `audit`, `build`, `push`, `fetch`,
  `publish-checkpoint`, `pack-and-push`.
- **Seven adapters** under `clae_data/adapters/`: `openslr53`,
  `bengaliai_speech`, `regspeech12`, `indicvoices`, `subak_ko`,
  `shrutilipi`, `kathbath`. Each is a `DatasetAdapter` subclass with
  `download()` + `iter_records()`. HF parquet handling is shared via
  `clae_data/adapters/_hf_parquet.py`. Adapters are registered in
  `clae_data/registry.py`.
- **`clae_data/pack.py`** is the one transform that turns adapter outputs
  into the canonical packed layout: 16 kHz mono FLAC files under
  `audio/<dataset>/`, plus four JSONL manifests (`train`, `val`,
  `asr_probe_train`, `asr_probe_val`) with paths relative to the staging
  root. Audit (drop <1s, >30s, unreadable) runs internally. Loudness norm
  is a TODO.
- **HF Hub as raw-file blob storage** (`upload_folder`), not parquet
  datasets. Rationale: lets the dataset grow incrementally — a new source
  is `upload_folder` more audio plus a versioned `train_v2.jsonl` — with
  no schema migrations. See `DATASET_PIPELINE_PLAN.md`.
- **`data/dataset.py`** now resolves relative `audio_filepath` paths
  against the manifest's parent directory, so the same JSONL works on a
  prep box and on a cloud GPU after `huggingface-cli download`.
  Standardized on the `audio_filepath` key; dropped `path` / `audio`
  fallbacks.
- **`clae_data/wandb_setup.py`** reads `WANDB_API_KEY` from
  `clae_data/_creds.py` when the env var is unset. Called from the top of
  `train.py`'s `main()` before `wandb.init`.
- **Credentials**: one gitignored file, `clae_data/_creds.py`.
  `_creds.example.py` is the committed template. Migration to env vars is
  documented in `.env.example` and in the README rotation banner.
- **`Makefile`** targets: `prepare`, `fetch-data`, `pack-and-push`,
  `train`, `evaluate`, `publish`, `all`. `make all` is the one-command
  cloud-GPU path (fetch -> train -> evaluate -> publish).
- **Deleted scripts** (all superseded by `clae_data/`):
  `scripts/datasets_download.py`, `scripts/prepare_openslr53.py`,
  `scripts/prepare_bengaliai.py`,
  `scripts/create_bengaliai_manifests.py`,
  `scripts/prepare_regspeech12.py`, `scripts/prepare_hf_parquet.py`,
  `scripts/prepare_remaining_datasets.py`, `scripts/audit_datasets.py`,
  `scripts/create_clean_manifest.py`,
  `scripts/create_dataset_splits.py`, `scripts/finalize_manifests.py`,
  `scripts/prepare_asr_manifest.py`, `scripts/create_sweep_subset.py`
  (stale 5%-subset generator for the old `combined_*.jsonl` layout), and
  `create_test_config.py` (root-level helper for the already-deleted
  `configs/exp0_test.yaml`).
- **README rewrite**: three new sections after the credentials banner —
  "One-command training on a cloud GPU", "Dataset preparation (one-time,
  on a prep instance)", and "Adding a new dataset source". The legacy
  "Data Preparation Workflow" subsection (which referenced the deleted
  scripts) was removed; the `uv` quick start now points at
  `$CLAE_DATA_ROOT/manifests/{train,val}.jsonl`. Removed a dangling smoke
  test reference to the long-deleted `scripts/smoke_gan_step.py`.
- **Stale comments** updated: `configs/exp0.yaml` data block and
  `clae_data/adapters/bengaliai_speech.py` no longer reference removed
  scripts.

## 2026-05-22

- **Encoder rewrite**: replaced Zipformer2 (`models/zipformer.py`, `models/zipformer_scaling.py`) with a clean Conformer implementation (`models/conformer.py`). `ConformerLayer` uses macaron-Conformer structure (FFN₁ → MHSA+RoPE → Conv → FFN₂ → LayerNorm) with `F.scaled_dot_product_attention` and rotary embeddings; compatible call signature preserves MHC wrapper plumbing in `models/encoder.py`. Old checkpoints are incompatible — restart from scratch.
- **Optimizer simplification**: ScaledAdam + Eden/Eden2 schedulers removed; replaced with AdamW + cosine schedule with linear warmup (`optim/lr_schedule.py`).
- **Pydantic config schema**: `utils/schema.py` added with `extra="forbid"` on all nested models; validated at startup in `utils/config.py`. `configs/exp0.yaml` is the canonical config.
- **Data pipeline**: WebDataset shard loading removed; JSONL manifest pipeline (`data/`) is now the only path. `scripts/pack_webdataset.py` deleted (orphan producer).
- **Projection head**: added `models/projector.py` MLP projection head applied before JEPA loss to stabilize training.
- **Loss rebalancing**: STFT/JEPA/SIGReg weights tuned; `loss.stft_weight` scalar added.
- **Dead code removed**: `models/zipformer.py`, `models/zipformer_scaling.py`, `optim/scaled_adam.py`, Eden scheduler, WebDataset helpers deleted. `configs/exp0_20pct_merged.yaml` and `configs/exp0_test.yaml` deleted (failed pydantic schema, legacy predictor block).
- Remaining deferred items: §3.8 cfg attribute-access migration, §3.3 MHC ablation, §3.6 eval entrypoints, §3.7 GAN path — see `SIMPLIFICATION_PLAN.md`.

## 2026-04-21

-   **Historical Commit Analysis & Documentation**: Conducted a comprehensive audit of all 23 repository commits. For each commit, analyzed changes, expected outcomes, and underlying research hypotheses.
-   **Created HISTORICAL_CHANGES.md**: Compiled the audit findings into a structured historical reference document.
-   **Git History Rewrite**: Systematically updated all commit messages and descriptions in the repository's history to reflect the refined technical understanding, outcomes, and hypotheses.
-   **Improved Repository Observability**: Standardized commit nomenclature (feat/fix/perf/refactor) and provided detailed context in the commit bodies to improve future maintainability.
