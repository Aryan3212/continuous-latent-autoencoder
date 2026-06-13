# Issue log (repo-wide)

Date format: `YYYY-MM-DD`

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

