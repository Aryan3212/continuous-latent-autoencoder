# Issue log (repo-wide)

Date format: `YYYY-MM-DD`

## 2026-02-01

- Created this file because repo instructions reference `agents.md` as the issue log, but it did not exist yet.
- Research-note audit: current codebase does not yet implement Zipformer+mHC encoder, LeJEPA Algorithm 2 wiring, LeJEPA SIGReg (Algorithm 1), RAE-inspired decoder mechanics, or GAN training; these gaps are now captured in `.plans/*` and `UNCERTAINTIES.md`.
- Ported Zipformer2 encoder layers + scaling into `models/zipformer.py`/`models/zipformer_scaling.py`, integrated mHC streams into `models/encoder.py`, and added a smoke test (`scripts/smoke_encoder_mhc.py`).

## 2026-04-30

- Created `COMMANDS.md` as a quick reference for training, evaluation, and data preparation commands.
- Updated `CODEBASE.md` to include `COMMANDS.md` in core documentation.

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

