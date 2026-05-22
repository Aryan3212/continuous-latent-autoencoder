# Dataset Pipeline Unification — Plan

Goal: collapse the ~10 scattered `scripts/prepare_*`, `create_*`, `audit_*`,
`finalize_*`, `datasets_download.py` into one cohesive pipeline that runs
end-to-end on a fresh cloud-GPU instance:

```
clone repo → make all  →  data downloaded → preprocessed → pushed to HF Hub
                          → training runs → wandb logs → checkpoint pushed to HF Hub
```

ASR-labelled samples retain their `text` field through every stage so the
eval probe in `eval/eval_asr.py` can still consume them.

---

## §0 — Credentials (user choice: keep hardcoded for now)

User decision (2026-05-22): the HF token + Kaggle key in
`scripts/datasets_download.py` stay hardcoded for now since this is a
small/personal research repo. **Add a `SECURITY.md` note** (or top-of-README
banner) reminding the maintainer to:

1. Rotate the HF token `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` at
   https://huggingface.co/settings/tokens before making the repo public or
   sharing it.
2. Rotate the Kaggle key `KGAT_38471085ebbafd3d0c544e1954296b39` at
   https://www.kaggle.com/settings.
3. If the repo ever goes public, also purge them from git history with
   `git filter-repo --replace-text` (the keys remain in git history even
   after deletion from `HEAD`).

The downloader script and training script run on *separate instances*:
- **Prep instance** (one-shot): runs `datasets_download.py` + adapters +
  pushes to HF Hub. Has the keys.
- **Train instance** (recurring, possibly ephemeral cloud GPU): pulls the
  prepared HF dataset, never touches Kaggle, never sees the raw archives.
  Only needs `HF_TOKEN` (for download) and `WANDB_API_KEY`.

---

## §1 — Current state inventory

### Dataset adapters (scattered)

| File | Source | Format | Has ASR text? |
|------|--------|--------|---------------|
| `scripts/datasets_download.py` | mixed (HF + Kaggle + OpenSLR) | downloads raw archives | n/a |
| `scripts/prepare_openslr53.py` | OpenSLR-53 (Bengali) | TSV → JSONL | yes |
| `scripts/prepare_bengaliai.py` | Kaggle bengaliai-speech | CSV → JSONL | yes |
| `scripts/create_bengaliai_manifests.py` | same as above (duplicate) | CSV → JSONL | yes |
| `scripts/prepare_regspeech12.py` | Kaggle regspeech12 | Excel → JSONL | yes |
| `scripts/prepare_hf_parquet.py` | generic HF parquet (IndicVoices etc.) | Parquet → JSONL | yes |
| `scripts/prepare_remaining_datasets.py` | orchestrator | calls others | n/a |
| `scripts/audit_datasets.py` | any JSONL | reads sf.info | n/a |
| `scripts/create_clean_manifest.py` | post-audit filter | JSONL → JSONL | n/a |
| `scripts/create_dataset_splits.py` | train/val split | JSONL → JSONL × 2 | n/a |
| `scripts/finalize_manifests.py` | hand-coded combiner | JSONL → JSONL | n/a |
| `scripts/prepare_asr_manifest.py` | ASR-probe subset extractor | CSV → JSONL × 2 | yes |

### Current JSONL schema (effective lingua franca)

```jsonc
{
  "audio_filepath": "/abs/or/rel/path.flac",   // required
  "text": "...",                                // optional — ASR label
  "duration": 5.0,                              // optional — sometimes placeholder
  "id": "utt_id",                               // optional
  "speaker_id": "spk_id",                       // optional
  "dataset": "openslr53"                        // optional, sometimes missing
}
```

`data/dataset.py:73-87` reads `audio_filepath | path | audio` (three fallback
keys) — a sign the upstream writers aren't consistent. Standardize on
`audio_filepath`.

### Downloaded-to paths (currently hardcoded, OS-specific)

- `datasets_download.py` writes to `C:\Bengali_Speech_Data\…` (Windows path
  on a Linux training box → broken).
- `prepare_remaining_datasets.py` reads from `data/Bengali_Speech_Data/…`
  (relative).
- `create_bengaliai_manifests.py` reads from `data/bengaliai-speech/…`
  (different path).

These three need to converge on a single root resolved from env / config.

---

## §2 — Target architecture

```
clae_data/                        # new package, replaces most of scripts/
  __init__.py
  schema.py                       # Record TypedDict, validation
  registry.py                     # name → adapter class
  cli.py                          # `python -m clae_data ...` entry
  adapters/
    __init__.py
    base.py                       # DatasetAdapter ABC
    openslr53.py                  # download + iter_records
    bengaliai_speech.py           # Kaggle competition
    regspeech12.py                # Kaggle dataset
    indicvoices.py                # HF parquet
    subak_ko.py                   # HF
    shrutilipi.py                 # HF
    kathbath.py                   # HF (probe-only)
  audit.py                        # parallelised sf.info check (moved from scripts/)
  pack.py                         # records → HF datasets.Dataset, splits, push_to_hub
  publish_checkpoint.py           # ckpt → HF Hub
scripts/                          # keep only project-local scripts that aren't dataset prep
  reconstruct_sample.py
  smoke_encoder_mhc.py
  verify_experiment.py
  visualize_latents.py
  check_rank.py
  count_params.py
```

### `clae_data.schema.Record`

```python
class Record(TypedDict, total=False):
    audio_filepath: str            # required; absolute after resolution
    text: Optional[str]            # null if no ASR label
    duration: Optional[float]      # filled in by audit step
    sample_rate: Optional[int]
    dataset: str                   # required; adapter name
    id: Optional[str]
    speaker_id: Optional[str]
    language: Optional[str]        # ISO 639-1, e.g. "bn"
```

`dataset` is now required so multi-source training has a stable per-source
filter without inspecting paths.

### `DatasetAdapter` ABC

```python
class DatasetAdapter(abc.ABC):
    name: str                                  # registry key, e.g. "openslr53"
    requires_credentials: tuple[str, ...]      # env-var names needed

    @abc.abstractmethod
    def download(self, dest_root: Path) -> Path:
        """Idempotent: returns the raw-data directory. Skip if already present."""

    @abc.abstractmethod
    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        """Yield one Record per audio clip. `audio_filepath` may be relative
        to raw_dir; pack.py will absolutize."""
```

### Credentials (user choice: hardcoded for now)

Living in **`clae_data/_creds.py`** (gitignored — see note below). Single
source of truth, imported by adapters and the publish step:

```python
# clae_data/_creds.py  — gitignored; provide a _creds.example.py template
HF_TOKEN = "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"   # ROTATE before going public
KAGGLE_USERNAME = "aryanrahman"
KAGGLE_KEY = "KGAT_38471085ebbafd3d0c544e1954296b39"  # ROTATE before going public
WANDB_API_KEY = "<paste-here>"                        # ROTATE before going public

# Hub targets (not secrets but kept together for one-stop config):
CLAE_HF_REPO   = "aryanrahman/clae-bengali"
CLAE_CKPT_REPO = "aryanrahman/clae-bengali-encoder"

# Local paths
CLAE_DATA_ROOT = "/data/clae"   # on cloud GPU: /workspace/data, etc.
```

Why a separate file (not env vars yet): user explicitly chose hardcoded for
research-velocity. Putting them in one gitignored file means:
- Rotation = edit one file.
- Public-repo migration = delete the file + switch to env vars + rotate
  keys. The README banner says so.
- Adapters do `from clae_data._creds import HF_TOKEN, …`. No `os.environ`
  reads scattered across the code.

`.gitignore` must include `clae_data/_creds.py`. Ship
`clae_data/_creds.example.py` with placeholders so new clones know what to
fill in. **Note**: the existing `scripts/datasets_download.py` already has
the keys committed to git history; rotating later requires
`git filter-repo` as noted in §0.

For `utils/logging.py:maybe_init_wandb`, set `os.environ["WANDB_API_KEY"] =
WANDB_API_KEY` once at training-script startup before `wandb.init()` runs.
That keeps the actual wandb library env-driven (its normal interface)
without changing its call sites.

---

## §2.5 — Audio preprocessing decisions

Sources are heterogeneous:

| Source | Format | Sample rate | Channels |
|--------|--------|-------------|----------|
| OpenSLR-53 | FLAC | 16 kHz | mono |
| BengaliAI Speech (Kaggle) | MP3 | 32 kHz | mono |
| RegSpeech12 | WAV | varies (22–48 kHz) | varies |
| IndicVoices_R (HF parquet) | FLAC bytes | 16 kHz | mono |
| SUBAK.KO | WAV | 16 kHz | mono |
| Shrutilipi | WAV | 16 kHz | mono |
| Kathbath (probe only) | WAV | 16 kHz | mono |

Training reads at 16 kHz mono (`cfg.data.sample_rate: 16000`). So pack-time
transforms every clip to a single canonical form:

1. **Decode** with `torchaudio.load(path)` (handles mp3/wav/flac/ogg).
2. **Mono fold** if `wav.size(0) > 1`: `wav.mean(dim=0)`.
3. **Resample** to 16 kHz via `torchaudio.functional.resample` (or
   `torchaudio.transforms.Resample` cached per source-rate for speed).
4. **Loudness normalize** (optional but recommended): RMS-normalize to
   −23 dB FS (broadcast standard). Skip for now — adds complexity, and
   `data/augment.py:WaveAugConfig.gain` already randomizes volume during
   training. Mark TODO.
5. **Duration filter**: drop clips outside `[1.0, 30.0]` seconds. Below
   1s: too short for SSL (segment_seconds=3 → would always be padded).
   Above 30s: rare, blows up VRAM if accidentally fed to ASR probe at
   15 s segment.
6. **Encode** to FLAC (compression level 5, default). FLAC is lossless,
   ~50% the size of the equivalent WAV. NEVER re-encode to MP3 (lossy +
   adds artifacts that hurt SSL representations).
7. **Write** to `staging/audio/<dataset>/<id>.flac`.

Why FLAC, not MP3:
- MP3 is lossy. Generative-audio SSL methods (which this is) are
  spectrally sensitive — MP3's frequency-domain truncation creates
  artifacts that the model can learn to reconstruct, contaminating the
  reconstruction metrics.
- FLAC is lossless. Bytes-in == bytes-out at 16-bit.
- Size: at 16 kHz mono, FLAC is ~80–120 kB/s. A 1M-utterance corpus
  averaging 8s/clip is ~0.8 TB raw → ~150 GB FLAC. Manageable on HF Hub
  LFS.

Pack-time also fills in `duration` (post-resample, in seconds) and
`sample_rate` (always 16000 after pack). Audit step is redundant after
pack — it's run *before* pack on raw files to catch corruption early.

You can pick where to stop. Tier 1 unifies the surface; Tier 2 makes cloud
training actually one-command.

### Tier 1 — Local unification (~1 day of subagent work)

Output is still JSONL. No HF Hub push. Goal: one entry point.

```bash
# Download raw archives for selected adapters
python -m clae_data download --datasets openslr53,bengaliai_speech,regspeech12

# Build records → audit → clean → split → emit JSONL
python -m clae_data build \
    --datasets openslr53,bengaliai_speech,regspeech12,indicvoices \
    --out data/manifests/ \
    --val_pct 0.05 \
    --min_duration 2.0
```

`build` writes:
- `data/manifests/train.jsonl` — combined pretraining manifest (all rows)
- `data/manifests/val.jsonl` — held-out from each dataset proportionally
- `data/manifests/asr_probe_train.jsonl` — subset of train where `text is not None`
- `data/manifests/asr_probe_val.jsonl` — same for val
- `data/manifests/build_meta.yaml` — per-dataset row counts, hashes,
  audit summary, git hash. Reproducibility record.

The 10+ scripts collapse to one module. Existing `train.py` keeps working
unchanged (it already eats `data/manifests/train.jsonl`).

**Subagent prompts for Tier 1** (give them one at a time):

1. *"Create `clae_data/schema.py` (Record TypedDict + validator) and
   `clae_data/adapters/base.py` (DatasetAdapter ABC). Port the
   download/iter logic for OpenSLR-53 from `scripts/prepare_openslr53.py`
   and `scripts/datasets_download.py:download_openslr_part` into
   `clae_data/adapters/openslr53.py`. Credentials: none."*
2. *"Port `scripts/prepare_bengaliai.py` + `scripts/create_bengaliai_manifests.py`
   (they overlap) into `clae_data/adapters/bengaliai_speech.py`. Use the
   Kaggle adapter from `scripts/datasets_download.py:download_kaggle_dataset`.
   Credentials: KAGGLE_USERNAME, KAGGLE_KEY."*
3. Repeat for regspeech12, indicvoices (from prepare_hf_parquet), subak_ko,
   shrutilipi, kathbath.
4. *"Port `scripts/audit_datasets.py` to `clae_data/audit.py` as a callable
   function `audit_records(records: Iterable[Record], num_workers=4)
   -> tuple[list[Record], AuditReport]` returning kept records and the
   report. Use soundfile.info per current behaviour."*
5. *"Write `clae_data/cli.py` with two subcommands: `download` and `build`.
   `build` calls each adapter's iter_records, runs audit, dedupes by
   audio_filepath, splits train/val per-dataset (so each source contributes
   proportionally to val), and writes the four JSONLs + build_meta.yaml
   listed above. Use a deterministic seed."*
6. *"Delete the old `scripts/prepare_*.py`, `scripts/create_*manifest*.py`,
   `scripts/audit_datasets.py`, `scripts/finalize_manifests.py`,
   `scripts/datasets_download.py`, `scripts/prepare_remaining_datasets.py`.
   Keep `scripts/reconstruct_sample.py`, `scripts/smoke_encoder_mhc.py`,
   `scripts/verify_experiment.py`, `scripts/visualize_latents.py`,
   `scripts/check_rank.py`, `scripts/count_params.py`, `scripts/get_param_count.py`."*

### Tier 2 — HF Hub as cloud storage (raw files + JSONL, NOT parquet)

**Decision (2026-05-22): use HF dataset repos as raw-file blob storage.**
NOT parquet shards. Reasons (from user discussion):

- The JSONL schema may evolve (add `language`, `emotion`, etc.) — parquet
  shards lock the schema and concat-republish for growth is heavy.
- Data grows incrementally as new sources are added. With raw files, growth
  = `upload_folder` more files + a new versioned `manifests/train_v2.jsonl`.
  With parquet, growth = download-everything, concatenate, push-everything.
- Audio counts in the low millions at most. Per-file LFS overhead is fine
  at this scale.
- `data/dataset.py` already eats JSONL → no parser changes, just resolve
  relative paths.

**Repo layout on HF Hub:**

```
aryanrahman/clae-bengali/                  (HF dataset repo, LFS-backed)
  audio/
    openslr53/<utt_id>.flac
    bengaliai_speech/<id>.mp3
    regspeech12/<id>.wav
    indicvoices/<id>.flac
  manifests/
    train.jsonl              # paths relative to repo root
    val.jsonl
    asr_probe_train.jsonl    # filtered subset where text is not None
    asr_probe_val.jsonl
  README.md                  # dataset card
  build_meta.yaml            # row counts, source provenance, build git hash
```

JSONL `audio_filepath` values are **relative to repo root**, e.g.
`audio/openslr53/utt00001.flac`. The training-side code resolves them by
prepending `$CLAE_DATA_ROOT` (where `snapshot_download` dropped the repo).

**Pack step (on prep instance):**

```bash
python -m clae_data pack \
    --datasets openslr53,bengaliai_speech,regspeech12,indicvoices \
    --target_sr 16000 \
    --repo_id aryanrahman/clae-bengali \
    --push
```

What it does:
1. For each adapter, iterate records (already audited).
2. Optionally resample/transcode to 16 kHz mono FLAC for size + uniformity.
   Skip if the source is already 16 kHz mono — preserves the original file
   bytes-for-bytes when possible.
3. Write to `<staging_dir>/audio/<dataset>/<id>.flac`.
4. Emit `<staging_dir>/manifests/*.jsonl` with relative paths.
5. `HfApi().upload_folder(folder_path=staging_dir, repo_id=...,
   repo_type="dataset")`. LFS-tracks `audio/**` automatically.

**Cloud GPU consumption (the seamless part):**

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
make train
```

`make train` runs:
```bash
huggingface-cli download --repo-type dataset aryanrahman/clae-bengali \
    --local-dir $CLAE_DATA_ROOT
python train.py --config configs/exp0.yaml \
    data.train_manifest=$CLAE_DATA_ROOT/manifests/train.jsonl \
    data.val_manifest=$CLAE_DATA_ROOT/manifests/val.jsonl
```

`huggingface-cli download` is idempotent and resumable — re-running just
verifies LFS pointers, doesn't re-download.

**Required `data/dataset.py` change (small):**
- Resolve relative `audio_filepath` against the *directory containing the
  manifest*. Two-line change in `AudioDataset.__getitem__`:
  ```python
  path = item.get("audio_filepath") or item.get("path") or item["audio"]
  if not os.path.isabs(path):
      path = os.path.join(self._manifest_dir, path)
  ```
  Where `self._manifest_dir` is captured in `__init__` from `cfg.manifest`.
- Drop the `path | audio` fallback keys (standardize on `audio_filepath`).

This is the entire training-side change. No `datasets` library import, no
parquet handling, no streaming dataset wrapper.

**Growth pattern (months later, new source arrives):**

```bash
# On prep instance:
python -m clae_data pack \
    --datasets new_source \
    --repo_id aryanrahman/clae-bengali \
    --append \
    --manifest_version v2

# This:
# - Uploads audio/new_source/*.flac (HF Hub LFS dedupes by content hash)
# - Writes new manifests/train_v2.jsonl combining the old manifest's lines
#   with the new source's lines
# - Leaves manifests/train.jsonl untouched (reproducibility)
# - Bumps build_meta.yaml
```

Training-side switch is one config edit:
`data.train_manifest: manifests/train_v2.jsonl`.

**Subagent prompts for Tier 2:**

1. *"Add `clae_data/pack.py` with one function:
   `pack_to_dir(adapters: list[DatasetAdapter], staging_dir: Path,
   target_sr: int = 16000) -> None`. For each adapter, iterate records,
   resample audio to target_sr mono if needed (use torchaudio.load +
   torchaudio.functional.resample), write FLAC via soundfile.write, emit
   the 4 JSONLs in `staging_dir/manifests/` with paths relative to
   staging_dir. Preserve all metadata fields (text, dataset, id, speaker_id,
   duration, language). Skip transcoding when source is already 16k mono."*

2. *"Add `clae_data/push.py:push_to_hub(staging_dir: Path, repo_id: str,
   token: str | None = None) -> None` that calls `HfApi(token=token or
   os.environ['HF_TOKEN']).upload_folder(folder_path=str(staging_dir),
   repo_id=repo_id, repo_type='dataset', commit_message='pack <git_hash>')`.
   Create repo with `exist_ok=True` first. Auto-generate a minimal
   `README.md` with a dataset card YAML header listing sources."*

3. *"Modify `data/dataset.py` to resolve relative `audio_filepath` against
   the manifest's parent directory. Drop the `path` / `audio` fallback keys
   — standardize on `audio_filepath`. The dataset's `__init__` should
   capture `self._manifest_dir = pathlib.Path(cfg.manifest).parent` and
   `__getitem__` should do `path = self._manifest_dir / item['audio_filepath']
   if not os.path.isabs(item['audio_filepath']) else item['audio_filepath']`.
   Also support a list-of-manifests `cfg.manifest` by capturing the parent
   per-row."*

4. *"Add `clae_data/cli.py:fetch_command` that wraps
   `huggingface_hub.snapshot_download(repo_id, repo_type='dataset',
   local_dir=dest)`. Default dest is `$CLAE_DATA_ROOT`. Print the resolved
   manifest paths so the user can paste them into a config or Make var."*

---

## §4 — Checkpoint publishing

Add `clae_data/publish_checkpoint.py`:

```bash
python -m clae_data publish_checkpoint \
    --ckpt runs/<run_id>/checkpoints/last.pt \
    --repo_id aryanrahman/clae-bengali-encoder \
    --commit_message "step 30000 — run <run_id>"
```

What it does:
- `huggingface_hub.HfApi().create_repo(repo_id, exist_ok=True)`
- Upload: `last.pt`, `config.yaml` (from the run's `run_meta.yaml`),
  a small auto-generated `README.md` model card with train args + W&B link.
- Optional: also upload `train.jsonl` first lines as a sample.

`train.py` does **not** call this — it's a separate step the Makefile
sequences after training. Keeps the training process free of network
side-effects when running.

---

## §5 — Cloud-GPU Makefile

Mirror your example. Drop into repo root as `Makefile`:

```makefile
# === Config (override at invocation, e.g. `make train RUN_NAME=foo`) ===
CLAE_DATA_ROOT  ?= $(HOME)/data/clae
CLAE_HF_REPO    ?= aryanrahman/clae-bengali
CLAE_CKPT_REPO  ?= aryanrahman/clae-bengali-encoder
CONFIG          ?= configs/exp0.yaml
OUTPUT_DIR      ?= runs
WANDB_PROJECT   ?= continuous-latent-ae

TRAIN_DEFAULT_ARGS ?= \
    --config $(CONFIG)

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: ensure-uv ensure-env prepare-data train evaluate publish all

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir $$HOME/.local/bin --force)

ensure-env:
	@: $${HF_TOKEN:?HF_TOKEN env var not set}
	@: $${WANDB_API_KEY:?WANDB_API_KEY env var not set}

# Tier 1 form: download raw + build local JSONL.
prepare-data-tier1: ensure-uv ensure-env
	uv sync
	uv run python -m clae_data download --datasets openslr53,bengaliai_speech,regspeech12,indicvoices
	uv run python -m clae_data build \
	    --datasets openslr53,bengaliai_speech,regspeech12,indicvoices \
	    --out data/manifests/ --val_pct 0.05

# Tier 2 form: download pre-packed HF dataset (recommended on cloud).
prepare-data: ensure-uv ensure-env
	uv sync
	uv run python -m clae_data fetch --repo_id $(CLAE_HF_REPO) --dest $(CLAE_DATA_ROOT)

train: prepare-data
	@set -e; \
	run_name=$${RUN_NAME:-clae-$$(date +%Y%m%d-%H%M%S)}; \
	uv run python train.py $(TRAIN_DEFAULT_ARGS) run.run_id=$$run_name run.out_dir=$(OUTPUT_DIR)

evaluate:
	@latest=$$(ls -1t $(OUTPUT_DIR)/*/checkpoints/last.pt 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then echo "No checkpoint found"; exit 1; fi; \
	uv run python -m eval.eval_asr \
	    --config $(CONFIG) --ckpt $$latest \
	    --train_manifest data/manifests/asr_probe_train.jsonl \
	    --dev_manifest data/manifests/asr_probe_val.jsonl \
	    --out $$(dirname $$latest)/../eval/asr.json

publish: ensure-env
	@latest=$$(ls -1t $(OUTPUT_DIR)/*/checkpoints/last.pt 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then echo "No checkpoint found"; exit 1; fi; \
	uv run python -m clae_data publish_checkpoint --ckpt $$latest --repo_id $(CLAE_CKPT_REPO)

all: train evaluate publish
```

**Cloud invocation (full end-to-end):**

```bash
git clone https://github.com/aryanrahman/continuous-latent-autoencoder
cd continuous-latent-autoencoder
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
make all
```

That's the goal state.

---

## §6 — Wandb wiring (small)

`utils/logging.py:maybe_init_wandb` already reads `cfg.run.wandb.{enabled,
project,name}` and calls `wandb.init`. The only missing piece for
cloud-seamless is auth: `wandb.init` honours `WANDB_API_KEY` env var
automatically (no explicit `wandb.login()` needed), so the Makefile's
`ensure-env` guard is sufficient.

Optional polishings (low priority):
- Set `wandb.config` from `cfg.model_dump()` already — good.
- Add `wandb.run.log_code(".")` after init to snapshot the source. One
  line, useful for reproducibility.
- After `make publish`, append the HF Hub model URL to the wandb run's
  summary via `wandb.run.summary["hf_repo"] = url`.

---

## §7 — Sequencing for subagents

Recommended order (each is one subagent task, results visible before next):

1. **§0 credential rotation** — you do this, not a subagent. Block on it.
2. **Tier 1, prompts 1–4** — port adapters one by one. Each adapter PR can
   be reviewed independently. Keep the old `scripts/prepare_*.py` files
   until all adapters are ported.
3. **Tier 1, prompt 5** — the CLI. Run on a small subset locally to
   validate (set `--limit 100`) before going full.
4. **Tier 1, prompt 6** — delete the old scripts.
5. **Decision point**: stop here if local-machine training is enough;
   otherwise proceed to Tier 2.
6. **Tier 2, prompts 1–2** — pack + push. Run pack locally on a tiny
   subset first; eyeball the resulting parquet.
7. **Tier 2, prompt 3** — `data/dataset.py` modification.
8. **§4 publish_checkpoint** — separate small module, ~50 lines.
9. **§5 Makefile** — last, once everything it calls exists.

Smoke test at each step: `python -m clae_data build --datasets openslr53
--limit 100 --out /tmp/clae_test/` should produce a working train.jsonl
that `train.py --config configs/exp0.yaml --max_steps 2
data.train_manifest=/tmp/clae_test/train.jsonl` can consume.

---

## §8 — Things explicitly out of scope

- Switching `train.py` to use `datasets.IterableDataset` for streaming.
  Worth doing eventually for very large corpora (>500 GB) but not now.
- WebDataset/tar shards. Considered earlier (`scripts/pack_webdataset.py`,
  now deleted) — HF parquet is the simpler answer and integrates with
  `datasets.load_dataset` natively.
- Multi-lingual support beyond Bengali. Adapters take a `language` field
  in Record but no current adapter sets it.
- Per-utterance forced alignment / frame-level labels. The current ASR
  probe is CTC over utterance text — adequate for sanity-checking.
