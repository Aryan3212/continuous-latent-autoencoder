# continuous-latent-autoencoder

> ⚠️ **CREDENTIALS NOTICE** — `clae_data/_creds.py` (gitignored) and
> `scripts/datasets_download.py` (in git history) contain hardcoded HF,
> Kaggle, and WandB API keys for research-velocity. **Before making this
> repo public, or before sharing access with anyone outside the author**:
>
> 1. Rotate the HF token at https://huggingface.co/settings/tokens
> 2. Rotate the Kaggle key at https://www.kaggle.com/settings
> 3. Rotate the WandB key at https://wandb.ai/settings
> 4. Purge the old keys from git history (they remain in history from the
>    deleted `scripts/datasets_download.py` even after removal from `HEAD`):
>    write a `replacements.txt` with one `OLD_SECRET==>REDACTED` line per key,
>    then run `git filter-repo --replace-text replacements.txt`, force-push,
>    and have any collaborators re-clone. (`pip install git-filter-repo`; run
>    on a fresh clone — it rewrites all commit hashes.)

Deterministic continuous-latent speech foundation autoencoder:

- Waveform-in (16kHz), strided Conv1D frontend → ~12.5 Hz tokens (hop 1280 samples)
- Encoder produces deterministic continuous latents `z ∈ R^{B×d×T'}` (default `d=16`, i.e. 200 dims/sec at 12.5 Hz)
- Joint objectives (Exp0): waveform reconstruction (Multi-Res STFT) + LeJEPA-style predictive loss + SIGReg
- Decoder is trained to reconstruct waveform and (later) tolerate latent noise

## One-command training on a cloud GPU

The training pipeline is wrapped behind a `Makefile`, with `setup.sh` as the
fresh-instance entrypoint. On a newly provisioned cloud-GPU instance with
`git` and `make` installed:

```bash
git clone <repo-url>
cd continuous-latent-autoencoder
cp .env.example .env && nano .env   # paste HF_TOKEN (+ WANDB_API_KEY)
tmux new -s train                   # so training outlives the SSH session
./setup.sh                          # deps -> creds -> fetch -> train
```

`setup.sh` generates `clae_data/_creds.py` from `.env`, falls back to
offline W&B logging when `WANDB_API_KEY` is unset, and accepts `--no-train`
to stop after the dataset fetch. Equivalently, by hand:

```bash
# Copy clae_data/_creds.example.py to clae_data/_creds.py and edit, then:
make all
```

`make all` runs four stages back-to-back:

1. `prepare` installs Python deps via `uv sync`.
2. `fetch-data` snapshot-downloads the packed dataset from HF Hub into
   `$CLAE_DATA_ROOT/` (default `$HOME/data/clae`). Idempotent — re-running
   skips files already present.
3. `train` runs `train.py` with the manifests at
   `$CLAE_DATA_ROOT/manifests/{train,val}.jsonl`. WandB receives logs;
   checkpoints land in `runs/<run_id>/checkpoints/`.
4. `evaluate` runs the offline ASR probe (`eval/eval_asr.py`) against the
   most recent `last.pt`.
5. `publish` uploads `last.pt` plus a generated model card to the HF model
   repo (`$CLAE_CKPT_REPO`).

Any variable can be overridden on the command line:

```bash
RUN_NAME=ablation-no-mhc make train
DATASETS=openslr53 make pack-and-push
CONFIG=configs/local_6gb.yaml make train
```

`make help` lists every target and the current value of every variable. The
training instance never needs Kaggle credentials — only HF + WandB.

## Dataset preparation (one-time, on a prep instance)

The packed dataset is built once on a separate "prep" instance (any beefy
box with HF + Kaggle keys configured) and pushed to HF Hub. The training
instance only consumes it via `make fetch-data`.

```bash
# On the prep instance:
# 1) Edit clae_data/_creds.py with HF + Kaggle + MDC keys.
make pack-and-push DATASETS=openslr53,common_voice_bn,regspeech12,indicvoices
```

What `pack-and-push` does:

- Downloads raw archives from HF Hub, Kaggle, and OpenSLR into
  `$CLAE_DATA_ROOT/`. Each adapter is responsible for its own source.
- Iterates each adapter's records (one per audio clip, with transcript +
  language + dataset tag).
- Audits files via `soundfile.info`, dropping clips < 1s, > 30s, or
  unreadable.
- Resamples every kept clip to 16 kHz mono FLAC.
- Writes `audio/<dataset>/<id>.flac` plus four JSONL manifests under
  `manifests/`: `train.jsonl`, `val.jsonl`, `asr_probe_train.jsonl`, and
  `asr_probe_val.jsonl`. Paths inside each manifest are relative to the
  staging root.
- Uploads the entire packed layout to `$CLAE_HF_REPO` (default
  `aryanrahman/clae-bengali`) via `huggingface_hub.upload_folder`.

Once pushed, the training instance just runs `make all` — no Kaggle
credentials needed there.

The packed format is "raw files + JSONL", not parquet shards, because that
makes incremental growth trivial: adding a new source is just another
`upload_folder` call plus a versioned `manifests/train_v2.jsonl` — no schema
migration or download-concatenate-republish cycle.

## Adding a new dataset source

1. Implement an adapter in `clae_data/adapters/<name>.py` that subclasses
   `DatasetAdapter`. Two methods are required:
   - `download(dest)` — idempotent; place raw archives under `dest` and
     return the path to the raw directory.
   - `iter_records(raw_dir)` — yield `Record` dicts with at minimum
     `audio_filepath`, optionally `text`, plus `dataset=<name>` and
     `language`.
2. Register the adapter in `clae_data/registry.py`.
3. On the prep instance, re-run
   `make pack-and-push DATASETS=...,<name>`. Use a versioned manifest
   (e.g. `manifests/train_v2.jsonl`) so older runs stay reproducible.

## Quick start (Exp0) with `uv`

For local development (no HF Hub fetch):

1) Create a `uv` environment (recommended Python is 3.11/3.12 for PyTorch wheels):

```bash
uv venv --python 3.11
```

2) Install deps (either):

```bash
uv sync
```

or:

```bash
uv pip install -r requirements.txt
```

3) Prepare your datasets. For a full pipeline build, use the prep-instance
workflow above (`make pack-and-push`). For an existing packed layout under
`$CLAE_DATA_ROOT/`, the manifests at
`$CLAE_DATA_ROOT/manifests/{train,val}.jsonl` are ready to use directly.

4) Run:

```bash
uv run python train.py --config configs/exp0.yaml \
    data.train_manifest=$HOME/data/clae/manifests/train.jsonl \
    data.val_manifest=$HOME/data/clae/manifests/val.jsonl
```

Artifacts:

- Checkpoints in `runs/<run_id>/checkpoints/`
- Logs in `runs/<run_id>/logs/` (and W&B if enabled)

## Notes

- This repo intentionally starts with a stable, low-compute Exp0 (no GAN, no mixture).
- Earlier experimental paths (mixture, primary classification, latent-noise decoding, GAN) have been **removed** from the code; they live in git history and can be reintroduced if needed. The current objective is reconstruction + JEPA + SIGReg only.

## Folder guide

- `configs/`: experiment configs (`exp0.yaml` cloud ~6M, `local_6gb.yaml` local PC)
- `clae_data/`: dataset prep package (adapters → pack → push/fetch)
- `data/`: dataset loading + augmentation
- `eval/`: probes + evaluation entrypoints (`eval_asr.py`, `eval_cls_probe.py`, `eval_recon.py`, `run_all.py`)
- `losses/`: loss functions (multi-res STFT)
- `models/`: core model components (frontend, encoder, mHC, projector, decoder, sigreg)
- `scripts/`: one-off utilities + smoke scripts
- `utils/`: misc helpers (config/schema, logging, checkpointing)
- `reference-implementations/`: slim in-tree references (single-file impls + notes); full vendored repos live at `../reference-implementations-archive`
- `docs/`: live research notes
- `runs/`: training outputs (checkpoints/logs)

## Running things

Train (use `configs/local_6gb.yaml` instead of `exp0.yaml` on the local PC):

```bash
uv run python train.py --config configs/exp0.yaml \
    data.train_manifest=$HOME/data/clae/manifests/train.jsonl \
    data.val_manifest=$HOME/data/clae/manifests/val.jsonl
```

Run all eval (reconstruction + all enabled probes in one go):

```bash
uv run python -m eval.run_all --config configs/exp0.yaml --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl --out_dir runs/eval
```

Reconstruction-only eval:

```bash
uv run python -m eval.eval_recon --config configs/exp0.yaml --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl --out runs/recon.json
```

Frozen-encoder ASR probe (CTC head → WER):

```bash
uv run python -m eval.eval_asr --config configs/exp0.yaml --ckpt /path/to/ckpt.pt \
    --train_manifest /path/to/asr_probe_train.jsonl \
    --dev_manifest /path/to/asr_probe_val.jsonl --out runs/asr_probe.json
```

Emotion / gender probes (pooled-embedding MLP; same arg shape as the ASR probe; `--label_key` picks the task):

```bash
uv run python -m eval.eval_cls_probe --config configs/exp0.yaml --ckpt /path/to/ckpt.pt --label_key emotion ...
uv run python -m eval.eval_cls_probe --config configs/exp0.yaml --ckpt /path/to/ckpt.pt --label_key gender  ...
```

Visualize the latent space (PCA/UMAP):

```bash
uv run python scripts/visualize_latents.py --config configs/exp0.yaml --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl --out runs/latents.png --limit 500
```

Unit tests:

```bash
uv run pytest tests/
```

Smoke tests (model components, no data needed):

```bash
PYTHONPATH=. uv run --no-project python scripts/smoke_encoder_mhc.py
```

## Static analysis (dead-code / unused-symbol audit)

The project has no static-analysis deps installed. Tools below run via
ephemeral `uvx` / `npx` envs and write reports under `.static-analysis/`.
None of them touch `pyproject.toml`.

```bash
mkdir -p .static-analysis

# 1. ruff — fast linter (Rust). Finds unused imports/vars, dead branches,
#    undefined names (real bugs), commented-out code.
uvx ruff check --select F,ARG,ERA,RUF --output-format=concise \
    train.py models/ data/ losses/ eval/ utils/ tests/ \
    > .static-analysis/ruff.txt

# 2. vulture — flags unused functions/classes/methods/attrs across modules.
#    The allowlist suppresses framework magic (torch.nn.Module.forward, etc.)
uvx vulture --min-confidence 60 \
    train.py models/ data/ losses/ eval/ utils/ tests/ \
    .static-analysis/vulture-allowlist.py \
    > .static-analysis/vulture.txt

# 3. pyright — cross-file unused-symbol + unreachable + undefined-name pass.
#    Ignore `reportMissingImports` warnings (torch/yaml/etc. aren't installed).
npx -y pyright --project .static-analysis/pyrightconfig.json --outputjson \
    > .static-analysis/pyright.json
jq -r '.generalDiagnostics[]
       | select(.rule | test("reportUnused|reportUndefined|reportUnreachable"))
       | "\(.severity)\t\(.rule)\t\(.file)\t\(.range.start.line + 1)\t\(.message)"' \
    .static-analysis/pyright.json \
    > .static-analysis/pyright-clean.txt
```

Static analysis catches **symbol-level** dead code (unused imports/functions/
vars, undefined names). It cannot catch **config-gated** dead branches (e.g.
`if mix_recon_enabled:`). For those, delete the branches by hand first, then
re-run the tools — orphans cascade out.
