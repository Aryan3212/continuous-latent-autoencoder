# continuous-latent-autoencoder

> ⚠️ **CREDENTIALS NOTICE** — credentials now live only in `.env` (gitignored)
> and are read from the environment at runtime; there is no committed creds
> file. However, **git history** still contains hardcoded HF, Kaggle, and WandB
> keys (from the old `clae_data/_creds.py` and `scripts/datasets_download.py`).
> **Before making this repo public, or before sharing access with anyone
> outside the author**:
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

## Workflow

The pipeline has two steps:

### 1. Manifests

Point at raw audio data and build `train.jsonl` / `val.jsonl` manifests:

```bash
# Download raw source datasets
make download-data DATASETS=openslr53,bengaliai_speech

# Build manifests from downloaded data
make make-manifests DATASETS=openslr53,bengaliai_speech
```

The `make-manifests` target calls `scripts/housekeeping.py make-manifests --data-root <DATA_ROOT> --datasets <DATASETS> --out-dir staging/manifests`, which iterates each adapter's records (audio path + transcript metadata), shuffles, and performs a per-dataset stratified split into train/val.

For Kaggle sessions where datasets are pre-mounted at known paths, use `--map` instead:

```bash
uv run python scripts/housekeeping.py make-manifests \
    --map regspeech12=/kaggle/input/regspeech12 \
    --map common_voice_bn=/kaggle/input/common-voice-24-bn \
    --out-dir /kaggle/working/manifests
```

### 2. Train

```bash
make train CONFIG=configs/exp0.yaml
```

This runs `train.py` with the manifests under `staging/manifests/`. Override the manifest path with `MANIFEST_DIR`:

```bash
make train MANIFEST_DIR=/custom/path/manifests
```

Override any config field via trailing dotted args:

```bash
CONFIG=configs/local_6gb.yaml make train TRAIN_EXTRA_ARGS="train.max_steps=5000 data.num_workers=4"
```

### Checkpoint resume

Resume from a saved checkpoint:

```bash
uv run python train.py --config configs/exp0.yaml \
    --resume runs/<run_id>/checkpoints/last.pt \
    data.train_manifest=... data.val_manifest=...
```

`--max_hours` stops cleanly after a wall-clock budget (useful for fixed-length sessions):

```bash
uv run python train.py --config configs/exp0.yaml --max_hours 11.5 \
    data.train_manifest=... data.val_manifest=...
```

### Publish to HF Hub

Upload the latest checkpoint to a Hugging Face model repo:

```bash
uv run python scripts/housekeeping.py publish-checkpoint \
    --ckpt runs/<run_id>/checkpoints/last.pt \
    --repo-id your-org/your-model
```

Pull the latest published checkpoint (e.g. to resume a multi-session training run):

```bash
uv run python scripts/housekeeping.py fetch-checkpoint \
    --repo-id your-org/your-model --dest runs/my_run/checkpoints/last.pt
```

### Kaggle multi-session workflow

See `scripts/kaggle_session.sh` for the production multi-session pattern: build manifests over attached datasets → pull latest checkpoint from HF → train with `--max_hours` → publish checkpoint on any exit. Each 12h Kaggle session resumes where the last left off.

## Quick start with `uv`

For local development with existing manifests:

```bash
uv sync
uv run python train.py --config configs/exp0.yaml \
    data.train_manifest=/path/to/manifests/train.jsonl \
    data.val_manifest=/path/to/manifests/val.jsonl
```

Artifacts:

- Checkpoints in `runs/<run_id>/checkpoints/`
- Logs in `runs/<run_id>/logs/` (and W&B if enabled)

## Folder guide

- `configs/`: experiment configs (`exp0.yaml` cloud ~6M, `local_6gb.yaml` local PC)
- `data_loading.py`: dataset loading (JSONL manifests) + waveform augmentation
- `datasets/`: gitignored; where housekeeping fetches raw archives (default `$DATA_ROOT`)
- `eval/`: probes + evaluation entrypoints (`eval_asr.py`, `eval_cls_probe.py`, `eval_recon.py`, `run_all.py`)
- `losses.py`: multi-res STFT reconstruction loss (`MultiResSTFTLoss`)
- `schema.py`: pydantic config schema (single source of truth, `extra="forbid"`)
- `config.py`: `load_config` / `apply_overrides` (YAML → validated `Config`)
- `models/`: core model components (frontend, encoder, mHC, projector, decoder, sigreg)
- `staging/`: manifests + transcoded audio (output of make-manifests)
- `scripts/`: utilities — `housekeeping.py` (data/artifact CLI: adapters → download/manifests + publish-checkpoint), `reconstruct_audio.py`, `visualize_latents.py`, `fill_durations.py`
- `reference-implementations/`: slim in-tree references (single-file impls + notes); full vendored repos live at `../reference-implementations-archive`
- `runs/`: training outputs (checkpoints/logs)

## Running things

Train (use `configs/local_6gb.yaml` instead of `exp0.yaml` on the local PC):

```bash
uv run python train.py --config configs/exp0.yaml \
    data.train_manifest=/path/to/manifests/train.jsonl \
    data.val_manifest=/path/to/manifests/val.jsonl
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

`train.py` prints a per-block trainable-parameter breakdown at startup.
