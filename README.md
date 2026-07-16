# Continuous Latent Autoencoder

A config-driven speech autoencoder for 16 kHz audio. It learns low-rate
continuous latents using waveform reconstruction, JEPA view consistency, and
SIGReg or VISReg. A spectrogram-domain GAN is optional.

For an implementation map, see [CODEBASE.md](CODEBASE.md). Configuration lives
in `schema.py`; unknown YAML fields are rejected.

## Quick start

You need Linux, an NVIDIA GPU, Python 3.12, [uv](https://docs.astral.sh/uv/),
and FFmpeg.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-extra
uv sync
```

Copy `.env.example` to `.env` and add only the credentials you need. Then build
manifests for one of the supported datasets:

```bash
cp .env.example .env
make download-data make-manifests DATASETS=openslr53
```

Start with the low-memory local config:

```bash
make train \
  CONFIG=configs/local_6gb.yaml \
  MANIFEST_DIR=staging/manifests \
  RUN_NAME=first-run \
  TRAIN_EXTRA_ARGS='run.wandb.enabled=false'
```

`local_6gb.yaml` is only a starting point, not a memory guarantee. Lower
`train.batch_size` if it does not fit your GPU.

Runs are written under `runs/<run-id>/`. The important files are
`checkpoints/last.pt`, interval checkpoints in `checkpoints/`, `config.yaml`,
and `logs/train.jsonl`. Training has no in-loop validation; evaluate saved
checkpoints separately.

## Data

The data CLI supports `openslr53`, `common_voice_bn`, `bengaliai_speech`,
`regspeech12`, `indicvoices`, `subak_ko`, `shrutilipi`, and `kathbath`.

The Make targets download data into `datasets/` and write `train.jsonl` and
`val.jsonl` to `staging/manifests/` by default. Override `DATA_ROOT`,
`MANIFEST_DIR`, or `DATASETS` when needed.

For datasets that are already mounted, build manifests directly:

```bash
uv run python scripts/housekeeping.py make-manifests \
  --map regspeech12=/path/to/regspeech12 \
  --map common_voice_bn=/path/to/common-voice-bn \
  --out-dir staging/manifests
```

Each JSONL row must contain `audio_filepath`. Relative paths resolve from the
manifest directory, or its parent when manifests live in `<root>/manifests/`.
Transcripts, durations, dataset names, and labels are optional.

## Configs and overrides

Use a full config from `configs/`; `kaggle_3m_gan.yaml` is the only inherited
variant. `train.py` accepts `--max_hours` and dotted `key=value` overrides. With
Make, put overrides in `TRAIN_EXTRA_ARGS`. Values are parsed as YAML, so numbers,
booleans, lists, and `null` work.

## Resume and multi-GPU

Resume with the checkpoint's original config, changing only runtime, data, or
schedule fields. The current config controls the LR schedule; training warns if
its LR, warmup, horizon, or minimum ratio differs from the checkpoint.

```bash
uv run python train.py \
  --config configs/local_6gb.yaml \
  --resume runs/first-run/checkpoints/last.pt \
  data.train_manifest=staging/manifests/train.jsonl \
  run.run_id=first-run
```

For NCCL DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/large_2kh.yaml \
  data.train_manifest=staging/manifests/train.jsonl
```

`train.batch_size` is per GPU. Effective batch size is
`batch_size × grad_accum_steps × GPU count`.

## Evaluate and listen

Run reconstruction metrics without configured probes:

```bash
uv run python -m eval.run_all \
  --config configs/local_6gb.yaml \
  --ckpt runs/first-run/checkpoints/last.pt \
  --manifest staging/manifests/val.jsonl \
  --out_dir runs/first-run/eval \
  --skip_probes
```

Remove `--skip_probes` to run individual probe flags enabled in the config.
Configure the train/dev manifests for every enabled probe first.

To reconstruct audio for listening:

```bash
uv run python scripts/reconstruct_audio.py \
  --config configs/local_6gb.yaml \
  --ckpt runs/first-run/checkpoints/last.pt \
  --out_dir recon_out \
  path/to/clip.wav
```

## Checkpoint sync and Kaggle

Hugging Face can carry `last.pt` between machines or notebook sessions:

```bash
uv run python scripts/housekeeping.py publish-checkpoint \
  --ckpt runs/first-run/checkpoints/last.pt --repo-id OWNER/MODEL

uv run python scripts/housekeeping.py fetch-checkpoint \
  --repo-id OWNER/MODEL --dest runs/first-run/checkpoints/last.pt
```

New repositories are created as private unless `publish-checkpoint` receives
`--public`; publishing to an existing repository keeps its current visibility.

For Kaggle, attach datasets, build manifests with repeated `--map` arguments,
and train with `configs/kaggle_3m_gan.yaml`. `scripts/kaggle_session.sh` is
currently stale because it calls manifest-cache commands that no longer exist;
use the commands above until that wrapper is repaired.

## Credentials

The CLIs read credentials from environment variables. `.env.example` lists the
available keys: Hugging Face and W&B for repository/logging access, and Kaggle
or Mozilla Data Collective for dataset downloads. The Make data targets load
`.env`; before training or direct commands, run
`set -a; source .env; set +a`.

Never commit real credentials. Any credential-like value that has previously
been committed should be rotated and removed from Git history before sharing
the repository.
