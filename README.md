# Continuous Latent Autoencoder
HF Spaces Demo: https://huggingface.co/spaces/aryan3212/clae-demo

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
make make-manifests DATASETS=openslr53 HOUSEKEEPING_WORKERS=4
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
`MANIFEST_DIR`, `DATASETS`, or `HOUSEKEEPING_WORKERS` when needed.

`make make-manifests` is the combined preparation path: it downloads missing
datasets concurrently, materializes records from different sources in
parallel, and atomically publishes both manifests. Completed ZIP/TAR archives
are removed after verified extraction. Hugging Face parquet shards are removed
only after extracted audio and an atomic `.records.jsonl` metadata cache are in
place, so repeat manifest builds remain fast and do not re-download data.
Use `make download-data` only when you explicitly want a download-only prefetch.

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

### Optional packed training inventory

The existing combined training manifest is the authoritative inventory for
optional packed storage; do not re-enumerate datasets or create new splits.
On the machine that holds the audio, create uncompressed TAR shards containing
16 kHz mono PCM16 FLAC members directly from that manifest:

```bash
uv run python scripts/prepare_audio_shards.py pack \
  --manifest staging/manifests/train.jsonl \
  --output-dir staging/packed/train \
  --sample-rate 16000 \
  --target-shard-size-gb 1.0 \
  --workers 4 \
  --seed 42 \
  --resume
```

The command stores `shard_manifest.json`, `index.jsonl`, and uncompressed TARs
under `shards/`. It preserves full utterances and original row metadata, mixes
the existing rows with a deterministic seed, records PCM16 quantization
statistics, and structurally verifies the finished output. Non-finite audio,
decode/encode failures, missing sources, and I/O failures still stop packing;
they are never silently dropped. A finite waveform above PCM16 full scale is
instead stored with a reversible per-sample `amplitude_restore_gain` and
explicit headroom. The packed streaming loader restores that gain before its
normal training preprocessing, so this is storage representation only—not
loudness normalization or a training-distribution change. Every 30 seconds the
producer reports records, rate, ETA, audio hours, current shard size, and
scaling totals; adjust it with `--progress-interval-seconds`.

`--resume` starts a new pack when the output directory is absent or empty;
otherwise it accepts only the same interrupted manifest fingerprint and pack
settings. Interrupted output made by the original v1 producer (which stopped
on a peak above one) is migrated in place: only its recorded active partial
shard is removed, finalized legacy shards/index parts remain intact, and their
implicit restore gain is one. Verify later without the source datasets using:

```bash
uv run python scripts/prepare_audio_shards.py verify \
  --output-dir staging/packed/train
```

When the original source files are still mounted, compare a deterministic
sample of the training-time TAR decode against the exact source load → mono →
resample path. This is read-only; run it while the training loader is idle so
it does not contend for disk bandwidth:

```bash
uv run python scripts/prepare_audio_shards.py audit \
  --output-dir staging/packed/train \
  --samples 256 \
  --seed 42
```

The command scopes its default sample to one deterministic random shard, so it
does not scan the whole archive set. Increase `--shards` after an initial pass.
It prints aggregate max/RMS waveform error and names every sampled record above
its `--max-abs-error` threshold (default `1e-3`).

This producer is backward-compatible with the current file-backed training
workflow. To train from the packed representation, use the compatible
`large_2kh_packed.yaml` variant and the same step-40k checkpoint. It leaves
the model, loss, optimizer, and schedule configuration unchanged while using
whole-shard streaming, rank/worker-safe assignment, and buffered sample
shuffle. The configured four persistent workers and 512 MiB compressed buffer
per worker are an HDD-oriented starting point; adjust only the data-loader
knobs after measuring the remote machine. The buffer is a compressed-byte
budget measured from the actual buffered FLAC and JSON member payloads, plus at
most one unusually large selected pair.

```bash
uv run python train.py \
  --config configs/large_2kh_packed.yaml \
  --resume /path/to/step_040000.pt \
  run.run_id=large-2kh-packed-resume
```

`data.backend=files` remains the default and preserves the existing
file-manifest loader exactly. For `data.backend=tar`, `data.train_manifest`
remains required provenance while `data.shard_manifest` selects the packed
inventory. The loader does not read `index.jsonl` during training. Each packed
epoch uses a fresh deterministic shard order, assigns every shard to one global
rank/worker consumer, emits an equal complete-batch quota per consumer, and
starts a fresh randomized epoch after resume rather than replaying a sample
position. Legacy shards without storage-scaling fields imply gain 1; newer
shards restore their validated per-sample gain before crop/pad, so the file
backend and effective training amplitude remain unchanged.

The producer accepts 1–8 packing workers (four is the HDD-safe starting point)
and pins Torch intra-op work to one thread per worker to avoid CPU
oversubscription.

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

## Checkpoint sync

Hugging Face can carry `last.pt` between machines or notebook sessions:

```bash
uv run python scripts/housekeeping.py publish-checkpoint \
  --ckpt runs/first-run/checkpoints/last.pt --repo-id OWNER/MODEL

uv run python scripts/housekeeping.py fetch-checkpoint \
  --repo-id OWNER/MODEL --dest runs/first-run/checkpoints/last.pt
```

New repositories are created as private unless `publish-checkpoint` receives
`--public`; publishing to an existing repository keeps its current visibility.

## Credentials

The data CLI, `train.py`, and the representation-evaluation adapters
automatically load the repository's gitignored `.env`; already-exported
environment variables take precedence. `.env.example` lists the available keys:
`HF_TOKEN`, `WANDB_API_KEY`, `MDC_API_KEY`, and the Kaggle credentials. Hugging
Face auth is optional for public, ungated datasets but required for gated
resources and publishing. The emotion2vec adapter additionally uses FunASR,
which is installed with the project. W&B remains optional.

Never commit real credentials. Any credential-like value that has previously
been committed should be rotated and removed from Git history before sharing
the repository.
