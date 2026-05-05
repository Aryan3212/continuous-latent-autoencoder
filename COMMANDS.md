# Project Commands Summary

This file provides a quick reference for the most common commands used in this repository for training, evaluation, and data preparation.

## Environment Setup

All commands assume you are using `uv`. To set up the environment:

```bash
uv venv --python 3.11
uv sync
```

---

## 1. Training

### Main Training Loop
Train the model using a YAML configuration:
```bash
uv run python train.py --config configs/exp0.yaml \
    data.train_manifest=/path/to/train.jsonl \
    data.val_manifest=/path/to/val.jsonl
```

### Weights & Biases Sweeps
1. Initialize the sweep:
   ```bash
   wandb sweep sweep.yaml
   ```
2. Start an agent:
   ```bash
   wandb agent <sweep_id>
   ```

---

## 2. Evaluation

### All-in-One Evaluation
Run reconstruction and all enabled probes (ASR, Emotion, Gender) in one go:
```bash
uv run python -m eval.run_all \
    --config configs/exp0.yaml \
    --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl \
    --out_dir runs/eval_results
```

### Waveform Reconstruction Evaluation
Compute STFT and other reconstruction metrics:
```bash
uv run python -m eval.eval_recon \
    --config configs/exp0.yaml \
    --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl \
    --out runs/recon_metrics.json
```

### Frozen-Encoder Probes
These train a small head on top of frozen latents to evaluate representation quality.

**ASR Probe (WER):**
```bash
uv run python -m eval.eval_asr \
    --config configs/exp_config.yaml \
    --ckpt /path/to/ckpt.pt \
    --train_manifest /path/to/asr_train.jsonl \
    --dev_manifest /path/to/asr_dev.jsonl \
    --out runs/asr_probe.json
```

**Emotion/Gender Probe:**
```bash
uv run python -m eval.eval_emotion --config ... (args similar to ASR)
uv run python -m eval.eval_gender --config ... (args similar to ASR)
```

---

## 3. Data Preparation & Utility Scripts

### Compute Latent Statistics
Required before training a decoder with latent normalization:
```bash
uv run python scripts/compute_latent_stats.py \
    --config configs/exp0.yaml \
    --ckpt /path/to/ckpt.pt \
    --out runs/latent_stats.pt
```

### Create JSONL Manifest from HF Parquet
Convert Hugging Face Parquet files to the internal JSONL format:
```bash
uv run python scripts/prepare_hf_parquet.py \
    --parquet_dir /path/to/parquet \
    --audio_output_dir /path/to/wav_storage \
    --output_path /path/to/manifest.jsonl \
    --split_pattern train
```

### Visualize Latents (PCA/UMAP)
Generate a 2D plot of the latent space:
```bash
uv run python scripts/visualize_latents.py \
    --config configs/exp0.yaml \
    --ckpt /path/to/ckpt.pt \
    --manifest /path/to/val.jsonl \
    --out runs/latents.png \
    --limit 500
```

### Parameter Counting
```bash
uv run python scripts/count_params.py --config configs/exp0.yaml
```

---

## 4. Verification & Testing

### Smoke Tests
Run basic functional checks for model components:
```bash
PYTHONPATH=. uv run python scripts/smoke_encoder_mhc.py
PYTHONPATH=. uv run python scripts/smoke_gan_step.py
```

### Unit Tests
```bash
uv run pytest tests/
```
