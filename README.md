# continuous-latent-autoencoder

Deterministic continuous-latent speech autoencoder (16kHz in → ~12.5 Hz tokens). Config-driven; `schema.py` is the single source of truth (`extra="forbid"`).

> Security: before sharing this repo, rotate HF/Kaggle/WandB tokens and purge them from git history (old keys remain in history).

## Setup

```bash
uv sync
# build manifests once (see scripts/housekeeping.py make-manifests --help)
python scripts/housekeeping.py make-manifests --data-root <DATA_ROOT> --datasets openslr53,bengaliai_speech --out-dir data/manifests
```

## Train (actual run)

`configs/large_2kh.yaml` = mel recon + JePA + VISReg, FastConformer encoder, GAN off.

```bash
uv run python train.py --config configs/large_2kh.yaml
```

`train.py` prints a per-block parameter breakdown at startup.

## Smoke test (GAN path)

Short run that flips the adversarial loss on early to verify the discriminator / feature-matching path doesn't crash or NaN:

```bash
uv run python train.py --config configs/large_2kh.yaml \
  loss.adv.enabled=true loss.adv.adv_start_step=20 loss.adv.fm_start_step=20 \
  train.max_steps=200 train.batch_size=4 train.eval_interval_steps=200 \
  train.save_interval_steps=200 run.wandb.enabled=false
```

## Eval (frozen checkpoint)

```bash
# reconstruction
uv run python -m eval.eval_recon --config configs/large_2kh.yaml --ckpt runs/<run>/checkpoints/last.pt --manifest data/manifests/val.jsonl --out runs/recon.json
# frozen-encoder ASR probe (CTC head -> WER)
uv run python -m eval.eval_asr --config configs/large_2kh.yaml --ckpt runs/<run>/checkpoints/last.pt --train_manifest data/manifests/asr_probe_train.jsonl --dev_manifest data/manifests/asr_probe_val.jsonl --out runs/asr_probe.json
```
