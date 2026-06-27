# CLAUDE.md

See `AGENTS.md` for additional agent instructions and `CODEBASE.md` for codebase orientation.

## Hardware / environment

- This dev box has **one NVIDIA GeForce GTX 1660 Super (6 GB VRAM)** — a small,
  consumer Turing card. It is **not** the machine for full training runs; use it
  for smoke tests, feature extraction, and probe-sized evals only.
- **fp16 AMP is unreliable on this card** (black-screen / hard lockups). Run eval
  in fp32 with `CLAE_AMP=0`, and train with `run.amp=false`. Keep batch sizes and
  `--max_samples` small to fit 6 GB.
- Python is managed by **uv** — run things with `uv run python ...` (the venv is
  `.venv/`, Python 3.11).

## Before running any GPU / ML command

Do **not** assume the toolchain is present. Before training, eval, or anything
that imports torch/CUDA, first confirm availability rather than guessing — e.g.:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If Python, torch, or CUDA is missing/unavailable, ask the user how they want to
proceed (CPU fallback, install, or skip) instead of running blindly.
