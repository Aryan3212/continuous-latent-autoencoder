#!/usr/bin/env bash
set -euo pipefail

uv run python train.py --config configs/exp2_latent_noise.yaml "$@"

