#!/usr/bin/env bash
set -euo pipefail

# Train on a mixture of seed + selected unlabeled. You control mixing by building
# a combined manifest externally (or by sampling in a custom dataset later).
#
# Required env vars:
#   TRAIN_MANIFEST, VAL_MANIFEST

uv run python train.py --config configs/exp1_mix.yaml \
  data.train_manifest="$TRAIN_MANIFEST" \
  data.val_manifest="$VAL_MANIFEST" \
  "$@"

