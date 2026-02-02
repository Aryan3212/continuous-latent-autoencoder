#!/usr/bin/env bash
set -euo pipefail

# Example pipeline:
# 1) VAD segment unlabeled audio list into 2-8s segments
# 2) Filter by simple quality metrics
# 3) Compute seed index stats from seed manifest using current encoder
# 4) Mine candidates by Mahalanobis distance to seed distribution
#
# Required args (as env vars or CLI overrides):
#   SEED_MANIFEST, UNLABELED_MANIFEST, CKPT, OUT_DIR

OUT_DIR="${OUT_DIR:-runs/round1_mining}"
mkdir -p "$OUT_DIR"

uv run python -m data.vad_segment --in_manifest "$UNLABELED_MANIFEST" --out_manifest "$OUT_DIR/unlabeled_vad.jsonl"
uv run python -m data.quality_filter --in_manifest "$OUT_DIR/unlabeled_vad.jsonl" --out_manifest "$OUT_DIR/unlabeled_filtered.jsonl" --reject_manifest "$OUT_DIR/unlabeled_rejected.jsonl"

uv run python -m data.embed_index --config configs/exp0.yaml --ckpt "$CKPT" --seed_manifest "$SEED_MANIFEST" --out_npz "$OUT_DIR/seed_index.npz"
uv run python -m data.mine_unlabeled --config configs/exp0.yaml --ckpt "$CKPT" --candidates_manifest "$OUT_DIR/unlabeled_filtered.jsonl" --seed_index_npz "$OUT_DIR/seed_index.npz" --out_manifest "$OUT_DIR/round1_selected.jsonl" --out_stats "$OUT_DIR/round1_stats.json"

