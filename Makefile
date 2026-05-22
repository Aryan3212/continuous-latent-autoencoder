# =============================================================================
# CLAE — One-command training on a fresh cloud-GPU instance.
#
# `make all` does: fetch-data -> train -> evaluate -> publish.
# Override any var on the command line, e.g.:
#     RUN_NAME=clae-2026-05-22 make train
#     DATASETS=openslr53 make pack-and-push
#     CONFIG=configs/exp1.yaml make train
#
# `make help` lists targets + variables.
# =============================================================================

# --- Overridable variables (use `?=` so command-line / env wins) ---
CONFIG          ?= configs/exp0.yaml
OUTPUT_DIR      ?= runs
CLAE_DATA_ROOT  ?= $(HOME)/data/clae
CLAE_HF_REPO    ?= aryanrahman/clae-bengali
CLAE_CKPT_REPO  ?= aryanrahman/clae-bengali-encoder
DATASETS        ?= openslr53,bengaliai_speech,regspeech12,indicvoices,subak_ko,shrutilipi,kathbath
TRAIN_EXTRA_ARGS ?=

# Ensure local user-installed binaries (uv) are on PATH for `uv run`.
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help ensure-uv prepare fetch-data pack-and-push train evaluate publish all clean-runs

help:
	@echo "CLAE — one-command training."
	@echo ""
	@echo "Targets:"
	@echo "  prepare         Install deps via uv sync; print key data paths."
	@echo "  fetch-data      Snapshot-download the packed HF dataset to CLAE_DATA_ROOT."
	@echo "  pack-and-push   (prep instance) Build packed dataset and push to HF Hub."
	@echo "  train           Train (depends on fetch-data, ensure-uv)."
	@echo "  evaluate        Run ASR probe against the most recent checkpoint."
	@echo "  publish         Upload the most recent checkpoint to HF Hub."
	@echo "  all             train -> evaluate -> publish."
	@echo "  clean-runs      Delete runs/* (with confirmation)."
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  CONFIG           = $(CONFIG)"
	@echo "  OUTPUT_DIR       = $(OUTPUT_DIR)"
	@echo "  CLAE_DATA_ROOT   = $(CLAE_DATA_ROOT)"
	@echo "  CLAE_HF_REPO     = $(CLAE_HF_REPO)"
	@echo "  CLAE_CKPT_REPO   = $(CLAE_CKPT_REPO)"
	@echo "  DATASETS         = $(DATASETS)"
	@echo "  TRAIN_EXTRA_ARGS = $(TRAIN_EXTRA_ARGS)"
	@echo "  RUN_NAME         = (auto: clae-YYYYMMDD-HHMMSS if unset)"

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir $$HOME/.local/bin --force)

prepare: ensure-uv
	uv sync
	@echo ""
	@echo "Data paths:"
	@echo "  CLAE_DATA_ROOT = $(CLAE_DATA_ROOT)"
	@echo "  CLAE_HF_REPO   = $(CLAE_HF_REPO)"
	@echo "  CLAE_CKPT_REPO = $(CLAE_CKPT_REPO)"

# Idempotent: huggingface_hub.snapshot_download verifies LFS pointers without
# re-downloading existing files.
fetch-data: ensure-uv
	uv run python -m clae_data fetch --repo-id $(CLAE_HF_REPO) --dest $(CLAE_DATA_ROOT)

# Prep instance: download raw archives -> transcode -> push to HF Hub.
pack-and-push: ensure-uv
	uv run python -m clae_data pack-and-push --datasets $(DATASETS) --repo-id $(CLAE_HF_REPO)

train: fetch-data ensure-uv
	@set -e; \
	run_name=$${RUN_NAME:-clae-$$(date +%Y%m%d-%H%M%S)}; \
	echo "[make] run_name=$$run_name config=$(CONFIG)"; \
	uv run python train.py \
	    --config $(CONFIG) \
	    data.train_manifest=$(CLAE_DATA_ROOT)/manifests/train.jsonl \
	    data.val_manifest=$(CLAE_DATA_ROOT)/manifests/val.jsonl \
	    run.run_id=$$run_name \
	    run.out_dir=$(OUTPUT_DIR) \
	    $(TRAIN_EXTRA_ARGS)

evaluate: ensure-uv
	@set -e; \
	latest=$$(ls -1t $(OUTPUT_DIR)/*/checkpoints/last.pt 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then echo "[make] no checkpoint found under $(OUTPUT_DIR)/*/checkpoints/last.pt"; exit 1; fi; \
	out_dir=$$(dirname $$latest)/../eval; \
	mkdir -p $$out_dir; \
	echo "[make] evaluating $$latest"; \
	uv run python -m eval.eval_asr \
	    --config $(CONFIG) \
	    --ckpt $$latest \
	    --train_manifest $(CLAE_DATA_ROOT)/manifests/asr_probe_train.jsonl \
	    --dev_manifest $(CLAE_DATA_ROOT)/manifests/asr_probe_val.jsonl \
	    --out $$out_dir/asr.json

publish: ensure-uv
	@set -e; \
	latest=$$(ls -1t $(OUTPUT_DIR)/*/checkpoints/last.pt 2>/dev/null | head -n1); \
	if [ -z "$$latest" ]; then echo "[make] no checkpoint found under $(OUTPUT_DIR)/*/checkpoints/last.pt"; exit 1; fi; \
	echo "[make] publishing $$latest -> $(CLAE_CKPT_REPO)"; \
	uv run python -m clae_data publish-checkpoint --ckpt $$latest --repo-id $(CLAE_CKPT_REPO)

all: train evaluate publish

clean-runs:
	@read -p "delete all runs under $(OUTPUT_DIR)/? [y/N] " ok; \
	if [ "$$ok" = "y" ] || [ "$$ok" = "Y" ]; then \
	    rm -rf $(OUTPUT_DIR)/*; \
	    echo "[make] cleaned $(OUTPUT_DIR)/"; \
	else \
	    echo "[make] aborted"; \
	fi
