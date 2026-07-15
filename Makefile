CONFIG        ?= configs/local_6gb.yaml
OUTPUT_DIR    ?= runs
DATA_ROOT     ?= $(CURDIR)/datasets
MANIFEST_DIR  ?= staging/manifests
DATASETS      ?= openslr53,common_voice_bn,regspeech12,indicvoices,subak_ko,shrutilipi,kathbath
TRAIN_EXTRA_ARGS ?=

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help prepare download-data make-manifests train clean-runs

help:
	@echo "Targets:"
	@echo "  prepare         Install deps via uv sync."
	@echo "  download-data   Download raw source datasets to DATA_ROOT."
	@echo "  make-manifests  Build train/val JSONL manifests from downloaded data."
	@echo "  train           Train the autoencoder (point MANIFEST_DIR at your manifests)."
	@echo "  clean-runs      Delete runs/* (with confirmation)."
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  CONFIG           = $(CONFIG)"
	@echo "  OUTPUT_DIR       = $(OUTPUT_DIR)"
	@echo "  DATA_ROOT        = $(DATA_ROOT)"
	@echo "  MANIFEST_DIR     = $(MANIFEST_DIR)"
	@echo "  DATASETS         = $(DATASETS)"
	@echo "  TRAIN_EXTRA_ARGS = $(TRAIN_EXTRA_ARGS)"
	@echo "  RUN_NAME         = (auto: run-YYYYMMDD-HHMMSS if unset)"

prepare:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir $$HOME/.local/bin --force)
	uv sync

download-data: prepare
	uv run python scripts/housekeeping.py download --datasets $(DATASETS) --data-root $(DATA_ROOT)

make-manifests: prepare
	uv run python scripts/housekeeping.py make-manifests \
	    --data-root $(DATA_ROOT) \
	    --datasets $(DATASETS) \
	    --out-dir $(MANIFEST_DIR)

train: prepare
	@set -e; \
	run_name=$${RUN_NAME:-run-$$(date +%Y%m%d-%H%M%S)}; \
	echo "[make] run_name=$$run_name config=$(CONFIG)"; \
	uv run python train.py \
	    --config $(CONFIG) \
	    data.train_manifest=$(MANIFEST_DIR)/train.jsonl \
	    data.val_manifest=$(MANIFEST_DIR)/val.jsonl \
	    run.run_id=$$run_name \
	    run.out_dir=$(OUTPUT_DIR) \
	    $(TRAIN_EXTRA_ARGS)

clean-runs:
	@read -p "delete all runs under $(OUTPUT_DIR)/? [y/N] " ok; \
	if [ "$$ok" = "y" ] || [ "$$ok" = "Y" ]; then \
	    rm -rf $(OUTPUT_DIR)/*; \
	    echo "[make] cleaned $(OUTPUT_DIR)/"; \
	else \
	    echo "[make] aborted"; \
	fi
