#!/usr/bin/env bash
# Fresh GPU VM bootstrap: deps -> data -> training.
#
# Flow on a newly provisioned instance:
#   git clone <repo> && cd continuous-latent-autoencoder
#   cp .env.example .env && $EDITOR .env     # paste tokens (or export the same vars)
#   tmux new -s train                        # training outlives the SSH session
#   ./setup.sh                               # deps + creds + fetch + train
#
# Flags:
#   --no-train   stop after the dataset fetch (inspect manifests first)
#
# Idempotent: uv sync is incremental; data adapters keep completion markers or
# extracted-record caches. Credentials are read from the environment (sourced
# from .env below) — no generated creds file.
set -euo pipefail
cd "$(dirname "$0")"

NO_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --no-train) NO_TRAIN=1 ;;
        *) echo "[setup] unknown arg: $arg (supported: --no-train)"; exit 1 ;;
    esac
done

# --- 0. Config: .env file, exported vars win over file values --------------
if [ -f .env ]; then
    echo "[setup] loading .env"
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

: "${DATA_ROOT:=$PWD/datasets}"
: "${HF_MODEL_REPO:=aryanrahman/clae-bengali-encoder}"
: "${MANIFEST_DIR:=$PWD/staging/manifests}"
: "${HOUSEKEEPING_WORKERS:=4}"
: "${CONFIG:=configs/exp0.yaml}"

# --- 1. Preflight -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    echo "[setup] WARNING: nvidia-smi not found — train.py requires CUDA and will fail."
fi
if [ -z "${TMUX:-}" ] && [ -t 1 ]; then
    echo "[setup] NOTE: not inside tmux/screen — training dies with your SSH session."
fi

# --- 2. Dependencies (uv + locked venv) -------------------------------------
make prepare

# Credentials/config come from the environment (the .env sourced above and
# re-exported via `set -a`). The housekeeping CLI and train.py read them directly.
export DATA_ROOT HF_MODEL_REPO MANIFEST_DIR HOUSEKEEPING_WORKERS

# --- 3. W&B auth (train.py's wandb.init reads the env) ----------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
else
    export WANDB_MODE=offline
    echo "[setup] WANDB_API_KEY not set -> WANDB_MODE=offline (metrics stay in runs/<id>/logs)"
fi

# --- 4. Data: parallel download/materialization + atomic manifests -----------
make make-manifests \
    DATA_ROOT="$DATA_ROOT" \
    MANIFEST_DIR="$MANIFEST_DIR" \
    HOUSEKEEPING_WORKERS="$HOUSEKEEPING_WORKERS" \
    DATASETS="${DATASETS:-openslr53,common_voice_bn,regspeech12,indicvoices,shrutilipi}"

for m in train.jsonl val.jsonl; do
    if [ ! -s "$MANIFEST_DIR/$m" ]; then
        echo "[setup] ERROR: $MANIFEST_DIR/$m missing or empty after preparation."
        exit 1
    fi
done
echo "[setup] manifests: $(wc -l < "$MANIFEST_DIR/train.jsonl") train rows," \
     "$(wc -l < "$MANIFEST_DIR/val.jsonl") val rows"

# --- 5. Train ----------------------------------------------------------------
if [ "$NO_TRAIN" = "1" ]; then
    echo "[setup] --no-train: done. Start later with:"
    echo "        make train MANIFEST_DIR=$MANIFEST_DIR CONFIG=$CONFIG"
    exit 0
fi

exec make train \
    DATA_ROOT="$DATA_ROOT" \
    MANIFEST_DIR="$MANIFEST_DIR" \
    CONFIG="$CONFIG" \
    TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
