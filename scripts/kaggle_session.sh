#!/usr/bin/env bash
# One Kaggle 12h training session for the continuous-latent autoencoder.
#
# Pipeline (idempotent — safe to re-run as the next session):
#   1. install the few deps not already in the Kaggle image
#   2. build train/val manifests over the ATTACHED raw datasets (skips if present)
#   3. pull the latest checkpoint from HF (no-op on the first session)
#   4. train until the wall-clock budget, saving last.pt on the way out
#   5. publish last.pt back to HF (resume point for the next session / final model)
#
# Run from the repo root in a Kaggle notebook cell:
#   !bash scripts/kaggle_session.sh
#
# Required secrets (Kaggle "Add-ons -> Secrets", exposed as env vars in a prior
# cell, or exported inline): WANDB_API_KEY, HF_TOKEN (a WRITE token).
#
# Attach these datasets to the notebook (read-only): the Common Voice 24 Bengali
# dataset (kaggle.com/datasets/sajidullah03/common-voice-24-bn) and regspeech12
# (kaggle.com/datasets/mdrezuwanhassan/regspeech12). Then run `ls /kaggle/input`
# and adjust REGSPEECH_DIR / CV_DIR below if the mount slugs differ.
set -euo pipefail

# --- config knobs (override via env) --------------------------------------- #
CONFIG="${CONFIG:-configs/kaggle_3m_gan.yaml}"
CKPT_REPO="${HF_MODEL_REPO:-aryanrahman/clae-bengali-encoder}"
MAX_HOURS="${MAX_HOURS:-11.5}"                  # stop before Kaggle's 12h hard kill
MANIFEST_DIR="${MANIFEST_DIR:-/kaggle/working/manifests}"
CKPT="${CKPT:-/kaggle/working/runs/clae_3m_kaggle/checkpoints/last.pt}"

# Attached-dataset mount points. Adjust to match `ls /kaggle/input` if needed.
REGSPEECH_DIR="${REGSPEECH_DIR:-/kaggle/input/regspeech12}"
CV_DIR="${CV_DIR:-/kaggle/input/common-voice-24-bn}"

# --- 0. sanity: secrets ----------------------------------------------------- #
: "${HF_TOKEN:?set HF_TOKEN (a HF WRITE token) before running}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Bound glibc per-worker heap growth (the host-RAM OOM lesson from the dev box).
export MALLOC_ARENA_MAX=2

# --- 1. deps (torch/torchaudio already in the Kaggle image; don't touch them) #
# If train.py later dies on a missing import, add that package here.
pip install -q wandb pydantic pyyaml pandas openpyxl "huggingface_hub>=0.23" soundfile

# --- 2. manifests over the attached raw datasets (skip if already built) ---- #
if [[ -f "$MANIFEST_DIR/train.jsonl" ]]; then
  echo "[kaggle] manifests already present in $MANIFEST_DIR — skipping build"
else
  python scripts/housekeeping.py make-manifests \
    --map "regspeech12=$REGSPEECH_DIR" \
    --map "common_voice_bn=$CV_DIR" \
    --out-dir "$MANIFEST_DIR"
fi

# --- 3. pull latest checkpoint from HF (no-op on the first session) --------- #
python scripts/housekeeping.py fetch-checkpoint \
  --repo-id "$CKPT_REPO" --dest "$CKPT" || true

# --- 4. train (crash-safe: publish last.pt on ANY exit) --------------------- #
# An EXIT trap publishes the checkpoint whether training finishes, hits the
# --max_hours budget, or crashes (OOM etc.), so a 12h session's progress is
# never lost. This is fault tolerance for a flaky environment, not paranoia:
#   - train.py writes last.pt atomically (tmp -> rename), so we never upload a
#     half-written file;
#   - `local rc=$?` is captured first and re-raised, so a training failure still
#     surfaces as a non-zero exit instead of being masked.
# The 12h HARD kill is handled by --max_hours self-stopping FIRST — Kaggle's
# post-SIGTERM grace window is too short to rely on for a network upload.
publish_on_exit() {
  local rc=$?
  if [[ -f "$CKPT" ]]; then
    echo "[kaggle] publishing checkpoint (training exit code $rc) ..."
    python scripts/housekeeping.py publish-checkpoint \
      --ckpt "$CKPT" --repo-id "$CKPT_REPO" \
      --commit-message "kaggle session $(date -u +%Y%m%dT%H%M%SZ) rc=$rc" \
      || echo "[kaggle] WARNING: publish failed; last.pt is still at $CKPT."
  else
    echo "[kaggle] WARNING: no checkpoint at $CKPT to publish."
  fi
  exit "$rc"
}
trap publish_on_exit EXIT

resume_arg=()
if [[ -f "$CKPT" ]]; then
  echo "[kaggle] resuming from $CKPT"
  resume_arg=(--resume "$CKPT")
else
  echo "[kaggle] no checkpoint found — fresh start"
fi

python train.py --config "$CONFIG" "${resume_arg[@]}" --max_hours "$MAX_HOURS"
echo "[kaggle] training finished cleanly — EXIT trap will publish the checkpoint."
