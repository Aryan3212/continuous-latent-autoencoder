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
#   !bash scripts/kaggle_session.sh                        # defaults
#   !bash scripts/kaggle_session.sh -H 6                   # 6h session
#   !bash scripts/kaggle_session.sh -- train.batch_size=48 data.num_workers=8
#
# Options (each also settable via the env var in parens):
#   -H, --hours N      wall-clock training budget in hours (MAX_HOURS, default 11.5)
#   -c, --config PATH  training config (CONFIG, default configs/kaggle_3m_gan.yaml)
#   -h, --help         show usage and exit
# Everything after `--` is forwarded verbatim to train.py as dotted config
# overrides — use it to size train.batch_size / data.num_workers to the GPU and
# vCPUs reported at startup so a session fully utilises the hardware.
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
CKPT_REPO="${HF_MODEL_REPO:-aryan3212/clae-bengali-encoder}"
MAX_HOURS="${MAX_HOURS:-11.5}"                  # stop before Kaggle's 12h hard kill
MANIFEST_DIR="${MANIFEST_DIR:-/kaggle/working/manifests}"
# CKPT (the resume target + publish source) is resolved AFTER CLI parsing from
# the config's run.out_dir/run.run_id with the `--` overrides applied — so it
# always tracks where train.py actually writes, even when run.run_id is changed
# at launch. Set CKPT explicitly to override the resolution.
CKPT="${CKPT:-}"

# Attached-dataset mount points. Adjust to match `ls /kaggle/input` if needed.
REGSPEECH_DIR="${REGSPEECH_DIR:-/kaggle/input/regspeech12}"
CV_DIR="${CV_DIR:-/kaggle/input/common-voice-24-bn}"

# --- CLI parsing (flags override the env defaults above) -------------------- #
# Anything after `--` is forwarded to train.py as trailing dotted overrides, so
# a session can be retuned without editing the config — the lever for filling
# the GPU/CPUs reported at startup, e.g.
#   bash scripts/kaggle_session.sh -H 6 -- train.batch_size=48 data.num_workers=8
usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/kaggle_session.sh [-H HOURS] [-c CONFIG] [-- TRAIN_OVERRIDES...]
  -H, --hours N      wall-clock training budget in hours (default 11.5; env MAX_HOURS)
  -c, --config PATH  training config (default configs/kaggle_3m_gan.yaml; env CONFIG)
  -h, --help         show this help
Args after `--` are forwarded to train.py as dotted overrides, e.g.
  -- train.batch_size=48 data.num_workers=8
EOF
}
TRAIN_OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--hours)  MAX_HOURS="$2"; shift 2 ;;
    -c|--config) CONFIG="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    --)          shift; TRAIN_OVERRIDES+=("$@"); break ;;
    -*)          echo "[kaggle] unknown option: $1" >&2; usage; exit 2 ;;
    *)           echo "[kaggle] unexpected arg '$1' (forward train.py overrides after '--')" >&2; exit 2 ;;
  esac
done

# --- 0. sanity: secrets ----------------------------------------------------- #
: "${HF_TOKEN:?set HF_TOKEN (a HF WRITE token) before running}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Bound glibc per-worker heap growth (the host-RAM OOM lesson from the dev box).
export MALLOC_ARENA_MAX=2

# --- 1. deps (torch/torchaudio already in the Kaggle image; don't touch them) #
# If train.py later dies on a missing import, add that package here.
pip install -q wandb pydantic pyyaml pandas openpyxl "huggingface_hub>=0.23" soundfile

# --- 1a. resolve the checkpoint path train.py will actually write ----------- #
# Mirror train.py exactly: ckpt = <run.out_dir>/<run.run_id>/checkpoints/last.pt
# with the trailing `--` overrides applied. This is what stops the publish step
# from looking in the wrong run dir when run.run_id is overridden at launch.
# OUT_DIR is also captured so the EXIT trap can fall back to globbing for the
# newest last.pt if the exact path can't be resolved (e.g. a timestamp run_id).
read -r OUT_DIR RESOLVED_CKPT < <(python - "$CONFIG" "${TRAIN_OVERRIDES[@]}" <<'PY'
import pathlib, sys
from config import apply_overrides, load_config
cfg = apply_overrides(load_config(sys.argv[1]), sys.argv[2:])
out_dir = cfg.run.out_dir
run_id = cfg.run.run_id  # empty -> train.py picks a timestamp we can't predict
ckpt = str(pathlib.Path(out_dir) / run_id / "checkpoints" / "last.pt") if run_id else "-"
print(out_dir, ckpt)
PY
)
[[ -z "${CKPT:-}" && "$RESOLVED_CKPT" != "-" ]] && CKPT="$RESOLVED_CKPT"
echo "[kaggle] checkpoint path: ${CKPT:-<unresolved; will glob $OUT_DIR on exit>}"

# --- 1b. report hardware so batch_size / num_workers can be sized to it ------ #
# Kaggle's T4/P100 (~15-16 GB) and ~30 GB host RAM dwarf the dev box, so the
# base config's small batch leaves the GPU mostly idle. Read the numbers below,
# then raise `train.batch_size` (VRAM) and `data.num_workers` (vCPUs) via `--`.
echo "[kaggle] hardware:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
  | sed 's/^/[kaggle]   gpu: /' || echo "[kaggle]   nvidia-smi unavailable"
python - <<'PY'
import os
print(f"[kaggle]   cpus: {os.cpu_count()}  (set data.num_workers near this)")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[kaggle]   torch: {torch.cuda.device_count()}x {p.name} "
              f"({p.total_memory / 1e9:.1f} GB) — raise train.batch_size until ~90% VRAM")
except Exception as e:  # torch import shouldn't fail on Kaggle, but don't abort
    print(f"[kaggle]   torch GPU query skipped: {e}")
PY

# --- 2. manifests over the attached raw datasets (skip if already built) ---- #
# Building manifests over the full corpus (~1M Common Voice clips) costs minutes
# every ephemeral session, so cache the deterministic output: use a local copy
# if present, else restore the gzipped cache from HF, else build once and
# publish it for the next session. The manifests embed absolute mount paths, so
# the cache is valid only while the same datasets stay attached at the same
# slugs — set REBUILD_MANIFESTS=1 to force a rebuild + re-publish if that changes
# (or attach a Kaggle Dataset of prebuilt manifests and point MANIFEST_DIR at it).
if [[ -z "${REBUILD_MANIFESTS:-}" && -f "$MANIFEST_DIR/train.jsonl" ]]; then
  echo "[kaggle] manifests already present in $MANIFEST_DIR — skipping build"
elif [[ -z "${REBUILD_MANIFESTS:-}" ]] && python scripts/housekeeping.py fetch-manifests \
       --repo-id "$CKPT_REPO" --dest-dir "$MANIFEST_DIR"; then
  echo "[kaggle] restored cached manifests from $CKPT_REPO"
else
  echo "[kaggle] building manifests from raw (slow, once) ..."
  python scripts/housekeeping.py make-manifests \
    --map "regspeech12=$REGSPEECH_DIR" \
    --map "common_voice_bn=$CV_DIR" \
    --out-dir "$MANIFEST_DIR"
  echo "[kaggle] publishing manifests to $CKPT_REPO for reuse next session ..."
  python scripts/housekeeping.py publish-manifests \
    --manifest-dir "$MANIFEST_DIR" --repo-id "$CKPT_REPO" \
    || echo "[kaggle] WARNING: manifest publish failed; will rebuild next session."
fi

# --- 3. pull latest checkpoint from HF (no-op on the first session) --------- #
if [[ -n "${CKPT:-}" ]]; then
  python scripts/housekeeping.py fetch-checkpoint \
    --repo-id "$CKPT_REPO" --dest "$CKPT" || true
else
  echo "[kaggle] run_id unresolved — skipping HF fetch (fresh start)"
fi

# --- 4. train (crash-safe: publish last.pt on ANY exit) --------------------- #
# An EXIT trap publishes the checkpoint whether training finishes, hits the
# --max_hours budget, or crashes (OOM etc.), so a 12h session's progress is
# never lost:
#   - train.py writes last.pt atomically (tmp -> rename), so we never upload a
#     half-written file;
#   - `local rc=$?` is captured first and re-raised, so a training failure still
#     surfaces as a non-zero exit instead of being masked.
# The 12h HARD kill is handled by --max_hours self-stopping FIRST — Kaggle's
# post-SIGTERM grace window is too short to rely on for a network upload.
# Periodic snapshot publisher (background): every SNAPSHOT_EVERY_SECS, publish the
# newest last.pt. publish-checkpoint dual-uploads, so each run drops a permanent
# step-<step>.pt alongside the rolling last.pt — mid-session insurance beyond the
# single exit-time publish. Skipped if the checkpoint hasn't changed since last
# push (no new save_interval write), so an idle interval is a no-op, not a commit.
SNAPSHOT_EVERY_SECS="${SNAPSHOT_EVERY_SECS:-1800}"   # 30 min; set 0 to disable
snapshot_loop() {
  local last_mtime=""
  while sleep "$SNAPSHOT_EVERY_SECS"; do
    local ck m
    ck="$(ls -t "$OUT_DIR"/*/checkpoints/last.pt 2>/dev/null | head -n1 || true)"
    [[ -n "$ck" && -f "$ck" ]] || continue
    m="$(stat -c %Y "$ck" 2>/dev/null || stat -f %m "$ck" 2>/dev/null || echo "")"
    [[ -n "$m" && "$m" == "$last_mtime" ]] && continue
    last_mtime="$m"
    echo "[kaggle] periodic snapshot -> $CKPT_REPO ($ck)"
    python scripts/housekeeping.py publish-checkpoint \
      --ckpt "$ck" --repo-id "$CKPT_REPO" \
      --commit-message "periodic snapshot $(date -u +%Y%m%dT%H%M%SZ)" \
      || echo "[kaggle] periodic snapshot publish failed (continuing)"
  done
}
SNAP_PID=""
if [[ "$SNAPSHOT_EVERY_SECS" -gt 0 ]] 2>/dev/null; then
  snapshot_loop & SNAP_PID=$!
  echo "[kaggle] periodic snapshots every ${SNAPSHOT_EVERY_SECS}s (pid $SNAP_PID)"
fi

publish_on_exit() {
  local rc=$?
  [[ -n "$SNAP_PID" ]] && kill "$SNAP_PID" 2>/dev/null || true
  local ckpt="$CKPT"
  # Fall back to the newest last.pt under OUT_DIR if the resolved path is
  # missing (unresolved run_id, or a path drift we didn't anticipate) — better
  # to publish a slightly different run dir than to lose the session entirely.
  if [[ ! -f "$ckpt" && -d "$OUT_DIR" ]]; then
    ckpt="$(ls -t "$OUT_DIR"/*/checkpoints/last.pt 2>/dev/null | head -n1 || true)"
    [[ -n "$ckpt" ]] && echo "[kaggle] resolved path missing; falling back to $ckpt"
  fi
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    echo "[kaggle] publishing checkpoint (training exit code $rc) ..."
    python scripts/housekeeping.py publish-checkpoint \
      --ckpt "$ckpt" --repo-id "$CKPT_REPO" \
      --commit-message "kaggle session $(date -u +%Y%m%dT%H%M%SZ) rc=$rc" \
      || echo "[kaggle] WARNING: publish failed; last.pt is still at $ckpt."
  else
    echo "[kaggle] WARNING: no checkpoint found (looked at '$CKPT' and under '$OUT_DIR') to publish."
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

# Multi-GPU -> DDP via torchrun (train.py reads RANK/WORLD_SIZE/LOCAL_RANK from the
# env and shards across the GPUs); single-GPU -> plain python (unchanged path).
# NOTE: under DDP train.batch_size is PER-GPU, so effective batch =
# batch_size * grad_accum_steps * NPROC. Force single-GPU with NPROC=1.
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" =~ ^[0-9]+$ ]] && (( NPROC >= 1 )) || NPROC=1
if (( NPROC > 1 )); then
  echo "[kaggle] launching DDP across $NPROC GPUs (torchrun)"
  launcher=(torchrun --standalone --nproc_per_node="$NPROC")
else
  echo "[kaggle] launching single-GPU"
  launcher=(python)
fi

"${launcher[@]}" train.py --config "$CONFIG" "${resume_arg[@]}" --max_hours "$MAX_HOURS" "${TRAIN_OVERRIDES[@]}"
echo "[kaggle] training finished cleanly — EXIT trap will publish the checkpoint."
