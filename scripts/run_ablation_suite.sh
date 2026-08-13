#!/usr/bin/env bash
# Train the packed-base reference and five 25k ablations serially, then evaluate
# all six checkpoints once. Re-running resumes the current condition and skips
# completed runs.

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_ablation_suite.sh [TRAIN_CONFIG_OVERRIDE ...]

Environment:
  ABLATION_OUT_DIR         Output root (default: runs/ablations)
  ABLATION_GPUS            GPUs/processes per run (default: 1)
  ABLATION_EVAL_MANIFEST   Validation manifest for reconstruction evaluation;
                           required unless reconstruction is skipped.
  ABLATION_EVAL_SKIP_RECON Requires ABLATION_EVAL_MIMI=0 and
                           ABLATION_EVAL_LISTENING=0 when set to 1
  ABLATION_EVAL_SUBESCO_DIR Explicit SUBESCO root for temporal emotion and
                           MOS-colored representation plots
  ABLATION_EVAL_*          Evaluation options; see run_ablation_evals.sh --help

Examples:
  ABLATION_EVAL_MANIFEST=staging/manifests/val.jsonl ABLATION_EVAL_SUBESCO_DIR=datasets/SUBESCO scripts/run_ablation_suite.sh
  ABLATION_GPUS=2 ABLATION_EVAL_MANIFEST=staging/manifests/val.jsonl scripts/run_ablation_suite.sh
  ABLATION_EVAL_SKIP_RECON=1 ABLATION_EVAL_MIMI=0 ABLATION_EVAL_LISTENING=0 scripts/run_ablation_suite.sh run.wandb.enabled=false
EOF
  exit 0
fi

for override in "$@"; do
  case "$override" in
    run.run_id=*|run.out_dir=*|train.max_steps=*|train.save_interval_steps=*|optim.scheduler.total_steps=*)
      echo "[ablation-suite] override is managed by the suite and cannot be changed: $override" >&2
      exit 2
      ;;
  esac
done

skip_recon="${ABLATION_EVAL_SKIP_RECON:-0}"
case "$skip_recon" in
  0)
    if [[ -z "${ABLATION_EVAL_MANIFEST:-}" ]]; then
      echo "[ablation-suite] set ABLATION_EVAL_MANIFEST to a validation manifest, or set ABLATION_EVAL_SKIP_RECON=1 before starting" >&2
      exit 2
    fi
    if [[ ! -f "$ABLATION_EVAL_MANIFEST" ]]; then
      echo "[ablation-suite] validation manifest does not exist: $ABLATION_EVAL_MANIFEST" >&2
      exit 2
    fi
    ;;
  1) ;;
  *)
    echo "[ablation-suite] ABLATION_EVAL_SKIP_RECON must be 0 or 1" >&2
    exit 2
    ;;
esac
for eval_boolean in ABLATION_EVAL_MIMI ABLATION_EVAL_LISTENING; do
  default_value=1
  if [[ "$eval_boolean" == "ABLATION_EVAL_LISTENING" ]]; then
    default_value=0
  fi
  case "${!eval_boolean:-$default_value}" in
    0|1) ;;
    *) echo "[ablation-suite] $eval_boolean must be 0 or 1" >&2; exit 2 ;;
  esac
done
if [[ "$skip_recon" == "1" ]]; then
  if [[ "${ABLATION_EVAL_MIMI:-1}" != "0" || "${ABLATION_EVAL_LISTENING:-0}" != "0" ]]; then
    echo "[ablation-suite] skipping reconstruction requires ABLATION_EVAL_MIMI=0 and ABLATION_EVAL_LISTENING=0" >&2
    exit 2
  fi
fi
if [[ -n "${ABLATION_EVAL_SUBESCO_DIR:-}" && ! -d "$ABLATION_EVAL_SUBESCO_DIR" ]]; then
  echo "[ablation-suite] SUBESCO directory does not exist: $ABLATION_EVAL_SUBESCO_DIR" >&2
  exit 2
fi
if [[ -n "${UTMOSV2_CHECKPOINT:-}" && ! -f "$UTMOSV2_CHECKPOINT" ]]; then
  echo "[ablation-suite] UTMOSV2_CHECKPOINT does not exist: $UTMOSV2_CHECKPOINT" >&2
  exit 2
fi
for positive_eval_var in ABLATION_EVAL_RECON_BATCH ABLATION_EVAL_RECON_BATCHES ABLATION_EVAL_LISTENING_N; do
  value="${!positive_eval_var:-}"
  if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ablation-suite] $positive_eval_var must be a positive integer" >&2
    exit 2
  fi
done
eval_segment_seconds="${ABLATION_EVAL_SEGMENT_SECONDS:-3.0}"
if [[ ! "$eval_segment_seconds" =~ ^[0-9]+([.][0-9]+)?$ || "$eval_segment_seconds" =~ ^0+([.]0+)?$ ]]; then
  echo "[ablation-suite] ABLATION_EVAL_SEGMENT_SECONDS must be positive" >&2
  exit 2
fi
eval_recon_budget="$(( ${ABLATION_EVAL_RECON_BATCH:-8} * ${ABLATION_EVAL_RECON_BATCHES:-50} ))"
if [[ "${ABLATION_EVAL_LISTENING:-0}" == "1" && "${ABLATION_EVAL_LISTENING_N:-20}" -gt "$eval_recon_budget" ]]; then
  echo "[ablation-suite] ABLATION_EVAL_LISTENING_N cannot exceed the reconstruction batch budget ($eval_recon_budget)" >&2
  exit 2
fi

out_dir="${ABLATION_OUT_DIR:-runs/ablations}"
gpu_count="${ABLATION_GPUS:-1}"
if [[ ! "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ablation-suite] ABLATION_GPUS must be a positive integer" >&2
  exit 2
fi

case "$out_dir" in
  /*) resolved_out_dir="$out_dir" ;;
  *) resolved_out_dir="$repo_root/$out_dir" ;;
esac

configs=(
  "configs/large_2kh_ablation_recon_only_50k.yaml"
  "configs/large_2kh_ablation_repr_only_50k.yaml"
  "configs/large_2kh_ablation_no_mhc_50k.yaml"
  "configs/large_2kh_ablation_25hz_50k.yaml"
  "configs/large_2kh_ablation_no_decoder_corruption_50k.yaml"
  "configs/large_2kh_packed_25k.yaml"
)

run_ids=(
  "large-2kh-ablation-recon-only-r-50k"
  "large-2kh-ablation-repr-only-j-visreg-50k"
  "large-2kh-ablation-no-mhc-50k"
  "large-2kh-ablation-25hz-full-50k"
  "large-2kh-ablation-no-decoder-corruption-50k"
  "large-2kh-packed-25k"
)

checkpoint_is_intact() {
  local checkpoint="$1"
  [[ -f "$checkpoint" && -s "$checkpoint" ]] || return 1

  # Current torch.save checkpoints are ZIP archives. Test their CRCs so a
  # partially copied last.pt is never selected. Retain support for legacy
  # non-ZIP PyTorch checkpoints, whose semantic validation remains train.py's.
  if [[ "$(LC_ALL=C od -An -tx1 -N2 "$checkpoint" | tr -d '[:space:]')" == "504b" ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -tqq "$checkpoint" >/dev/null 2>&1
      return $?
    fi
    echo "[ablation-suite] warning: unzip unavailable; checking only that $checkpoint is nonempty" >&2
  fi
  return 0
}

newest_intact_checkpoint() {
  local checkpoint_dir="$1"
  local last="$checkpoint_dir/last.pt"
  local step_checkpoint=""
  local candidate
  local candidates=()

  shopt -s nullglob
  candidates=("$checkpoint_dir"/step_*.pt)
  shopt -u nullglob
  if ((${#candidates[@]})); then
    while IFS= read -r candidate; do
      if checkpoint_is_intact "$candidate"; then
        step_checkpoint="$candidate"
        break
      fi
      echo "[ablation-suite] ignoring damaged checkpoint: $candidate" >&2
    done < <(printf '%s\n' "${candidates[@]}" | LC_ALL=C sort -r)
  fi

  if checkpoint_is_intact "$last"; then
    # A clean early stop can make last.pt newer than the latest periodic file.
    # Conversely, an interrupted last.pt copy can leave a newer step file.
    if [[ -z "$step_checkpoint" || ! "$step_checkpoint" -nt "$last" ]]; then
      printf '%s\n' "$last"
      return 0
    fi
  elif [[ -e "$last" ]]; then
    echo "[ablation-suite] ignoring damaged checkpoint: $last" >&2
  fi

  if [[ -n "$step_checkpoint" ]]; then
    printf '%s\n' "$step_checkpoint"
    return 0
  fi
  return 1
}

if ((gpu_count == 1)); then
  train_command=(uv run python train.py)
else
  train_command=(uv run torchrun --standalone "--nproc_per_node=$gpu_count" train.py)
fi

trap 'echo "[ablation-suite] interrupted; rerun the same command to resume" >&2' INT TERM

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  run_id="${run_ids[$index]}"
  run_dir="$resolved_out_dir/$run_id"
  checkpoint_dir="$run_dir/checkpoints"
  completion_checkpoint="$checkpoint_dir/step_025000.pt"

  if checkpoint_is_intact "$completion_checkpoint"; then
    echo "[ablation-suite] skip complete: $run_id"
    continue
  fi

  resume_args=()
  resume_checkpoint=""
  if resume_checkpoint="$(newest_intact_checkpoint "$checkpoint_dir")"; then
    resume_args=(--resume "$resume_checkpoint")
    echo "[ablation-suite] resume: $run_id"
    echo "[ablation-suite] checkpoint: $resume_checkpoint"
  else
    echo "[ablation-suite] start: $run_id"
  fi

  if "${train_command[@]}" --config "$config" "${resume_args[@]}" \
      "run.run_id=$run_id" "run.out_dir=$out_dir" "$@"; then
    if ! checkpoint_is_intact "$completion_checkpoint"; then
      echo "[ablation-suite] $run_id exited cleanly before step 25000; stopping the suite" >&2
      echo "[ablation-suite] rerun this command to resume it before later conditions" >&2
      exit 3
    fi
    echo "[ablation-suite] complete: $run_id"
  else
    status=$?
    echo "[ablation-suite] $run_id stopped with status $status" >&2
    echo "[ablation-suite] rerun this command to resume from its newest intact checkpoint" >&2
    exit "$status"
  fi
done

echo "[ablation-suite] all six training runs are complete"
echo "[ablation-suite] starting evaluation for all six checkpoints"
scripts/run_ablation_evals.sh
