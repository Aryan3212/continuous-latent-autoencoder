#!/usr/bin/env bash
# Run the five matched non-reference 50k ablations serially. Re-running this
# script resumes the current condition from its newest intact checkpoint and
# skips completed runs.

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_ablation_suite.sh [TRAIN_CONFIG_OVERRIDE ...]

Environment:
  ABLATION_OUT_DIR         Output root (default: runs/ablations)
  ABLATION_GPUS            GPUs/processes per run (default: 1)

Examples:
  scripts/run_ablation_suite.sh
  ABLATION_GPUS=2 scripts/run_ablation_suite.sh
  scripts/run_ablation_suite.sh run.wandb.enabled=false
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
)

run_ids=(
  "large-2kh-ablation-recon-only-r-50k"
  "large-2kh-ablation-repr-only-j-visreg-50k"
  "large-2kh-ablation-no-mhc-50k"
  "large-2kh-ablation-25hz-full-50k"
  "large-2kh-ablation-no-decoder-corruption-50k"
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
  completion_checkpoint="$checkpoint_dir/step_050000.pt"

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
      echo "[ablation-suite] $run_id exited cleanly before step 50000; stopping the suite" >&2
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

echo "[ablation-suite] all six conditions are complete"
