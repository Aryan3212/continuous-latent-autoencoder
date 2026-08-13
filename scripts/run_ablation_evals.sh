#!/usr/bin/env bash
# Evaluate the packed-base reference and five 25k ablation-suite runs.
# Results are written per condition and consolidated into one JSON/Markdown
# comparison under ABLATION_EVAL_OUT_DIR.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_ablation_evals.sh

Evaluates the packed-base reference plus all five 25k ablation conditions
serially. Every completed condition writes
step_25000/summary.json; the launcher also writes results.json and results.md
at the evaluation root.

Environment:
  ABLATION_OUT_DIR             Training output root (default: runs/ablations)
  ABLATION_EVAL_OUT_DIR        Evaluation output root (default: <out>/evaluations)
  ABLATION_EVAL_MANIFEST       Validation manifest for reconstruction evaluation;
                                required unless ABLATION_EVAL_SKIP_RECON=1.
  ABLATION_EVAL_SKIP_RECON     Set to 1 to skip reconstruction metrics; also set
                                ABLATION_EVAL_MIMI=0 and ABLATION_EVAL_LISTENING=0
  ABLATION_EVAL_SKIP_PROBES    Set to 1 to skip all configured frozen probes
  ABLATION_EVAL_VISUALIZE      Set to 1 to add latent visualizations
  ABLATION_EVAL_REPORT_EVALS   Set to 1 (default) for both SUBESCO temporal
                                emotion heads and MOS-colored t-SNE/UMAP
  ABLATION_EVAL_SUBESCO_DIR    Explicit SUBESCO root. Report tasks are recorded
                                as skipped when this is unset; no path is guessed.
  ABLATION_EVAL_MIMI           Set to 1 (default) for pinned Mimi 8q/1.1 kbps
  ABLATION_EVAL_LISTENING      Set to 1 to build blinded stimuli after
                                all reconstructions; it does not collect ratings
  ABLATION_EVAL_LISTENING_N    Shared source clips in listening bundle (default: 20)
  ABLATION_EVAL_REPORT_SEED    Temporal heads/plots/listening seed (default: 0)
  ABLATION_EVAL_TEMPORAL_UTTS  SUBESCO clip cap per condition (default: 2100)
  ABLATION_EVAL_ATTN_EPOCHS    Attentive-statistics epochs (default: 40)
  ABLATION_EVAL_TRANS_EPOCHS   Transformer-head epochs (default: 30)
  ABLATION_EVAL_VIZ_UTTS       UMAP/t-SNE clip cap (default: 300)
  UTMOSV2_CHECKPOINT            Optional explicit local UTMOS weight file; the
                                loaded state hash is always recorded
  ABLATION_EVAL_RECON_BATCH    Reconstruction batch size (default: 8)
  ABLATION_EVAL_RECON_BATCHES  Reconstruction batch limit (default: 50)
  ABLATION_EVAL_SEGMENT_SECONDS Fixed first-segment length for CLAE/Mimi
                                (default: 3.0)
  ABLATION_EVAL_RECON_TIMEOUT  Reconstruction timeout in seconds (default: 1800)
  ABLATION_EVAL_PROBE_TIMEOUT  Per-probe timeout in seconds (default: 1800)

Reconstruction commands run with the locked `reconstruction-metrics` project
extra. On Linux this requires both pystoi and the compiled pesq extension; uv
must be able to materialize that extra before evaluation begins.

The config's eval.* manifests control which emotion, gender, and ASR probes run.
Missing probe manifests are recorded as skipped in the consolidated results.
EOF
  exit 0
fi

if (($#)); then
  echo "[ablation-evals] this launcher accepts no positional arguments; use --help" >&2
  exit 2
fi

out_dir="${ABLATION_OUT_DIR:-runs/ablations}"
case "$out_dir" in
  /*) resolved_out_dir="$out_dir" ;;
  *) resolved_out_dir="$repo_root/$out_dir" ;;
esac

eval_out_dir="${ABLATION_EVAL_OUT_DIR:-$resolved_out_dir/evaluations}"
case "$eval_out_dir" in
  /*) resolved_eval_out_dir="$eval_out_dir" ;;
  *) resolved_eval_out_dir="$repo_root/$eval_out_dir" ;;
esac

skip_recon="${ABLATION_EVAL_SKIP_RECON:-0}"
skip_probes="${ABLATION_EVAL_SKIP_PROBES:-0}"
visualize="${ABLATION_EVAL_VISUALIZE:-0}"
report_evals="${ABLATION_EVAL_REPORT_EVALS:-1}"
run_mimi="${ABLATION_EVAL_MIMI:-1}"
run_listening="${ABLATION_EVAL_LISTENING:-0}"
for boolean_var in skip_recon skip_probes visualize report_evals run_mimi run_listening; do
  case "${!boolean_var}" in
    0|1) ;;
    *) echo "[ablation-evals] ${boolean_var} must be 0 or 1" >&2; exit 2 ;;
  esac
done
if [[ "$skip_recon" == "1" ]]; then
  if [[ "$run_mimi" == "1" || "$run_listening" == "1" ]]; then
    echo "[ablation-evals] Mimi/listening require reconstruction; set ABLATION_EVAL_MIMI=0 and ABLATION_EVAL_LISTENING=0" >&2
    exit 2
  fi
fi
for positive_var in ABLATION_EVAL_LISTENING_N ABLATION_EVAL_TEMPORAL_UTTS ABLATION_EVAL_ATTN_EPOCHS ABLATION_EVAL_TRANS_EPOCHS ABLATION_EVAL_VIZ_UTTS ABLATION_EVAL_RECON_BATCH ABLATION_EVAL_RECON_BATCHES ABLATION_EVAL_RECON_TIMEOUT ABLATION_EVAL_PROBE_TIMEOUT; do
  value="${!positive_var:-}"
  if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ablation-evals] $positive_var must be a positive integer" >&2
    exit 2
  fi
done
listening_n="${ABLATION_EVAL_LISTENING_N:-20}"
recon_audio_limit=0
if [[ "$run_listening" == "1" ]]; then
  recon_audio_limit="$listening_n"
fi
recon_budget="$(( ${ABLATION_EVAL_RECON_BATCH:-8} * ${ABLATION_EVAL_RECON_BATCHES:-50} ))"
if [[ "$run_listening" == "1" && "$listening_n" -gt "$recon_budget" ]]; then
  echo "[ablation-evals] listening item count exceeds the reconstruction sample budget" >&2
  exit 2
fi
if [[ ! "${ABLATION_EVAL_REPORT_SEED:-0}" =~ ^-?[0-9]+$ ]]; then
  echo "[ablation-evals] ABLATION_EVAL_REPORT_SEED must be an integer" >&2
  exit 2
fi
segment_seconds="${ABLATION_EVAL_SEGMENT_SECONDS:-3.0}"
if [[ ! "$segment_seconds" =~ ^[0-9]+([.][0-9]+)?$ || "$segment_seconds" =~ ^0+([.]0+)?$ ]]; then
  echo "[ablation-evals] ABLATION_EVAL_SEGMENT_SECONDS must be positive" >&2
  exit 2
fi

manifest="${ABLATION_EVAL_MANIFEST:-}"
if [[ "$skip_recon" == "0" ]]; then
  if [[ -z "$manifest" ]]; then
    echo "[ablation-evals] set ABLATION_EVAL_MANIFEST to a validation manifest, or set ABLATION_EVAL_SKIP_RECON=1" >&2
    exit 2
  fi
  if [[ ! -f "$manifest" ]]; then
    echo "[ablation-evals] validation manifest does not exist: $manifest" >&2
    exit 2
  fi
fi
if [[ -n "${UTMOSV2_CHECKPOINT:-}" && ! -f "$UTMOSV2_CHECKPOINT" ]]; then
  echo "[ablation-evals] UTMOSV2_CHECKPOINT does not exist: $UTMOSV2_CHECKPOINT" >&2
  exit 2
fi

step=25000
configs=(
  "configs/large_2kh_packed_25k.yaml"
  "configs/large_2kh_ablation_recon_only_50k.yaml"
  "configs/large_2kh_ablation_repr_only_50k.yaml"
  "configs/large_2kh_ablation_no_mhc_50k.yaml"
  "configs/large_2kh_ablation_25hz_50k.yaml"
  "configs/large_2kh_ablation_no_decoder_corruption_50k.yaml"
)
labels=(
  "packed-base"
  "reconstruction-only"
  "representation-only"
  "no-mhc"
  "25hz"
  "no-decoder-corruption"
)
run_ids=(
  "large-2kh-packed-25k"
  "large-2kh-ablation-recon-only-r-50k"
  "large-2kh-ablation-repr-only-j-visreg-50k"
  "large-2kh-ablation-no-mhc-50k"
  "large-2kh-ablation-25hz-full-50k"
  "large-2kh-ablation-no-decoder-corruption-50k"
)

mkdir -p "$resolved_eval_out_dir"
failed_conditions=0
summary_args=()
listening_args=()

checkpoint_is_intact() {
  local checkpoint="$1"
  [[ -f "$checkpoint" && -s "$checkpoint" ]] || return 1
  if [[ "$(LC_ALL=C od -An -tx1 -N2 "$checkpoint" | tr -d '[:space:]')" == "504b" ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -tqq "$checkpoint" >/dev/null 2>&1
      return $?
    fi
    echo "[ablation-evals] warning: unzip unavailable; checking only that $checkpoint is nonempty" >&2
  fi
  return 0
}

clear_generated_eval_dir() {
  local target="$1"
  case "$target" in
    "$resolved_eval_out_dir"/*) rm -rf -- "$target" ;;
    *)
      echo "[ablation-evals] refusing to clear path outside evaluation root: $target" >&2
      return 1
      ;;
  esac
}

for index in "${!configs[@]}"; do
  label="${labels[$index]}"
  config="${configs[$index]}"
  run_id="${run_ids[$index]}"
  checkpoint="$resolved_out_dir/$run_id/checkpoints/step_025000.pt"
  condition_out_dir="$resolved_eval_out_dir/$label"
  summary_args+=(--condition "$label" "$config" "$run_id" "$checkpoint" "$condition_out_dir")
  listening_args+=(
    --condition "$label" "$condition_out_dir/step_$step/reconstruction_audio/index.json"
  )
  if ! checkpoint_is_intact "$checkpoint"; then
    echo "[ablation-evals] missing or damaged checkpoint for $label: $checkpoint" >&2
    failed_conditions=1
    continue
  fi
  if ! clear_generated_eval_dir "$condition_out_dir/step_$step"; then
    exit 2
  fi

  command=(
    uv run --extra reconstruction-metrics python -m eval.run_all
    --config "$config"
    --ckpt "$checkpoint"
    --out_dir "$condition_out_dir"
    --step "$step"
    --recon_batch_size "${ABLATION_EVAL_RECON_BATCH:-8}"
    --recon_max_batches "${ABLATION_EVAL_RECON_BATCHES:-50}"
    --recon_timeout_seconds "${ABLATION_EVAL_RECON_TIMEOUT:-1800}"
    --probe_timeout_seconds "${ABLATION_EVAL_PROBE_TIMEOUT:-1800}"
    --recon_segment_seconds "$segment_seconds"
  )
  if [[ "$skip_recon" == "1" ]]; then
    command+=(--skip_recon)
  else
    command+=(--manifest "$manifest")
  fi
  if [[ "$skip_probes" == "1" ]]; then
    command+=(--skip_probes)
  fi
  if [[ "$visualize" == "1" ]]; then
    command+=(--visualize)
  fi
  command+=(--recon_audio_limit "$recon_audio_limit")
  if [[ "$report_evals" == "1" ]]; then
    command+=(
      --report_evals
      --report_seed "${ABLATION_EVAL_REPORT_SEED:-0}"
      --temporal_max_utts "${ABLATION_EVAL_TEMPORAL_UTTS:-2100}"
      --temporal_attn_epochs "${ABLATION_EVAL_ATTN_EPOCHS:-40}"
      --temporal_transformer_epochs "${ABLATION_EVAL_TRANS_EPOCHS:-30}"
      --repr_viz_max_utts "${ABLATION_EVAL_VIZ_UTTS:-300}"
    )
    if [[ -n "${ABLATION_EVAL_SUBESCO_DIR:-}" ]]; then
      command+=(--subesco_dir "$ABLATION_EVAL_SUBESCO_DIR")
    fi
  fi

  echo "[ablation-evals] evaluating: $label"
  if ! "${command[@]}"; then
    echo "[ablation-evals] evaluation failed: $label" >&2
    failed_conditions=1
  fi
done

mimi_out_dir="$resolved_eval_out_dir/baselines/mimi_8q_1.1kbps"
mimi_result="$mimi_out_dir/mimi_metrics.json"
mimi_summary_args=()
if [[ "$run_mimi" == "1" ]]; then
  if ! clear_generated_eval_dir "$mimi_out_dir"; then
    exit 2
  fi
  echo "[ablation-evals] evaluating pinned Mimi at exactly 8 quantizers / 1.1 kbps"
  mimi_command=(
    uv run --extra reconstruction-metrics python -m eval.eval_mimi_recon
    --manifest "$manifest"
    --out_dir "$mimi_out_dir"
    --batch_size "${ABLATION_EVAL_RECON_BATCH:-8}"
    --segment_seconds "$segment_seconds"
    --max_batches "${ABLATION_EVAL_RECON_BATCHES:-50}"
    --num_recon_wavs "$recon_audio_limit"
  )
  if ! "${mimi_command[@]}"; then
    echo "[ablation-evals] Mimi 8-quantizer evaluation failed" >&2
    failed_conditions=1
  fi
  if [[ -s "$mimi_result" ]]; then
    mimi_summary_args=(--mimi_result "$mimi_result")
  fi
  if [[ -s "$mimi_out_dir/audio_pairs/index.json" ]]; then
    listening_args+=(
      --condition "mimi-8q-1.1kbps" "$mimi_out_dir/audio_pairs/index.json"
    )
  fi
fi

listening_dir="$resolved_eval_out_dir/listening_study"
listening_summary_args=()
if [[ "$run_listening" == "1" ]]; then
  if ! clear_generated_eval_dir "$listening_dir"; then
    exit 2
  fi
  echo "[ablation-evals] building blinded fixed-source listening package"
  if ! uv run python scripts/build_listening_study.py \
      --out_dir "$listening_dir" \
      --num_items "${ABLATION_EVAL_LISTENING_N:-20}" \
      --seed "${ABLATION_EVAL_REPORT_SEED:-0}" \
      "${listening_args[@]}"; then
    echo "[ablation-evals] listening package generation failed" >&2
    failed_conditions=1
  else
    listening_summary_args=(--listening_dir "$listening_dir")
  fi
fi

if ! uv run python scripts/summarize_ablation_evals.py \
    --out_dir "$resolved_eval_out_dir" \
    --step "$step" \
    "${mimi_summary_args[@]}" \
    "${listening_summary_args[@]}" \
    "${summary_args[@]}"; then
  echo "[ablation-evals] could not write consolidated results" >&2
  exit 1
fi

echo "[ablation-evals] wrote $resolved_eval_out_dir/results.json"
echo "[ablation-evals] wrote $resolved_eval_out_dir/results.md"
exit "$failed_conditions"
