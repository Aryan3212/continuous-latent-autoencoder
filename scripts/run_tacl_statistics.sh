#!/usr/bin/env bash
# Run fixed-checkpoint TACL evaluations and paired bootstrap statistics.
# This script trains probe heads only; it never launches autoencoder training.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_tacl_statistics.sh

Runs the six step-25k checkpoints on fixed data, repeats stochastic probes with
probe seeds 0, 1, and 2 by default, then computes paired bootstrap intervals
from saved per-item/trial artifacts. It does not run autoencoder training.

Required:
  TACL_RECON_MANIFEST        Fixed held-out reconstruction JSONL manifest

Optional task inputs (a task runs only when its inputs are supplied):
  TACL_ASR_TRAIN_MANIFEST    Attention-ASR training manifest
  TACL_ASR_DEV_MANIFEST      Attention-ASR development/test manifest
  TACL_SUBESCO_DIR           Local SUBESCO root for emotion classification
  TACL_CV_ROOT               Common Voice Bengali root for age classification

Speaker tasks:
  TACL_RUN_SPEAKER           1 to run OpenSLR-53 speaker ID/verification (default: 1
                             when datasets/OpenSLR53/asr_bengali/utt_spk_text.tsv exists)

Controls:
  TACL_OUT_DIR               Statistics output root (default: runs/ablations/statistics)
  ABLATION_OUT_DIR           Checkpoint root (default: runs/ablations)
  PROBE_SEEDS                Space-separated probe seeds (default: "0 1 2")
  DATA_SEED                  Fixed dataset/split seed (default: 0)
  BOOTSTRAP_SEED             Resampling seed (default: 0)
  BOOTSTRAP_REPLICATES       Number of bootstrap draws (default: 10000)
  TACL_RECON_BATCH           Reconstruction batch size (default: 8)
  TACL_RECON_BATCHES         Reconstruction batch limit (default: 50)
  TACL_SEGMENT_SECONDS       Reconstruction segment length (default: 3.0)
  TACL_ASR_STEPS             Attention-ASR probe updates (default: 8000)
  TACL_ASR_BATCH             Attention-ASR batch size (default: 16)
  TACL_MAX_UTTS              Speaker/paralinguistic utterance cap (default: 2000)
  TACL_VERIFICATION_TRIALS   Verification trial cap (default: 20000)
  TACL_EMOTION_MAX_FRAMES    Shared temporal-head frame cap (default: 300)
  TACL_TEMPORAL_EPOCHS       Attentive-statistics emotion epochs (default: 40)
  TACL_TRANSFORMER_EPOCHS    Transformer emotion epochs (default: 30)

All configured paths and all six checkpoints are validated before any output
directory is created. Frozen attention-ASR and temporal-emotion features are
cached once per checkpoint and reused across probe seeds.
EOF
  exit 0
fi
if (($#)); then
  echo "[tacl-statistics] this launcher accepts no arguments; use --help" >&2
  exit 2
fi

training_root="${ABLATION_OUT_DIR:-runs/ablations}"
stats_root="${TACL_OUT_DIR:-$training_root/statistics}"
recon_manifest="${TACL_RECON_MANIFEST:-}"
asr_train_manifest="${TACL_ASR_TRAIN_MANIFEST:-}"
asr_dev_manifest="${TACL_ASR_DEV_MANIFEST:-}"
subesco_dir="${TACL_SUBESCO_DIR:-}"
cv_root="${TACL_CV_ROOT:-}"
probe_seeds="${PROBE_SEEDS:-0 1 2}"
data_seed="${DATA_SEED:-0}"
bootstrap_seed="${BOOTSTRAP_SEED:-0}"
bootstrap_replicates="${BOOTSTRAP_REPLICATES:-10000}"
run_speaker="${TACL_RUN_SPEAKER:-auto}"
verification_trials="${TACL_VERIFICATION_TRIALS:-20000}"

case "$training_root" in /*) ;; *) training_root="$repo_root/$training_root" ;; esac
case "$stats_root" in /*) ;; *) stats_root="$repo_root/$stats_root" ;; esac

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

fail() {
  echo "[tacl-statistics] $*" >&2
  exit 2
}
require_file() {
  [[ -f "$1" ]] || fail "$2 does not exist: $1"
}
require_dir() {
  [[ -d "$1" ]] || fail "$2 does not exist: $1"
}
positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer"
}
integer() {
  [[ "$2" =~ ^-?[0-9]+$ ]] || fail "$1 must be an integer"
}
checkpoint_is_intact() {
  local checkpoint="$1"
  [[ -f "$checkpoint" && -s "$checkpoint" ]] || return 1
  if [[ "$(LC_ALL=C od -An -tx1 -N2 "$checkpoint" | tr -d '[:space:]')" == "504b" ]] \
      && command -v unzip >/dev/null 2>&1; then
    unzip -tqq "$checkpoint" >/dev/null 2>&1
  fi
}

# Preflight everything before creating or replacing outputs.
[[ -n "$recon_manifest" ]] || fail "set TACL_RECON_MANIFEST to the fixed held-out manifest"
require_file "$recon_manifest" "TACL_RECON_MANIFEST"
if [[ -n "$asr_train_manifest" || -n "$asr_dev_manifest" ]]; then
  [[ -n "$asr_train_manifest" && -n "$asr_dev_manifest" ]] || \
    fail "set both TACL_ASR_TRAIN_MANIFEST and TACL_ASR_DEV_MANIFEST"
  require_file "$asr_train_manifest" "TACL_ASR_TRAIN_MANIFEST"
  require_file "$asr_dev_manifest" "TACL_ASR_DEV_MANIFEST"
fi
[[ -z "$subesco_dir" ]] || require_dir "$subesco_dir" "TACL_SUBESCO_DIR"
[[ -z "$cv_root" ]] || require_dir "$cv_root" "TACL_CV_ROOT"
positive_integer "BOOTSTRAP_REPLICATES" "$bootstrap_replicates"
positive_integer "TACL_RECON_BATCH" "${TACL_RECON_BATCH:-8}"
positive_integer "TACL_RECON_BATCHES" "${TACL_RECON_BATCHES:-50}"
positive_integer "TACL_ASR_STEPS" "${TACL_ASR_STEPS:-8000}"
positive_integer "TACL_ASR_BATCH" "${TACL_ASR_BATCH:-16}"
positive_integer "TACL_MAX_UTTS" "${TACL_MAX_UTTS:-2000}"
positive_integer "TACL_VERIFICATION_TRIALS" "$verification_trials"
positive_integer "TACL_EMOTION_MAX_FRAMES" "${TACL_EMOTION_MAX_FRAMES:-300}"
positive_integer "TACL_TEMPORAL_EPOCHS" "${TACL_TEMPORAL_EPOCHS:-40}"
positive_integer "TACL_TRANSFORMER_EPOCHS" "${TACL_TRANSFORMER_EPOCHS:-30}"
integer "DATA_SEED" "$data_seed"
integer "BOOTSTRAP_SEED" "$bootstrap_seed"
segment_seconds="${TACL_SEGMENT_SECONDS:-3.0}"
if [[ ! "$segment_seconds" =~ ^[0-9]+([.][0-9]+)?$ || "$segment_seconds" =~ ^0+([.]0+)?$ ]]; then
  fail "TACL_SEGMENT_SECONDS must be positive"
fi
read -r -a seed_array <<< "$probe_seeds"
((${#seed_array[@]})) || fail "PROBE_SEEDS must contain at least one integer"
for seed in "${seed_array[@]}"; do integer "each PROBE_SEEDS value" "$seed"; done
if [[ "$run_speaker" == "auto" ]]; then
  if [[ -f "datasets/OpenSLR53/asr_bengali/utt_spk_text.tsv" ]]; then run_speaker=1; else run_speaker=0; fi
fi
[[ "$run_speaker" == "0" || "$run_speaker" == "1" ]] || \
  fail "TACL_RUN_SPEAKER must be 0 or 1"
if [[ "$run_speaker" == "1" ]]; then
  require_file "datasets/OpenSLR53/asr_bengali/utt_spk_text.tsv" "OpenSLR-53 speaker metadata"
  require_dir "datasets/OpenSLR53/asr_bengali/data" "OpenSLR-53 audio directory"
fi

checkpoints=()
for index in "${!configs[@]}"; do
  require_file "${configs[$index]}" "condition config"
  checkpoint="$training_root/${run_ids[$index]}/checkpoints/step_025000.pt"
  checkpoint_is_intact "$checkpoint" || fail "checkpoint is missing or damaged: $checkpoint"
  checkpoints+=("$checkpoint")
done

mkdir -p "$stats_root"
current_bootstrap="$stats_root/bootstrap_statistics.json"
if [[ -f "$current_bootstrap" ]]; then
  archive_dir="$stats_root/archive"
  mkdir -p "$archive_dir"
  archive_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "$current_bootstrap" \
    "$archive_dir/bootstrap_statistics.${archive_stamp}.$$.json" || \
    fail "could not archive the previous bootstrap aggregate"
fi
failed=0
bootstrap_entries=()
for index in "${!configs[@]}"; do
  label="${labels[$index]}"
  config="${configs[$index]}"
  checkpoint="${checkpoints[$index]}"
  condition_dir="$stats_root/$label"

  recon_out="$condition_dir/reconstruction.json"
  echo "[tacl-statistics] reconstruction: $label"
  if uv run --extra reconstruction-metrics python -m eval.eval_recon \
      --config "$config" --ckpt "$checkpoint" --manifest "$recon_manifest" \
      --batch_size "${TACL_RECON_BATCH:-8}" \
      --max_batches "${TACL_RECON_BATCHES:-50}" \
      --segment_seconds "$segment_seconds" --out "$recon_out"; then
    bootstrap_entries+=(--entry reconstruction "$label" none "$recon_out")
  else
    echo "[tacl-statistics] reconstruction failed: $label" >&2
    failed=1
  fi

  if [[ -n "$asr_train_manifest" ]]; then
    asr_cache="$condition_dir/asr_features"
    for seed in "${seed_array[@]}"; do
      asr_out="$condition_dir/asr_seed_${seed}.json"
      echo "[tacl-statistics] attention-ASR: $label seed $seed"
      if uv run python -m eval.eval_asr_attn \
          --model ours --config "$config" --ckpt "$checkpoint" \
          --train_manifest "$asr_train_manifest" --dev_manifest "$asr_dev_manifest" \
          --steps "${TACL_ASR_STEPS:-8000}" --batch_size "${TACL_ASR_BATCH:-16}" \
          --seed "$seed" \
          --feature_cache_dir "$asr_cache" --out "$asr_out"; then
        bootstrap_entries+=(--entry asr "$label" "$seed" "$asr_out")
      else
        echo "[tacl-statistics] attention-ASR failed: $label seed $seed" >&2
        failed=1
      fi
    done
  fi

  if [[ -n "$subesco_dir" ]]; then
    for seed in "${seed_array[@]}"; do
      emotion_out="$condition_dir/emotion_seed_${seed}.json"
      echo "[tacl-statistics] emotion: $label seed $seed"
      if uv run python -m eval.eval_emotion \
          --models ours --ckpt "$checkpoint" --subesco-dir "$subesco_dir" \
          --max-utts "${TACL_MAX_UTTS:-2000}" --data-seed "$data_seed" \
          --split-seed "$data_seed" --seed "$seed" --out "$emotion_out"; then
        bootstrap_entries+=(--entry classification:emotion "$label" "$seed" "$emotion_out")
      else
        echo "[tacl-statistics] emotion failed: $label seed $seed" >&2
        failed=1
      fi

      temporal_out="$condition_dir/emotion_temporal_seed_${seed}.json"
      echo "[tacl-statistics] attentive-statistics emotion: $label seed $seed"
      if uv run python -m eval.eval_emotion_temporal \
          --models ours --ckpt "$checkpoint" --subesco-dir "$subesco_dir" \
          --max-utts "${TACL_MAX_UTTS:-2000}" \
          --max-frames "${TACL_EMOTION_MAX_FRAMES:-300}" \
          --epochs "${TACL_TEMPORAL_EPOCHS:-40}" \
          --data-seed "$data_seed" --split-seed "$data_seed" --seed "$seed" \
          --feature-cache-dir "$condition_dir/emotion_frame_cache" \
          --out "$temporal_out"; then
        bootstrap_entries+=(--entry classification:emotion_temporal "$label" "$seed" "$temporal_out")
      else
        echo "[tacl-statistics] attentive-statistics emotion failed: $label seed $seed" >&2
        failed=1
      fi

      transformer_out="$condition_dir/emotion_transformer_seed_${seed}.json"
      echo "[tacl-statistics] Transformer emotion: $label seed $seed"
      if uv run python -m eval.eval_emotion_transformer \
          --models ours --ckpt "$checkpoint" --subesco-dir "$subesco_dir" \
          --max-utts "${TACL_MAX_UTTS:-2000}" \
          --max-frames "${TACL_EMOTION_MAX_FRAMES:-300}" \
          --epochs "${TACL_TRANSFORMER_EPOCHS:-30}" \
          --data-seed "$data_seed" --split-seed "$data_seed" --seed "$seed" \
          --feature-cache-dir "$condition_dir/emotion_frame_cache" \
          --out "$transformer_out"; then
        bootstrap_entries+=(--entry classification:emotion_transformer "$label" "$seed" "$transformer_out")
      else
        echo "[tacl-statistics] Transformer emotion failed: $label seed $seed" >&2
        failed=1
      fi
    done
  fi

  if [[ -n "$cv_root" ]]; then
    for seed in "${seed_array[@]}"; do
      age_out="$condition_dir/age_seed_${seed}.json"
      echo "[tacl-statistics] age: $label seed $seed"
      if uv run python -m eval.eval_age \
          --models ours --ckpt "$checkpoint" --cv_root "$cv_root" \
          --max-utts "${TACL_MAX_UTTS:-2000}" --data-seed "$data_seed" \
          --split-seed "$data_seed" --seed "$seed" --out "$age_out"; then
        bootstrap_entries+=(--entry classification:age "$label" "$seed" "$age_out")
      else
        echo "[tacl-statistics] age failed: $label seed $seed" >&2
        failed=1
      fi
    done
  fi

  if [[ "$run_speaker" == "1" ]]; then
    for seed in "${seed_array[@]}"; do
      speaker_id_out="$condition_dir/speaker_id_seed_${seed}.json"
      echo "[tacl-statistics] speaker ID: $label seed $seed"
      if uv run python -m eval.eval_speaker_id \
          --models ours --ckpt "$checkpoint" --max-utts "${TACL_MAX_UTTS:-2000}" \
          --data-seed "$data_seed" --split-seed "$data_seed" \
          --seed "$seed" --out "$speaker_id_out"; then
        bootstrap_entries+=(--entry classification:speaker_id "$label" "$seed" "$speaker_id_out")
      else
        echo "[tacl-statistics] speaker ID failed: $label seed $seed" >&2
        failed=1
      fi
    done
    verification_out="$condition_dir/speaker_verification.json"
    echo "[tacl-statistics] speaker verification: $label"
    if uv run python -m eval.eval_speaker_verif \
        --models ours --pools meanstd --ckpt "$checkpoint" \
        --max-utts "${TACL_MAX_UTTS:-2000}" \
        --max-trials "$verification_trials" \
        --data-seed "$data_seed" --split-seed "$data_seed" \
        --out "$verification_out"; then
      bootstrap_entries+=(--entry verification "$label" none "$verification_out")
    else
      echo "[tacl-statistics] speaker verification failed: $label" >&2
      failed=1
    fi
  fi
done

if [[ "$run_speaker" == "0" ]]; then
  echo "[tacl-statistics] speaker tasks skipped: set TACL_RUN_SPEAKER=1 after materializing OpenSLR-53"
fi
[[ -n "$asr_train_manifest" ]] || echo "[tacl-statistics] ASR skipped: both ASR manifests are unset"
[[ -n "$subesco_dir" ]] || echo "[tacl-statistics] emotion skipped: TACL_SUBESCO_DIR is unset"
[[ -n "$cv_root" ]] || echo "[tacl-statistics] age skipped: TACL_CV_ROOT is unset"

if [[ "$failed" != "0" ]]; then
  echo "[tacl-statistics] one or more evaluations failed; preserving artifacts and skipping bootstrap" >&2
  exit 1
fi
if ((${#bootstrap_entries[@]})); then
  bootstrap_tmp="$stats_root/.bootstrap_statistics.json.tmp.$$"
  if ! uv run python -m eval.bootstrap_statistics \
      --replicates "$bootstrap_replicates" --bootstrap-seed "$bootstrap_seed" \
      --out "$bootstrap_tmp" "${bootstrap_entries[@]}"; then
    echo "[tacl-statistics] bootstrap analysis failed" >&2
    rm -f -- "$bootstrap_tmp"
    failed=1
  else
    mv -- "$bootstrap_tmp" "$stats_root/bootstrap_statistics.json"
  fi
fi
exit "$failed"
