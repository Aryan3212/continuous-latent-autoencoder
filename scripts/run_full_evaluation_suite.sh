#!/usr/bin/env bash
# Resume-aware evaluation matrix for the six 25k ablations and packed 210k run.
# This trains downstream probe heads only. It never launches autoencoder training.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_full_evaluation_suite.sh

Runs the final step-25k checkpoint for each packed ablation and last.pt from
the packed 210k run. It evaluates FP32 reconstruction, pinned Mimi 8q/1.1 kbps
reconstruction, generic and temporal emotion, age, Common Voice gender,
closed-set speaker ID, speaker verification, and attention-ASR CER/WER.

Required environment:
  FULL_EVAL_RECON_MANIFEST       Fixed held-out reconstruction JSONL
  FULL_EVAL_ASR_TRAIN_MANIFEST   Fixed text-labelled ASR training JSONL
  FULL_EVAL_ASR_DEV_MANIFEST     Fixed text-labelled ASR development JSONL
  FULL_EVAL_SUBESCO_DIR          Local SUBESCO root
  FULL_EVAL_CV_ROOT              Common Voice Bengali release/root

Optional controls:
  FULL_EVAL_OUT_DIR              Output root (default: runs/full_evaluations)
  FULL_EVAL_FORCE                1 reruns successful tasks (default: 0)
  FULL_EVAL_MAIN_STEP            Label for packed last.pt (default: 210000)
  FULL_EVAL_PROBE_SEEDS          Probe seeds (default: "0 1 2")
  FULL_EVAL_RECON_BATCH          Reconstruction batch size (default: 8)
  FULL_EVAL_RECON_BATCHES        Reconstruction batch limit (default: 50)
  FULL_EVAL_SEGMENT_SECONDS      Reconstruction segment length (default: 3.0)
  FULL_EVAL_EMOTION_MAX_UTTS     0=all SUBESCO clips (default: 0)
  FULL_EVAL_CV_MAX_UTTS          0=all labelled Common Voice clips (default: 0)
  FULL_EVAL_SPEAKER_ID_MAX_UTTS  Speaker-ID utterance cap (default: 1000)
  FULL_EVAL_SPEAKER_VERIF_MAX_UTTS  Verification utterance cap (default: 2000)
  FULL_EVAL_VERIFICATION_TRIALS  0=all pairs (default: 0)
  FULL_EVAL_ASR_MAX_SAMPLES      Per-split ASR cap (default: 10000)
  FULL_EVAL_ASR_STEPS            ASR probe updates (default: 8000)
  FULL_EVAL_ASR_BATCH            ASR probe batch size (default: 16)
  FULL_EVAL_EXTERNAL_MODELS      Reusable non-CLAE report baselines
  FULL_EVAL_ASR_BASELINES        Comma-separated ASR baseline adapters (default: wavlm)
  FULL_EVAL_TEMPORAL_BASELINES   Temporal emotion baselines (default: ours_random,wavlm)

Completed tasks have a neighbouring .ok marker. Reinvocation retries only
outputs without that marker, including tasks that previously wrote diagnostics
then returned nonzero. Set FULL_EVAL_FORCE=1 to discard the markers and rerun.

The generic Mimi representation baseline intentionally uses continuous
pre-quantization features. The separate reconstruction command enforces eight
quantizers and refuses an uncontrolled bitrate.
EOF
  exit 0
fi
if (($#)); then
  echo "[full-eval] this launcher accepts no positional arguments; use --help" >&2
  exit 2
fi

fail() {
  echo "[full-eval] $*" >&2
  exit 2
}
require_file() { [[ -f "$1" ]] || fail "$2 does not exist: $1"; }
require_dir() { [[ -d "$1" ]] || fail "$2 does not exist: $1"; }
positive_integer() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer"; }
nonnegative_integer() { [[ "$2" =~ ^[0-9]+$ ]] || fail "$1 must be a non-negative integer"; }
integer() { [[ "$2" =~ ^-?[0-9]+$ ]] || fail "$1 must be an integer"; }

checkpoint_is_intact() {
  local checkpoint="$1"
  [[ -f "$checkpoint" && -s "$checkpoint" ]] || return 1
  if [[ "$(LC_ALL=C od -An -tx1 -N2 "$checkpoint" | tr -d '[:space:]')" == "504b" ]] \
      && command -v unzip >/dev/null 2>&1; then
    unzip -tqq "$checkpoint" >/dev/null 2>&1
  fi
}

run_task() {
  local out="$1"
  local description="$2"
  shift 2
  local marker="${out}.ok"
  if [[ "$force" == "0" && -s "$out" && -f "$marker" ]]; then
    echo "[full-eval] reuse: $description"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  rm -f -- "$marker"
  echo "[full-eval] running: $description"
  if "$@"; then
    touch "$marker"
    return 0
  fi
  echo "[full-eval] failed: $description (diagnostics retained at $out)" >&2
  failed=1
  return 0
}

asr_task() {
  local out="$1"
  local description="$2"
  local model="$3"
  local config="$4"
  local checkpoint="$5"
  local cache_dir="$6"
  local seed="$7"
  local -a command=(
    uv run python -m eval.eval_asr_attn
    --model "$model"
    --train_manifest "$asr_train_manifest"
    --dev_manifest "$asr_dev_manifest"
    --steps "$asr_steps"
    --batch_size "$asr_batch"
    --seed "$seed"
    --segment_seconds 15
    --max_utt_seconds 15
    --chunk_seconds 3
    --max_samples "$asr_max_samples"
    --feature_cache_dir "$cache_dir"
    --out "$out"
  )
  if [[ "$model" == "ours" ]]; then
    command+=(--config "$config" --ckpt "$checkpoint")
  elif [[ "$model" == "ours_random" ]]; then
    command+=(--ckpt "$checkpoint")
  fi
  run_task "$out" "$description" "${command[@]}"
}

recon_manifest="${FULL_EVAL_RECON_MANIFEST:-}"
asr_train_manifest="${FULL_EVAL_ASR_TRAIN_MANIFEST:-}"
asr_dev_manifest="${FULL_EVAL_ASR_DEV_MANIFEST:-}"
subesco_dir="${FULL_EVAL_SUBESCO_DIR:-}"
cv_root="${FULL_EVAL_CV_ROOT:-}"
force="${FULL_EVAL_FORCE:-0}"
case "$force" in 0|1) ;; *) fail "FULL_EVAL_FORCE must be 0 or 1" ;; esac

[[ -n "$recon_manifest" ]] || fail "set FULL_EVAL_RECON_MANIFEST"
[[ -n "$asr_train_manifest" ]] || fail "set FULL_EVAL_ASR_TRAIN_MANIFEST"
[[ -n "$asr_dev_manifest" ]] || fail "set FULL_EVAL_ASR_DEV_MANIFEST"
[[ -n "$subesco_dir" ]] || fail "set FULL_EVAL_SUBESCO_DIR"
[[ -n "$cv_root" ]] || fail "set FULL_EVAL_CV_ROOT"
require_file "$recon_manifest" "FULL_EVAL_RECON_MANIFEST"
require_file "$asr_train_manifest" "FULL_EVAL_ASR_TRAIN_MANIFEST"
require_file "$asr_dev_manifest" "FULL_EVAL_ASR_DEV_MANIFEST"
require_dir "$subesco_dir" "FULL_EVAL_SUBESCO_DIR"
require_dir "$cv_root" "FULL_EVAL_CV_ROOT"
require_file "datasets/OpenSLR53/asr_bengali/utt_spk_text.tsv" "OpenSLR-53 speaker metadata"
require_dir "datasets/OpenSLR53/asr_bengali/data" "OpenSLR-53 audio directory"

out_root="${FULL_EVAL_OUT_DIR:-runs/full_evaluations}"
case "$out_root" in /*) ;; *) out_root="$repo_root/$out_root" ;; esac
probe_seeds="${FULL_EVAL_PROBE_SEEDS:-0 1 2}"
read -r -a seed_array <<< "$probe_seeds"
((${#seed_array[@]})) || fail "FULL_EVAL_PROBE_SEEDS must contain at least one integer"
for seed in "${seed_array[@]}"; do integer "each FULL_EVAL_PROBE_SEEDS value" "$seed"; done

recon_batch="${FULL_EVAL_RECON_BATCH:-8}"
recon_batches="${FULL_EVAL_RECON_BATCHES:-50}"
segment_seconds="${FULL_EVAL_SEGMENT_SECONDS:-3.0}"
emotion_max_utts="${FULL_EVAL_EMOTION_MAX_UTTS:-0}"
cv_max_utts="${FULL_EVAL_CV_MAX_UTTS:-0}"
speaker_id_max_utts="${FULL_EVAL_SPEAKER_ID_MAX_UTTS:-1000}"
speaker_verif_max_utts="${FULL_EVAL_SPEAKER_VERIF_MAX_UTTS:-2000}"
verification_trials="${FULL_EVAL_VERIFICATION_TRIALS:-0}"
asr_max_samples="${FULL_EVAL_ASR_MAX_SAMPLES:-10000}"
asr_steps="${FULL_EVAL_ASR_STEPS:-8000}"
asr_batch="${FULL_EVAL_ASR_BATCH:-16}"
data_seed="${FULL_EVAL_DATA_SEED:-0}"
split_seed="${FULL_EVAL_SPLIT_SEED:-0}"
temporal_max_frames="${FULL_EVAL_TEMPORAL_MAX_FRAMES:-300}"
temporal_attn_epochs="${FULL_EVAL_TEMPORAL_ATTN_EPOCHS:-40}"
temporal_transformer_epochs="${FULL_EVAL_TEMPORAL_TRANSFORMER_EPOCHS:-30}"
external_models="${FULL_EVAL_EXTERNAL_MODELS:-wavlm,whisper_tiny,ecapa,emotion2vec,mimi,higgs_audio_v2}"
asr_baselines="${FULL_EVAL_ASR_BASELINES:-wavlm}"
temporal_baselines="${FULL_EVAL_TEMPORAL_BASELINES:-ours_random,wavlm}"

positive_integer "FULL_EVAL_RECON_BATCH" "$recon_batch"
positive_integer "FULL_EVAL_RECON_BATCHES" "$recon_batches"
nonnegative_integer "FULL_EVAL_EMOTION_MAX_UTTS" "$emotion_max_utts"
nonnegative_integer "FULL_EVAL_CV_MAX_UTTS" "$cv_max_utts"
positive_integer "FULL_EVAL_SPEAKER_ID_MAX_UTTS" "$speaker_id_max_utts"
positive_integer "FULL_EVAL_SPEAKER_VERIF_MAX_UTTS" "$speaker_verif_max_utts"
nonnegative_integer "FULL_EVAL_VERIFICATION_TRIALS" "$verification_trials"
positive_integer "FULL_EVAL_ASR_MAX_SAMPLES" "$asr_max_samples"
positive_integer "FULL_EVAL_ASR_STEPS" "$asr_steps"
positive_integer "FULL_EVAL_ASR_BATCH" "$asr_batch"
positive_integer "FULL_EVAL_TEMPORAL_MAX_FRAMES" "$temporal_max_frames"
positive_integer "FULL_EVAL_TEMPORAL_ATTN_EPOCHS" "$temporal_attn_epochs"
positive_integer "FULL_EVAL_TEMPORAL_TRANSFORMER_EPOCHS" "$temporal_transformer_epochs"
integer "FULL_EVAL_DATA_SEED" "$data_seed"
integer "FULL_EVAL_SPLIT_SEED" "$split_seed"
if [[ ! "$segment_seconds" =~ ^[0-9]+([.][0-9]+)?$ || "$segment_seconds" =~ ^0+([.]0+)?$ ]]; then
  fail "FULL_EVAL_SEGMENT_SECONDS must be positive"
fi

labels=(
  "packed-base-25k"
  "reconstruction-only-25k"
  "representation-only-25k"
  "no-mhc-25k"
  "25hz-25k"
  "no-decoder-corruption-25k"
  "packed-210k-last"
)
configs=(
  "configs/large_2kh_packed_25k.yaml"
  "configs/large_2kh_ablation_recon_only_50k.yaml"
  "configs/large_2kh_ablation_repr_only_50k.yaml"
  "configs/large_2kh_ablation_no_mhc_50k.yaml"
  "configs/large_2kh_ablation_25hz_50k.yaml"
  "configs/large_2kh_ablation_no_decoder_corruption_50k.yaml"
  "runs/large-2kh-packed-210k-tail-lr-1e4/config.yaml"
)
checkpoints=(
  "runs/ablations/large-2kh-packed-25k/checkpoints/step_025000.pt"
  "runs/ablations/large-2kh-ablation-recon-only-r-50k-v2/checkpoints/step_025000.pt"
  "runs/ablations/large-2kh-ablation-repr-only-j-visreg-50k/checkpoints/step_025000.pt"
  "runs/ablations/large-2kh-ablation-no-mhc-50k/checkpoints/step_025000.pt"
  "runs/ablations/large-2kh-ablation-25hz-full-50k/checkpoints/step_025000.pt"
  "runs/ablations/large-2kh-ablation-no-decoder-corruption-50k/checkpoints/step_025000.pt"
  "runs/large-2kh-packed-210k-tail-lr-1e4/checkpoints/last.pt"
)
steps=(25000 25000 25000 25000 25000 25000 "${FULL_EVAL_MAIN_STEP:-210000}")

for index in "${!labels[@]}"; do
  require_file "${configs[$index]}" "condition config"
  checkpoint_is_intact "${checkpoints[$index]}" || fail "checkpoint is missing or damaged: ${checkpoints[$index]}"
done
integer "FULL_EVAL_MAIN_STEP" "${steps[6]}"

emotion_limit_args=()
((emotion_max_utts)) && emotion_limit_args=(--max-utts "$emotion_max_utts")
cv_limit_args=()
((cv_max_utts)) && cv_limit_args=(--max-utts "$cv_max_utts")
failed=0
mkdir -p "$out_root"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  config="${configs[$index]}"
  checkpoint="${checkpoints[$index]}"
  step="${steps[$index]}"
  condition_dir="$out_root/conditions/$label/step_$step"

  recon_out="$condition_dir/reconstruction.json"
  run_task "$recon_out" "FP32 reconstruction: $label" \
    uv run --extra reconstruction-metrics python -m eval.eval_recon \
    --config "$config" --ckpt "$checkpoint" --manifest "$recon_manifest" \
    --batch_size "$recon_batch" --max_batches "$recon_batches" \
    --segment_seconds "$segment_seconds" --out "$recon_out"

  for seed in "${seed_array[@]}"; do
    run_task "$condition_dir/emotion_seed_$seed.json" "emotion: $label seed $seed" \
      uv run python -m eval.eval_emotion --models ours --ckpt "$checkpoint" \
      --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
      --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
      --out "$condition_dir/emotion_seed_$seed.json"
    run_task "$condition_dir/age_seed_$seed.json" "age: $label seed $seed" \
      uv run python -m eval.eval_age --models ours --ckpt "$checkpoint" \
      --cv_root "$cv_root" --label-column age "${cv_limit_args[@]}" \
      --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
      --out "$condition_dir/age_seed_$seed.json"
    run_task "$condition_dir/gender_seed_$seed.json" "gender: $label seed $seed" \
      uv run python -m eval.eval_age --models ours --ckpt "$checkpoint" \
      --cv_root "$cv_root" --label-column gender "${cv_limit_args[@]}" \
      --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
      --out "$condition_dir/gender_seed_$seed.json"
    run_task "$condition_dir/speaker_id_seed_$seed.json" "speaker ID: $label seed $seed" \
      uv run python -m eval.eval_speaker_id --models ours --ckpt "$checkpoint" \
      --max-utts "$speaker_id_max_utts" --data-seed "$data_seed" \
      --split-seed "$split_seed" --seed "$seed" \
      --out "$condition_dir/speaker_id_seed_$seed.json"
    asr_task "$condition_dir/asr_seed_$seed.json" "attention ASR: $label seed $seed" \
      ours "$config" "$checkpoint" "$condition_dir/asr_features" "$seed"

    run_task "$condition_dir/emotion_temporal_seed_$seed.json" "temporal emotion: $label seed $seed" \
      uv run python -m eval.eval_emotion_temporal --models ours --ckpt "$checkpoint" \
      --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
      --max-frames "$temporal_max_frames" --epochs "$temporal_attn_epochs" \
      --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
      --feature-cache-dir "$condition_dir/emotion_frame_cache" \
      --out "$condition_dir/emotion_temporal_seed_$seed.json"
    run_task "$condition_dir/emotion_transformer_seed_$seed.json" "Transformer emotion: $label seed $seed" \
      uv run python -m eval.eval_emotion_transformer --models ours --ckpt "$checkpoint" \
      --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
      --max-frames "$temporal_max_frames" --epochs "$temporal_transformer_epochs" \
      --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
      --feature-cache-dir "$condition_dir/emotion_frame_cache" \
      --out "$condition_dir/emotion_transformer_seed_$seed.json"
  done

  run_task "$condition_dir/speaker_verification.json" "speaker verification: $label" \
    uv run python -m eval.eval_speaker_verif --models ours --pools mean,meanstd \
    --ckpt "$checkpoint" --max-utts "$speaker_verif_max_utts" \
    --max-trials "$verification_trials" --data-seed "$data_seed" \
    --split-seed "$split_seed" --out "$condition_dir/speaker_verification.json"
done

# Reusable baselines are evaluated once against the final packed model's random
# control. They are not redundantly recomputed for each ablation checkpoint.
main_index=6
main_checkpoint="${checkpoints[$main_index]}"
main_config="${configs[$main_index]}"
main_step="${steps[$main_index]}"
baseline_dir="$out_root/baselines/packed-210k-last/step_$main_step"
baseline_models="ours_random,$external_models"
for seed in "${seed_array[@]}"; do
  run_task "$baseline_dir/emotion_seed_$seed.json" "report emotion baselines seed $seed" \
    uv run python -m eval.eval_emotion --models "$baseline_models" --ckpt "$main_checkpoint" \
    --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
    --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
    --out "$baseline_dir/emotion_seed_$seed.json"
  run_task "$baseline_dir/age_seed_$seed.json" "report age baselines seed $seed" \
    uv run python -m eval.eval_age --models "$baseline_models" --ckpt "$main_checkpoint" \
    --cv_root "$cv_root" --label-column age "${cv_limit_args[@]}" \
    --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
    --out "$baseline_dir/age_seed_$seed.json"
  run_task "$baseline_dir/gender_seed_$seed.json" "report gender baselines seed $seed" \
    uv run python -m eval.eval_age --models "$baseline_models" --ckpt "$main_checkpoint" \
    --cv_root "$cv_root" --label-column gender "${cv_limit_args[@]}" \
    --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
    --out "$baseline_dir/gender_seed_$seed.json"
  run_task "$baseline_dir/speaker_id_seed_$seed.json" "report speaker-ID baselines seed $seed" \
    uv run python -m eval.eval_speaker_id --models "$baseline_models" --ckpt "$main_checkpoint" \
    --max-utts "$speaker_id_max_utts" --data-seed "$data_seed" \
    --split-seed "$split_seed" --seed "$seed" \
    --out "$baseline_dir/speaker_id_seed_$seed.json"
  run_task "$baseline_dir/emotion_temporal_seed_$seed.json" "temporal emotion baselines seed $seed" \
    uv run python -m eval.eval_emotion_temporal --models "$temporal_baselines" --ckpt "$main_checkpoint" \
    --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
    --max-frames "$temporal_max_frames" --epochs "$temporal_attn_epochs" \
    --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
    --feature-cache-dir "$baseline_dir/emotion_frame_cache" \
    --out "$baseline_dir/emotion_temporal_seed_$seed.json"
  run_task "$baseline_dir/emotion_transformer_seed_$seed.json" "Transformer emotion baselines seed $seed" \
    uv run python -m eval.eval_emotion_transformer --models "$temporal_baselines" --ckpt "$main_checkpoint" \
    --subesco-dir "$subesco_dir" "${emotion_limit_args[@]}" \
    --max-frames "$temporal_max_frames" --epochs "$temporal_transformer_epochs" \
    --data-seed "$data_seed" --split-seed "$split_seed" --seed "$seed" \
    --feature-cache-dir "$baseline_dir/emotion_frame_cache" \
    --out "$baseline_dir/emotion_transformer_seed_$seed.json"
done

run_task "$baseline_dir/speaker_verification.json" "report speaker-verification baselines" \
  uv run python -m eval.eval_speaker_verif --models "$baseline_models" --pools mean,meanstd \
  --ckpt "$main_checkpoint" --max-utts "$speaker_verif_max_utts" \
  --max-trials "$verification_trials" --data-seed "$data_seed" \
  --split-seed "$split_seed" --out "$baseline_dir/speaker_verification.json"

IFS=',' read -r -a asr_baseline_array <<< "$asr_baselines"
for model in "${asr_baseline_array[@]}"; do
  [[ -n "$model" ]] || continue
  for seed in "${seed_array[@]}"; do
    asr_task "$baseline_dir/asr_${model}_seed_$seed.json" "attention ASR baseline: $model seed $seed" \
      "$model" "$main_config" "$main_checkpoint" "$baseline_dir/asr_${model}_features" "$seed"
  done
done

mimi_out_dir="$out_root/baselines/mimi_8q_1.1kbps"
mimi_out="$mimi_out_dir/mimi_metrics.json"
run_task "$mimi_out" "pinned Mimi reconstruction at exactly 8 quantizers / 1.1 kbps" \
  uv run --extra reconstruction-metrics python -m eval.eval_mimi_recon \
  --manifest "$recon_manifest" --out_dir "$mimi_out_dir" --batch_size "$recon_batch" \
  --segment_seconds "$segment_seconds" --max_batches "$recon_batches" --num_recon_wavs 0

if [[ "$failed" == "0" ]]; then
  echo "[full-eval] completed: $out_root"
else
  echo "[full-eval] one or more tasks failed; rerun this command to retry only those tasks" >&2
fi
exit "$failed"
