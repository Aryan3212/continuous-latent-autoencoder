#!/bin/bash
set -e

# Research Suite Launcher
# Usage: ./scripts/run_research_suite.sh <experiment_name> <base_config> [optional_overrides...]

EXP_NAME=$1
BASE_CONFIG=$2
shift 2
OVERRIDES="$@"

if [ -z "$EXP_NAME" ] || [ -z "$BASE_CONFIG" ]; then
    echo "Usage: $0 <experiment_name> <base_config> [overrides...]"
    echo "Example: $0 my_study configs/exp0.yaml model.encoder.n_layers=8"
    exit 1
fi

echo "=== Starting Research Suite: $EXP_NAME ==="
echo "Base Config: $BASE_CONFIG"
echo "Overrides: $OVERRIDES"

SEEDS=(42 43 44)

for SEED in "${SEEDS[@]}"; do
    RUN_ID="${EXP_NAME}_s${SEED}"
    echo "------------------------------------------------"
    echo "Running Seed $SEED -> $RUN_ID"
    echo "------------------------------------------------"
    
    uv run python train.py \
        --config "$BASE_CONFIG" \
        $OVERRIDES \
        run.run_id="$RUN_ID" \
        run.seed="$SEED"
        
    # Optional: Run quick eval immediately after training
    # CKPT="runs/$RUN_ID/checkpoints/best_composite.pt"
    # if [ -f "$CKPT" ]; then
    #    echo "Evaluating $RUN_ID..."
    #    uv run python eval/run_all.py \
    #        --config "$BASE_CONFIG" \
    #        --ckpt "$CKPT" \
    #        --manifest "data/manifests/val.jsonl" \
    #        --out_dir "runs/$RUN_ID/eval_suite"
    # fi
done

echo "=== Suite Completed: $EXP_NAME ==="
