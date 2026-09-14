#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v5_smoke.yaml}
SEED=${2:-42}
CHECKPOINT_DIR=strong_residual_outputs/ncore_performance_v5_smoke/mortality

python scripts/make_dummy_strong_data.py
bash scripts/run_v5.sh "$CFG" "$SEED"

if [[ -f "$CHECKPOINT_DIR/best_final_v5.pt" ]]; then
  CHECKPOINT="$CHECKPOINT_DIR/best_final_v5.pt"
else
  CHECKPOINT="$CHECKPOINT_DIR/best_unconfirmed.pt"
fi

python evaluate.py \
  --config "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --stage grpo \
  --split test \
  --oracle-diagnostics \
  --output-json "$CHECKPOINT_DIR/smoke_single_test.json"

if [[ -f "$CHECKPOINT_DIR/ensemble_weights_v5.json" ]]; then
  python evaluate_ensemble.py \
    --config "$CFG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --split test \
    --output-json "$CHECKPOINT_DIR/smoke_ensemble_test.json"
fi
