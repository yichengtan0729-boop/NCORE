#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v6_mortality.yaml}
SEED=${2:-42}
SPLIT=${3:-test}
OUT=$(python -c 'import sys; from ncore.config import load_config, ensure_output_dir; c=load_config(sys.argv[1]); c["seed"]=int(sys.argv[2]); print(ensure_output_dir(c))' "$CFG" "$SEED")

CHECKPOINT=""
for name in best_final_v6.pt best_unconfirmed.pt best_policy_warmup.pt best_supervised.pt; do
  if [[ -f "$OUT/$name" ]]; then
    CHECKPOINT="$OUT/$name"
    break
  fi
done
if [[ -z "$CHECKPOINT" ]]; then
  echo "No v6 evaluation checkpoint found in $OUT" >&2
  exit 1
fi

python evaluate.py \
  --config "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --stage grpo \
  --split "$SPLIT" \
  --oracle-diagnostics \
  --output-json "$OUT/${SPLIT}_v6_seed_${SEED}.json"
