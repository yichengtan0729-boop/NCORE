#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v6_mortality.yaml}
SEED=${2:-42}
OUT=$(python -c 'import sys; from ncore.config import load_config, ensure_output_dir; c=load_config(sys.argv[1]); c["seed"]=int(sys.argv[2]); print(ensure_output_dir(c))' "$CFG" "$SEED")

run_stage() {
  local stage=$1
  local last_name=$2
  if [[ -f "$OUT/$last_name" ]]; then
    python train.py --config "$CFG" --stage "$stage" --seed "$SEED" --resume
  else
    python train.py --config "$CFG" --stage "$stage" --seed "$SEED"
  fi
}

run_stage direct last_direct.pt
run_stage operator_warmup last_operator_warmup.pt
run_stage supervised last_supervised.pt
run_stage policy_warmup last_policy_warmup.pt
run_stage grpo last.pt
