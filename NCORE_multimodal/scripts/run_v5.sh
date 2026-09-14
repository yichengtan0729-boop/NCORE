#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v5_mortality.yaml}
SEED=${2:-42}

python train.py --config "$CFG" --stage direct --seed "$SEED"
python train.py --config "$CFG" --stage operator_warmup --seed "$SEED"
python train.py --config "$CFG" --stage supervised --seed "$SEED"
python train.py --config "$CFG" --stage policy_warmup --seed "$SEED"
python train.py --config "$CFG" --stage grpo --seed "$SEED"

