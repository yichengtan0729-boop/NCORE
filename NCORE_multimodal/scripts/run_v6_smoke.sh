#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v6_smoke.yaml}
SEED=${2:-42}

python scripts/make_dummy_strong_data.py
bash scripts/run_v6.sh "$CFG" "$SEED"
bash scripts/evaluate_v6.sh "$CFG" "$SEED" test
