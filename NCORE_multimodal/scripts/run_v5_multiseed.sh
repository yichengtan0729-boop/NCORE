#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/ncore_performance_v5_mortality.yaml}
for SEED in 13 42 2026; do
  bash scripts/run_v5.sh "$CFG" "$SEED"
done

