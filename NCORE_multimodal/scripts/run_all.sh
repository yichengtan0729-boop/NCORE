#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/ncore_tri.yaml}
python train.py --config "$CFG" --stage supervised
python train.py --config "$CFG" --stage policy_warmup
python train.py --config "$CFG" --stage grpo
