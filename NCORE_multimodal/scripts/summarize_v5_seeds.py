from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Per-seed evaluation JSON files")
    args = parser.parse_args()
    rows = [json.loads(Path(path).read_text()) for path in args.results]
    numeric_keys = sorted(
        set.intersection(
            *[
                {key for key, value in row.items() if isinstance(value, (int, float))}
                for row in rows
            ]
        )
    )
    summary = {"num_seeds": len(rows), "metrics": {}}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "mean_plus_minus_std": f"{np.nanmean(values):.6f}±{np.nanstd(values):.6f}",
        }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
