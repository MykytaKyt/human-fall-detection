#!/usr/bin/env python3
"""CLI: sequence-grouped cross-validation of the physics-threshold, LSTM,
and LTC fall classifiers on precomputed DDL features.

Usage:
    python scripts/train_eval.py --features data/features.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urfd_cascade.evaluate import run_cross_validation, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features.parquet")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/metrics.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded {len(df)} frames from {args.features}; device={device}")

    results, params = run_cross_validation(
        df, window=args.window, n_splits=args.folds,
        recall_target=args.recall_target, device=device, seed=args.seed,
    )

    print(f"\n===== Results (mean +/- std over {args.folds} folds, "
          f"Recall >= {args.recall_target}) =====")
    print(summarize(results, params))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": {k: v for k, v in results.items()}, "params": params}, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
