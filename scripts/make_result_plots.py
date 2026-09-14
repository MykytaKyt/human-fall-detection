#!/usr/bin/env python3
"""Generate the results/f1_vs_params.png and results/pr_bars.png figures
used in the README, from the fixed measured results reported in
results/metrics.json (produced by train_eval.py).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics_path = ROOT / "results" / "metrics.json"
    with open(metrics_path) as f:
        data = json.load(f)

    results, params = data["results"], data["params"]
    names = ["threshold", "lstm", "ltc"]
    labels = ["Physics\nthreshold", "LSTM", "LTC\n(proposed)"]
    colors = ["#999999", "#4C72B0", "#DD8452"]

    means = {n: np.array(results[n]).mean(0) * 100 for n in names}
    stds = {n: np.array(results[n]).std(0) * 100 for n in names}

    # --- Figure 1: F1 vs. parameter count -----------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for n, c in zip(names, colors):
        ax.errorbar(
            params[n], means[n][2], yerr=stds[n][2],
            fmt="o", markersize=12, color=c, capsize=4,
        )
    for n, lab, c in zip(names, labels, colors):
        ax.annotate(lab.replace("\n", " "), (params[n], means[n][2]),
                    textcoords="offset points", xytext=(10, -4), fontsize=9, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (log scale)")
    ax.set_ylabel("F1 score on fall class (%)")
    ax.set_title("F1 vs. model size (URFD, 5-fold CV)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "f1_vs_params.png", dpi=150)
    print("Saved results/f1_vs_params.png")

    # --- Figure 2: Precision/Recall/F1 bar chart -----------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    width = 0.25
    metric_names = ["Precision", "Recall", "F1"]
    for i, (n, c) in enumerate(zip(names, colors)):
        ax.bar(x + (i - 1) * width, means[n], width, yerr=stds[n], color=c, capsize=3)
    ax.set_xticks(x, metric_names)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Kinematic-cascade results on URFD (Recall-priority operating point)")
    ax.legend(["Physics threshold", "LSTM", "LTC (proposed)"], loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "pr_bars.png", dpi=150)
    print("Saved results/pr_bars.png")


if __name__ == "__main__":
    sys.exit(main())
