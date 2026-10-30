#!/usr/bin/env python3
"""Plot held-out Credit Card Fraud results from run_creditcard.py.

The plotted confidence intervals are paired transaction bootstraps within each
split. Reused transactions across splits make them unsuitable as independent
replicates for a confidence interval on the aggregate mean.
"""
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(result, out):
    rows = result["splits"]
    if not rows:
        raise ValueError("result has no splits")
    x = np.arange(len(rows))
    series = [
        ("ROC baseline", "roc_test_cost", "#8c564b"),
        ("Exact two thresholds", "exact_two_threshold_test_cost", "#ff7f0e"),
        ("Binned two thresholds", "test_cost_two", "#1f77b4"),
        ("Per-bin policy", "test_cost_bin", "#2ca02c"),
    ]
    fig, (ax_cost, ax_regret) = plt.subplots(2, 1, figsize=(10, 7),
                                              layout="constrained")
    for label, key, color in series:
        values = np.array([row[key] for row in rows], float)
        ax_cost.plot(x, values, marker="o", linewidth=1.5, markersize=4,
                     label=f"{label} (mean {values.mean():.4f})", color=color)
    ax_cost.set(ylabel="Mean cost per test transaction",
                title="Credit Card Fraud: held-out policy costs")
    ax_cost.legend(fontsize=8, ncol=2)
    ax_cost.grid(alpha=0.25)

    regret = 100 * np.array([row["regret_rel"] for row in rows], float)
    ci = 100 * np.array([row["regret_rel_ci"] for row in rows], float)
    err = np.maximum(np.vstack((regret - ci[:, 0], ci[:, 1] - regret)), 0)
    ax_regret.errorbar(x, regret, yerr=err, fmt="o", capsize=3,
                       color="#1f77b4", label="Binned two thresholds vs per-bin (95% CI)")
    roc = np.array([row["roc_test_cost"] for row in rows], float)
    exact = np.array([row["exact_two_threshold_test_cost"] for row in rows], float)
    ax_regret.plot(x, 100 * (roc / exact - 1), "s--", markersize=4,
                   color="#8c564b", label="ROC vs exact two thresholds (point estimate)")
    ax_regret.axhline(0, color="black", linewidth=0.8)
    ax_regret.set(xlabel="Split seed", ylabel="Relative cost difference (%)",
                  title="Paired bootstrap within each test split")
    ax_regret.set_xticks(x)
    ax_regret.grid(alpha=0.25)
    ax_regret.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON created by run_creditcard.py")
    ap.add_argument("--out", help="Output PNG (default: alongside JSON)")
    args = ap.parse_args()
    with open(args.results) as f:
        result = json.load(f)
    out = args.out or os.path.splitext(args.results)[0] + ".png"
    plot_results(result, out)
    print(out)


if __name__ == "__main__":
    main()
