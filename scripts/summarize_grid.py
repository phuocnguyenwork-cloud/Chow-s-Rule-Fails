#!/usr/bin/env python
"""

  python scripts/summarize_grid.py                 # doc outputs/grid_*.json
  python scripts/summarize_grid.py --pattern "outputs/cc_*.json"

Cot trong bang:
  K_eff          : so khoang sau khi gop (min-max qua cac split)
  cost bin/two   : chi phi test TB moi giao dich cua chinh sach theo khoang / hai nguong tren bien khoang
  regret two     : (two - bin) / bin, TB qua cac split, kem khoang [2.5%, 97.5%] qua cac split
  >0             : so split co regret > 0 (hai nguong ton hon theo khoang)
  cost ROC/exact : chi phi test TB cua baseline ROC (Pietraszek) / hai nguong quet luoi chinh xac
  ROC vs exact   : (ROC - exact) / exact, TB qua cac split; "thang" = so split exact re hon ROC
Luu y: 20 split dung chung du lieu nen khong doc lap; khoang o day chi mo ta do phan tan.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarize(path):
    d = json.load(open(path, encoding="utf-8"))
    a, rows = d["args"], d["splits"]
    keff = [r["K_eff"] for r in rows]
    reg = np.array([r["regret_rel"] for r in rows])
    roc = np.array([r["roc_test_cost"] for r in rows])
    ex = np.array([r["exact_two_threshold_test_cost"] for r in rows])
    lo, hi = np.percentile(reg, [2.5, 97.5])
    return {
        "file": os.path.basename(path),
        "method": a["method"], "K": a["K"], "n_splits": len(rows),
        "K_eff": f"{min(keff)}-{max(keff)}" if min(keff) != max(keff) else str(keff[0]),
        "cost_bin": np.mean([r["test_cost_bin"] for r in rows]),
        "cost_two": np.mean([r["test_cost_two"] for r in rows]),
        "regret_two_pct": 100 * reg.mean(),
        "regret_two_lo": 100 * lo, "regret_two_hi": 100 * hi,
        "n_regret_pos": int((reg > 0).sum()),
        "cost_roc": roc.mean(), "cost_exact": ex.mean(),
        "roc_vs_exact_pct": 100 * np.mean((roc - ex) / ex),
        "n_exact_wins": int((ex < roc).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(ROOT, "outputs", "grid_*.json"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "outputs", "grid_summary.csv"))
    a = ap.parse_args()
    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"khong thay file nao khop {a.pattern}")
    S = sorted((summarize(f) for f in files), key=lambda s: (s["method"], s["K"]))

    print("| method | K | K_eff | cost bin | cost two | regret two (%) [2.5%, 97.5%] | >0 "
          "| cost ROC | cost exact | ROC vs exact (%) | exact thang |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in S:
        n = s["n_splits"]
        print(f"| {s['method']} | {s['K']} | {s['K_eff']} | {s['cost_bin']:.5f} | {s['cost_two']:.5f} "
              f"| {s['regret_two_pct']:+.2f} [{s['regret_two_lo']:+.2f}, {s['regret_two_hi']:+.2f}] "
              f"| {s['n_regret_pos']}/{n} | {s['cost_roc']:.5f} | {s['cost_exact']:.5f} "
              f"| {s['roc_vs_exact_pct']:+.2f} | {s['n_exact_wins']}/{n} |")

    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(S[0]))
        w.writeheader()
        w.writerows(S)
    print(f"\nda ghi {a.csv}")


if __name__ == "__main__":
    main()