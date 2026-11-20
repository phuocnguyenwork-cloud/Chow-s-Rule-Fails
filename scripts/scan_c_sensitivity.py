#!/usr/bin/env python
"""Sensitivity scan for escalation cost c across frozen 20 splits.

Evaluates policies across c in {0.01, 0.02, 0.05, 0.1, 0.2} with:
  1. Tier 1: Logistic Regression + group-aware OOF Platt calibration.
  2. Tier 2: Gradient Boosting + group-aware OOF Platt calibration.
  3. Bayes-optimal Tier 2 threshold t2 = C_FB / (C_MISS + C_FB) = 1/101.
  4. Models are fit ONCE per split and evaluated across all c values in seconds.

Produces:
  outputs/c_sensitivity_results.json: complete split-by-split and aggregate metrics.

Usage:
  python scripts/scan_c_sensitivity.py
  python scripts/scan_c_sensitivity.py --n-splits 20 --c-list 0.01,0.02,0.05,0.1,0.2
"""
import argparse
import json
import os
import sys
import time

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from threeaction.calibrate import PlattCalibrator, ece, reliability_table
from threeaction.costs import C_MISS, C_FALSE_BLOCK
from threeaction.pipeline import run_policies
from threeaction.tier2 import bayes_t2

ALL_FEATURES = ["Amount"] + [f"V{i}" for i in range(1, 29)]
TIER1_FEATURES = ["Amount"] + [f"V{i}" for i in range(1, 11)]


def get_group_cv_folds(groups_tr, y_tr, n_splits=5, seed=0):
    """Generate group-aware stratified cross-validation folds on train."""
    unique_groups, group_first, group_inv = np.unique(
        groups_tr, return_index=True, return_inverse=True
    )
    group_y = y_tr[group_first]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_fold = np.empty(len(unique_groups), dtype=np.int8)
    for f_idx, (_, te_grp_idx) in enumerate(skf.split(unique_groups, group_y)):
        group_fold[te_grp_idx] = f_idx
    sample_fold = group_fold[group_inv]
    folds = []
    for i in range(n_splits):
        cv_tr = np.where(sample_fold != i)[0]
        cv_va = np.where(sample_fold == i)[0]
        folds.append((cv_tr, cv_va))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "creditcard.csv"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data", "splits_20.npz"))
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--c-list", default="0.01,0.02,0.05,0.1,0.2",
                    help="Comma-separated list of escalation costs c")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--method", default="logit", choices=["quantile", "logit", "fraud_quantile"])
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "c_sensitivity_results.json"))
    a = ap.parse_args()

    c_values = [float(x.strip()) for x in a.c_list.split(",") if x.strip()]
    print(f"=== QUÉT ĐỘ NHẠY CHI PHÍ CHUYỂN TẦNG c ===")
    print(f"Dải c: {c_values}")
    print(f"Số splits: {a.n_splits}")

    if not os.path.exists(a.csv):
        raise FileNotFoundError(f"CSV file not found: {a.csv}")
    if not os.path.exists(a.splits):
        raise FileNotFoundError(f"Splits archive not found: {a.splits}")

    df = pd.read_csv(a.csv)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    label = "Class"
    y = df[label].to_numpy(dtype=int)
    X = df[ALL_FEATURES].reset_index(drop=True)
    group_ids = df.groupby(ALL_FEATURES, sort=False).ngroup().to_numpy(dtype=np.int64)
    splits_archive = np.load(a.splits)

    t2 = bayes_t2(C_MISS, C_FALSE_BLOCK)  # 1/101 approx 0.009901
    print(f"Tier 2 Bayes threshold t2: {t2:.6f} (C_MISS={C_MISS}, C_FB={C_FALSE_BLOCK})")

    # Storage structure: per_c_splits[c] = list of split results
    per_c_splits = {c: [] for c in c_values}

    t0_total = time.time()
    for seed in range(a.n_splits):
        t0_split = time.time()
        tr = splits_archive[f"train_{seed}"]
        va = splits_archive[f"val_{seed}"]
        te = splits_archive[f"test_{seed}"]

        cv_folds_tr = get_group_cv_folds(group_ids[tr], y[tr], n_splits=5, seed=seed)

        # 1. Tier 1: LR + OOF Platt calibration
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p_raw_oof_tr = cross_val_predict(
            lr,
            X.iloc[tr][TIER1_FEATURES], y[tr],
            cv=cv_folds_tr,
            method="predict_proba",
        )[:, 1]
        cal1 = PlattCalibrator().fit(p_raw_oof_tr, y[tr])
        lr.fit(X.iloc[tr][TIER1_FEATURES], y[tr])
        p_raw_full = lr.predict_proba(X[TIER1_FEATURES])[:, 1]
        s = cal1.transform(p_raw_full)
        s[tr] = cal1.transform(p_raw_oof_tr)

        # 2. Tier 2: GB + OOF Platt calibration + Bayes decision
        gb = HistGradientBoostingClassifier(random_state=seed)
        p2_raw_oof_tr = cross_val_predict(
            gb,
            X.iloc[tr][ALL_FEATURES], y[tr],
            cv=cv_folds_tr,
            method="predict_proba",
        )[:, 1]
        cal2 = PlattCalibrator().fit(p2_raw_oof_tr, y[tr])
        p2_oof_cal = cal2.transform(p2_raw_oof_tr)
        gb.fit(X.iloc[tr][ALL_FEATURES], y[tr])
        p2_raw_full = gb.predict_proba(X[ALL_FEATURES])[:, 1]
        p2_full_cal = cal2.transform(p2_raw_full)
        p2 = p2_full_cal.copy()
        p2[tr] = p2_oof_cal

        block2 = (p2 >= t2).astype(int)

        parts = [{"s": s[i], "y": y[i], "tier2_block": block2[i]} for i in (tr, va, te)]

        # 3. Evaluate across all c values
        for c in c_values:
            res = run_policies(*parts, K=a.K, method=a.method, min_pos=a.min_pos,
                               n_boot=a.n_boot, seed=seed, c_escalate=c)
            o = res["roc"]
            ex = res["exact_two_threshold"]
            rg = res["test_regret"]

            entry = {
                "seed": seed,
                "c_escalate": c,
                "t2": t2,
                "K_eff": res["K_effective"],
                "t_low": res["t_low"],
                "t_high": res["t_high"],
                "test_cost_two": res["test_cost_two"],
                "test_cost_bin": res["test_cost_bin"],
                "regret_abs": rg["abs"],
                "regret_rel": rg["rel"],
                "regret_rel_ci": list(rg["rel_ci"]),
                "per_bin_two_threshold_shape": res["per_bin_is_two_threshold_shape"],
                "exact_two_threshold_test_cost": ex["test_cost"],
                "roc_test_cost": o["test_cost"],
                "roc_snapped_test_cost": o["snapped_test_cost"],
                "roc_t_low": o["t_low"],
                "roc_t_high": o["t_high"],
                "roc_a_plus": o["a_plus"],
                "roc_a_minus": o["a_minus"],
                "escalate_rate_two": res["escalate_rate_two"],
                "escalate_rate_bin": res["escalate_rate_bin"],
                "escalate_rate_exact": res["escalate_rate_exact"],
                "escalate_rate_roc": res["escalate_rate_roc"],
                "ece_test_t1": float(ece(s[te], y[te])),
                "ece_test_t2": float(ece(p2[te], y[te])),
            }
            per_c_splits[c].append(entry)

        elapsed_split = time.time() - t0_split
        print(f"Split {seed:2d}/20 hoàn thành trong {elapsed_split:.1f}s")

    # Aggregate summaries per c
    aggregates = {}
    print("\n" + "=" * 95)
    print(f"{'c':>6} | {'Cost Two':>10} | {'Cost Bin':>10} | {'Cost Exact':>10} | {'Cost ROC':>10} | {'Rel Regret':>11} | {'Esc Two':>8} | {'Esc Exact':>9} | {'Esc ROC':>8}")
    print("-" * 95)

    for c in c_values:
        entries = per_c_splits[c]
        c_two = float(np.mean([e["test_cost_two"] for e in entries]))
        c_bin = float(np.mean([e["test_cost_bin"] for e in entries]))
        c_exact = float(np.mean([e["exact_two_threshold_test_cost"] for e in entries]))
        c_roc = float(np.mean([e["roc_test_cost"] for e in entries]))
        abs_regret = float(np.mean([e["regret_abs"] for e in entries]))
        rel_regret = (c_two - c_bin) / c_bin if c_bin > 0 else 0.0
        esc_two = float(np.mean([e["escalate_rate_two"] for e in entries]))
        esc_exact = float(np.mean([e["escalate_rate_exact"] for e in entries]))
        esc_roc = float(np.mean([e["escalate_rate_roc"] for e in entries]))
        ci_lower = float(np.mean([e["regret_rel_ci"][0] for e in entries]))
        ci_upper = float(np.mean([e["regret_rel_ci"][1] for e in entries]))

        aggregates[str(c)] = {
            "c_escalate": c,
            "mean_cost_two": c_two,
            "mean_cost_bin": c_bin,
            "mean_cost_exact_two": c_exact,
            "mean_cost_roc": c_roc,
            "mean_abs_regret": abs_regret,
            "mean_rel_regret": rel_regret,
            "mean_rel_regret_ci95": [ci_lower, ci_upper],
            "mean_escalate_rate_two": esc_two,
            "mean_escalate_rate_exact": esc_exact,
            "mean_escalate_rate_roc": esc_roc,
            "splits": entries,
        }
        print(f"{c:6.2f} | {c_two:10.5f} | {c_bin:10.5f} | {c_exact:10.5f} | {c_roc:10.5f} | {100*rel_regret:+10.2f}% | {100*esc_two:7.2f}% | {100*esc_exact:8.2f}% | {100*esc_roc:7.2f}%")

    print("=" * 95)
    total_elapsed = time.time() - t0_total
    print(f"Toàn bộ 20 splits x 5 giá trị c hoàn thành trong {total_elapsed:.1f}s!")

    output_data = {
        "metadata": {
            "experiment": "c_sensitivity_scan_with_bayes_tier2",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_splits": a.n_splits,
            "c_values": c_values,
            "t2_fixed_bayes": t2,
            "tier1_features": TIER1_FEATURES,
            "tier2_features": ALL_FEATURES,
            "total_elapsed_seconds": total_elapsed,
        },
        "aggregates": aggregates,
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả chi tiết vào: {a.out}")


if __name__ == "__main__":
    main()
