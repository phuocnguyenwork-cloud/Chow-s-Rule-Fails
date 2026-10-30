#!/usr/bin/env python
"""Real-data experiment on Credit Card Fraud (OpenML id 1597 or a local CSV).

  python scripts/run_creditcard.py --csv data/creditcard.csv --n-splits 20
  python scripts/run_creditcard.py --openml --n-splits 1        # needs internet

Per split: stratified 60/20/20 -> tier 1 (logistic regression on a feature subset)
-> Platt calibration on VAL -> tier 2 (gradient boosting, all features) ->
t2 from out-of-fold TRAIN predictions -> bins from TRAIN -> policies chosen on
VAL -> cost and paired-bootstrap regret on TEST.

The tier-1 feature subset below is a PLACEHOLDER: the group must decide it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from threeaction.calibrate import PlattCalibrator, ece, reliability_table   # noqa: E402
from threeaction.pipeline import run_policies                                # noqa: E402
from threeaction.tier2 import choose_t2                                      # noqa: E402
from plot_creditcard import plot_results                                       # noqa: E402

REQUIRED_FEATURES = ["Amount"] + [f"V{i}" for i in range(1, 29)]
TIER1_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 11)]  # placeholder


def load(csv=None, openml=False, dedupe=True):
    if csv:
        df = pd.read_csv(csv)
    elif openml:
        from sklearn.datasets import fetch_openml
        df = fetch_openml(data_id=1597, as_frame=True,
                          data_home=os.path.join(ROOT, "data", "openml")).frame
    else:
        raise SystemExit("give --csv PATH or --openml")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    label = "Class" if "Class" in df.columns else df.columns[-1]
    df[label] = df[label].astype(int)
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    # OpenML 1597 omits Time; the common Kaggle CSV contains it.
    all_features = (["Time"] if "Time" in df.columns else []) + REQUIRED_FEATURES
    n0 = len(df)
    if dedupe:
        df = df.drop_duplicates(subset=all_features + [label]).reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after dedupe={dedupe}; frauds: {int(df[label].sum())} "
          f"({100 * df[label].mean():.3f}%)")
    print(f"features: {len(all_features)} total; Time present: {'Time' in all_features}")
    return df[all_features].reset_index(drop=True), df[label].to_numpy()


def one_split(X, y, seed, K, method, min_pos, n_boot):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)

    # tier 1 + Platt (learned on VAL only)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    tier1_features = [c for c in TIER1_FEATURES if c in X.columns]
    lr.fit(X.iloc[tr][tier1_features], y[tr])
    p_raw = lr.predict_proba(X[tier1_features])[:, 1]
    cal = PlattCalibrator().fit(p_raw[va], y[va])
    s = cal.transform(p_raw)

    # tier 2 and its fixed threshold t2 (out-of-fold on TRAIN)
    gb = HistGradientBoostingClassifier(random_state=seed)
    p2_oof = cross_val_predict(gb, X.iloc[tr], y[tr],
                               cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                               method="predict_proba")[:, 1]
    t2, _ = choose_t2(p2_oof, y[tr])
    gb.fit(X.iloc[tr], y[tr])
    p2 = gb.predict_proba(X)[:, 1]
    block2 = (p2 >= t2).astype(int)
    block2[tr] = (p2_oof >= t2).astype(int)       # train rows use out-of-fold decisions

    parts = [{"s": s[i], "y": y[i], "tier2_block": block2[i]} for i in (tr, va, te)]
    res = run_policies(*parts, K=K, method=method, min_pos=min_pos,
                       n_boot=n_boot, seed=seed)
    res.update(t2=float(t2), ece_test=float(ece(s[te], y[te])),
               reliability=reliability_table(s[te], y[te], 10))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--openml", action="store_true")
    ap.add_argument("--n-splits", type=int, default=1)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--method", default="logit", choices=["quantile", "logit", "fraud_quantile"])
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "creditcard_results.json"))
    a = ap.parse_args()

    X, y = load(a.csv, a.openml, dedupe=not a.no_dedupe)
    rows = []
    t0 = time.time()
    for seed in range(a.n_splits):
        r = one_split(X, y, seed, a.K, a.method, a.min_pos, a.n_boot)
        rg = r["test_regret"]
        print(f"split {seed:2d}: K_eff={r['K_effective']:2d} t2={r['t2']:.4f} "
              f"test cost two={r['test_cost_two']:.5f} bin={r['test_cost_bin']:.5f} "
              f"regret={rg['abs']:+.5f} ({100 * rg['rel']:+.2f}%) ECE={r['ece_test']:.5f}")
        o, ex = r["roc"], r["exact_two_threshold"]
        print(f"          ROC: a+={o['a_plus']:.2f} a-={o['a_minus']:.3f} t=({o['t_low']:.5f},"
              f" {o['t_high']:.4f}) test cost={o['test_cost']:.5f} "
              f"| exact two-thr test cost={ex['test_cost']:.5f}")
        rows.append({"roc_a_plus": o["a_plus"], "roc_a_minus": o["a_minus"],
                     "roc_t_low": o["t_low"], "roc_t_high": o["t_high"],
                     "roc_test_cost": o["test_cost"],
                     "roc_snapped_test_cost": o["snapped_test_cost"],
                     "roc_regret_vs_per_bin_rel": o["regret_vs_per_bin"]["rel"],
                     "exact_two_threshold_test_cost": ex["test_cost"],
                     "seed": seed, "K_eff": r["K_effective"], "t2": r["t2"],
                     "t_low": r["t_low"], "t_high": r["t_high"],
                     "test_cost_two": r["test_cost_two"], "test_cost_bin": r["test_cost_bin"],
                     "regret_abs": rg["abs"], "regret_rel": rg["rel"],
                     "regret_rel_ci": list(rg["rel_ci"]),
                     "per_bin_two_threshold_shape": r["per_bin_is_two_threshold_shape"],
                     "ece_test": r["ece_test"]})
    ct = np.mean([r["test_cost_two"] for r in rows]); cb = np.mean([r["test_cost_bin"] for r in rows])
    cr = np.mean([r["roc_test_cost"] for r in rows])
    agg = {"mean_cost_roc": cr,
           "mean_rel_regret_roc": (cr - cb) / cb if cb > 0 else None,
           "mean_cost_two": ct, "mean_cost_bin": cb, "mean_abs_regret": ct - cb,
           "mean_rel_regret": (ct - cb) / cb if cb > 0 else None,
           "std_abs_regret_across_splits": float(np.std([r["regret_abs"] for r in rows]))}
    print("\nAGGREGATE over", a.n_splits, "split(s):", json.dumps(agg, indent=2))
    print(f"elapsed {time.time() - t0:.1f}s")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    result = {"args": vars(a), "splits": rows, "aggregate": agg}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    figure = os.path.splitext(a.out)[0] + ".png"
    plot_results(result, figure)
    print(f"saved results: {a.out}\nsaved figure: {figure}")


if __name__ == "__main__":
    main()
