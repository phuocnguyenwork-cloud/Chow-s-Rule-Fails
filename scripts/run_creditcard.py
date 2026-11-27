#!/usr/bin/env python
"""Real-data experiment on Credit Card Fraud (OpenML id 1597 or a local CSV).

  python scripts/run_creditcard.py --csv data/creditcard.csv --n-splits 20
  python scripts/run_creditcard.py --openml --n-splits 1        # needs internet

Per split: stratified 60/20/20 -> tier 1 (logistic regression on a feature subset)
-> Platt calibration on OOF TRAIN -> tier 2 (gradient boosting, all features) ->
t2 from out-of-fold TRAIN predictions -> bins from TRAIN -> policies chosen on
VAL -> cost and paired-bootstrap regret on TEST.

The existing tier-1 feature subset is fixed; no subset search.
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from threeaction.calibrate import PlattCalibrator, ece, reliability_table   # noqa: E402
from threeaction.costs import C_MISS, C_FALSE_BLOCK, C_ESCALATE              # noqa: E402
from threeaction.pipeline import run_policies                                # noqa: E402
from threeaction.data_split import feature_groups, grouped_split, validate_split
from threeaction.tier2 import choose_t2, bayes_t2                            # noqa: E402

ALL_FEATURES = [ "Amount"] + [f"V{i}" for i in range(1, 29)]
TIER1_FEATURES = [ "Amount"] + [f"V{i}" for i in range(1, 11)]  # fixed feature set


def load(csv=None, openml=False, dedupe=False):
    if csv:
        df = pd.read_csv(csv)
    elif openml:
        from sklearn.datasets import fetch_openml
        df = fetch_openml(data_id=1597, as_frame=True).frame
    else:
        raise SystemExit("give --csv PATH or --openml")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    label = "Class" if "Class" in df.columns else df.columns[-1]
    df[label] = pd.to_numeric(df[label], errors="raise")
    if not np.isin(df[label], [0, 1]).all():
        raise ValueError("Labels must be binary")
    df[label] = df[label].astype(int)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    df[ALL_FEATURES] = df[ALL_FEATURES].astype(np.float64)
    feature_groups(df[ALL_FEATURES])
    n0 = len(df)
    if dedupe:
        df = df.drop_duplicates(subset=ALL_FEATURES + [label]).reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} (dedupe={dedupe}); frauds: {int(df[label].sum())} "
          f"({100 * df[label].mean():.3f}%)")
    group_ids = feature_groups(df[ALL_FEATURES])
    return df[ALL_FEATURES].reset_index(drop=True), df[label].to_numpy(), group_ids


def get_group_cv_folds(groups_tr, y_tr, n_splits=5, seed=0):
    """Generate group-aware stratified cross-validation folds on train."""
    unique_groups, group_first, group_inv = np.unique(
        groups_tr, return_index=True, return_inverse=True
    )
    group_y = np.zeros(len(unique_groups), dtype=int)
    np.maximum.at(group_y, group_inv, y_tr)
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


def one_split(X, y, group_ids, seed, K, method, min_pos, n_boot, splits_archive=None,
              c_escalate=C_ESCALATE, t2_mode="bayes"):
    # Derive identity from features, never trust externally supplied grouping.
    group_ids = feature_groups(X[ALL_FEATURES])
    if splits_archive is not None:
        required = [f"{part}_{seed}" for part in ("train", "val", "test")]
        if any(k not in splits_archive for k in required):
            raise ValueError(f"Incomplete frozen split for seed {seed}")
        tr, va, te = [splits_archive[k] for k in required]
    else:
        tr, va, te = grouped_split(y, group_ids, seed)
    validate_split(y, group_ids, tr, va, te)

    # Group-aware CV folds inside train for both Tier 1 and Tier 2
    cv_folds_tr = get_group_cv_folds(group_ids[tr], y[tr], n_splits=5, seed=seed)

    # tier 1: OOF predictions on TRAIN for Platt calibration (group-aware CV, no val leakage)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p_raw_oof_tr = cross_val_predict(
        lr,
        X.iloc[tr][TIER1_FEATURES], y[tr],
        cv=cv_folds_tr,
        method="predict_proba",
    )[:, 1]
    cal = PlattCalibrator().fit(p_raw_oof_tr, y[tr])  # Platt fit on OOF/train only
    # Final tier-1 model fit on all train; used for val and test scoring
    lr.fit(X.iloc[tr][TIER1_FEATURES], y[tr])
    p_raw_full = lr.predict_proba(X[TIER1_FEATURES])[:, 1]
    # Build score array: train uses OOF raw -> Platt; val/test use full-fit raw -> Platt
    s = cal.transform(p_raw_full)          # Platt applied to full-fit raw for val/test
    s[tr] = cal.transform(p_raw_oof_tr)   # train rows override with OOF-based score

    # tier 2: OOF predictions on TRAIN, group-aware CV
    gb = HistGradientBoostingClassifier(random_state=seed)
    p2_raw_oof_tr = cross_val_predict(
        gb,
        X.iloc[tr][ALL_FEATURES], y[tr],
        cv=cv_folds_tr,
        method="predict_proba",
    )[:, 1]

    # Platt calibration for Tier 2 using train OOF predictions
    cal2 = PlattCalibrator().fit(p2_raw_oof_tr, y[tr])
    p2_oof_cal = cal2.transform(p2_raw_oof_tr)

    # Fit final Tier 2 model on all train features
    gb.fit(X.iloc[tr][ALL_FEATURES], y[tr])
    p2_raw_full = gb.predict_proba(X[ALL_FEATURES])[:, 1]
    p2_full_cal = cal2.transform(p2_raw_full)

    # Build tier-2 calibrated probabilities: train rows use OOF
    p2 = p2_full_cal.copy()
    p2[tr] = p2_oof_cal

    # Threshold t2: Bayes theoretical (1/101) or empirical search
    if t2_mode == "bayes":
        t2 = bayes_t2(C_MISS, C_FALSE_BLOCK)
    else:
        t2, _ = choose_t2(p2[tr], y[tr])

    block2 = (p2 >= t2).astype(int)

    parts = [{"s": s[i], "y": y[i], "tier2_block": block2[i]} for i in (tr, va, te)]
    res = run_policies(*parts, K=K, method=method, min_pos=min_pos,
                       n_boot=n_boot, seed=seed, c_escalate=c_escalate)
    res.update(
        t2=float(t2),
        t2_mode=t2_mode,
        c_escalate=float(c_escalate),
        ece_test=float(ece(s[te], y[te])),
        ece_test_t2=float(ece(p2[te], y[te])),
        reliability=reliability_table(s[te], y[te], 10),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--openml", action="store_true")
    ap.add_argument("--n-splits", type=int, default=1)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--method", default="logit", choices=["quantile", "logit", "fraud_quantile"])
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--c-escalate", type=float, default=C_ESCALATE,
                    help=f"Cost of escalation (default: {C_ESCALATE})")
    ap.add_argument("--t2-mode", default="bayes", choices=["bayes", "empirical"],
                    help="Tier 2 threshold selection: 'bayes' (fixed 1/101) or 'empirical' (default: bayes)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop duplicates (for sensitivity analysis only; default: False)")
    ap.add_argument("--splits", default=os.path.join(ROOT, "data", "splits_20.npz"),
                    help="Path to frozen splits npz (optional)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "creditcard_results.json"))
    a = ap.parse_args()

    splits_archive = None
    if a.splits and os.path.exists(a.splits):
        splits_archive = np.load(a.splits)
        print(f"Loaded frozen splits from {a.splits}")

    X, y, group_ids = load(a.csv, a.openml, dedupe=a.dedupe)
    rows = []
    t0 = time.time()
    for seed in range(a.n_splits):
        r = one_split(X, y, group_ids, seed, a.K, a.method, a.min_pos, a.n_boot,
                      splits_archive=splits_archive, c_escalate=a.c_escalate, t2_mode=a.t2_mode)
        rg = r["test_regret"]
        print(f"split {seed:2d}: K_eff={r['K_effective']:2d} t2={r['t2']:.4f} (mode={a.t2_mode}, c={a.c_escalate}) "
              f"test cost two={r['test_cost_two']:.5f} bin={r['test_cost_bin']:.5f} "
              f"regret={rg['abs']:+.5f} ({100 * rg['rel']:+.2f}%) ECE(t1={r['ece_test']:.5f}, t2={r['ece_test_t2']:.5f}) "
              f"esc_rate(two={100*r['escalate_rate_two']:.2f}%, exact={100*r['escalate_rate_exact']:.2f}%)")
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
                     "t2_mode": a.t2_mode, "c_escalate": a.c_escalate,
                     "t_low": r["t_low"], "t_high": r["t_high"],
                     "test_cost_two": r["test_cost_two"], "test_cost_bin": r["test_cost_bin"],
                     "escalate_rate_two": r["escalate_rate_two"],
                     "escalate_rate_bin": r["escalate_rate_bin"],
                     "escalate_rate_exact": r["escalate_rate_exact"],
                     "escalate_rate_roc": r["escalate_rate_roc"],
                     "regret_abs": rg["abs"], "regret_rel": rg["rel"],
                     "regret_rel_ci": list(rg["rel_ci"]),
                     "per_bin_two_threshold_shape": r["per_bin_is_two_threshold_shape"],
                     "ece_test": r["ece_test"],
                     "ece_test_t2": r["ece_test_t2"]})
    ct = np.mean([r["test_cost_two"] for r in rows]); cb = np.mean([r["test_cost_bin"] for r in rows])
    cr = np.mean([r["roc_test_cost"] for r in rows])
    cx = np.mean([r["exact_two_threshold_test_cost"] for r in rows])
    agg = {"c_escalate": a.c_escalate, "t2_mode": a.t2_mode,
           "mean_cost_roc": cr,
           "mean_cost_exact_two": cx,
           "mean_rel_regret_roc": (cr - cb) / cb if cb > 0 else None,
           "mean_cost_two": ct, "mean_cost_bin": cb, "mean_abs_regret": ct - cb,
           "mean_rel_regret": (ct - cb) / cb if cb > 0 else None,
           "mean_escalate_rate_two": float(np.mean([r["escalate_rate_two"] for r in rows])),
           "mean_escalate_rate_exact": float(np.mean([r["escalate_rate_exact"] for r in rows])),
           "mean_escalate_rate_roc": float(np.mean([r["escalate_rate_roc"] for r in rows])),
           "std_abs_regret_across_splits": float(np.std([r["regret_abs"] for r in rows]))}
    print("\nAGGREGATE over", a.n_splits, "split(s):", json.dumps(agg, indent=2))
    print(f"elapsed {time.time() - t0:.1f}s")
    import hashlib
    import subprocess

    def file_sha256(path):
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    provenance = {
        "git_commit": git_commit,
        "csv_sha256": file_sha256(a.csv),
        "splits_sha256": file_sha256(a.splits),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    from pathlib import Path
    import platform
    import sklearn
    provenance.update({
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_hashes": {str(p.relative_to(ROOT)): file_sha256(p)
                          for folder in ("src", "scripts")
                          for p in (Path(ROOT) / folder).rglob("*.py")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "command": sys.argv,
        "tier1_features": TIER1_FEATURES, "tier2_features": ALL_FEATURES,
        "protocol": "retained-row group splits; train OOF Platt for both tiers; explicit t2_mode",
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"provenance": provenance, "args": vars(a), "splits": rows, "aggregate": agg}, f, indent=2)


if __name__ == "__main__":
    main()
