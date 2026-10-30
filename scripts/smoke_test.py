#!/usr/bin/env python
"""End-to-end smoke test on synthetic data (no network, ~10 s).

Runs the full chain:
  generate -> split 60/20/20 -> Platt calibration -> bins from train ->
  cost table on val -> per-bin policy + best two-threshold ->
  cost / paired-bootstrap regret on test
for two scenarios whose correct answer is known:

  constant   : tier-2 error rates do not depend on the score -> regret ~ 0
  blind_spot : tier-2 is blind in a score window -> fragmented optimum, regret > 0

Exit code 0 and the line 'SMOKE TEST PASSED' mean the pipeline is wired correctly.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from threeaction.calibrate import PlattCalibrator, ece          # noqa: E402
from threeaction.costs import ACTION_NAMES                      # noqa: E402
from threeaction.pipeline import run_policies                   # noqa: E402
from threeaction.synthetic import SyntheticConfig, generate     # noqa: E402

K, METHOD, MIN_POS, N_BOOT = 16, "logit", 5, 300


def one_split(scenario, seed):
    d = generate(SyntheticConfig(scenario=scenario), seed=seed)
    idx = np.arange(d["y"].size)
    tr, rest = train_test_split(idx, test_size=0.4, stratify=d["y"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=d["y"][rest], random_state=seed)

    cal = PlattCalibrator().fit(d["p_raw"][va], d["y"][va])     # learned on VAL only
    s = cal.transform(d["p_raw"])
    ece_raw = ece(d["p_raw"][te], d["y"][te])
    ece_cal = ece(s[te], d["y"][te])

    parts = [{"s": s[i], "y": d["y"][i], "tier2_block": d["tier2_block"][i]}
             for i in (tr, va, te)]
    res = run_policies(*parts, K=K, method=METHOD, min_pos=MIN_POS,
                       n_boot=N_BOOT, seed=seed)
    res["ece_raw"], res["ece_cal"] = ece_raw, ece_cal
    return res


def fmt_actions(a):
    return " ".join(ACTION_NAMES[x][0] for x in a)   # P / E / B per bin


def main():
    seeds = {"constant": [0, 1, 2, 3], "blind_spot": [0, 1, 2]}
    summary, checks = {}, []

    for scenario, seed_list in seeds.items():
        print(f"\n=== scenario: {scenario} ===")
        rels, abss, shapes = [], [], []
        roc_rel_vs_exact = []
        for seed in seed_list:
            r = one_split(scenario, seed)
            rg = r["test_regret"]
            rels.append(rg["rel"]); abss.append(rg["abs"])
            shapes.append(r["per_bin_is_two_threshold_shape"])
            print(f" seed {seed}: K_eff={r['K_effective']:2d}  per-bin: {fmt_actions(r['actions_per_bin'])}"
                  f"  | two-thr: {fmt_actions(r['actions_two_threshold'])}")
            print(f"          test cost two={r['test_cost_two']:.4f} bin={r['test_cost_bin']:.4f}"
                  f"  regret={rg['abs']:+.4f} ({100*rg['rel']:+.1f}%)"
                  f" CI95 rel [{100*rg['rel_ci'][0]:+.1f}%, {100*rg['rel_ci'][1]:+.1f}%]"
                  f"  ECE raw->cal {r['ece_raw']:.4f}->{r['ece_cal']:.4f}")
            o, ex = r["roc"], r["exact_two_threshold"]
            roc_rel_vs_exact.append(o["test_cost"] / ex["test_cost"] - 1.0)
            print(f"          ROC baseline: a+={o['a_plus']:.2f} a-={o['a_minus']:.3f} "
                  f"t=({o['t_low']:.5f}, {o['t_high']:.4f}) | test cost roc={o['test_cost']:.4f} "
                  f"exact-two-thr={ex['test_cost']:.4f}  roc vs per-bin "
                  f"{100*o['regret_vs_per_bin']['rel']:+.1f}%")
            checks.append((f"{scenario}/seed{seed}: ROC val cost >= exact two-threshold optimum",
                           o["val_cost"] >= ex["val_cost"] - 1e-12))
            checks.append((f"{scenario}/seed{seed}: snapped ROC val cost >= best two-threshold on bins",
                           o["snapped_val_cost"] >= r["val_cost_two"] - 1e-12))
            # invariants that must hold for ANY data
            checks.append((f"{scenario}/seed{seed}: val cost per-bin <= two-threshold",
                           r["val_cost_bin"] <= r["val_cost_two"] + 1e-12))
            checks.append((f"{scenario}/seed{seed}: costs finite and positive",
                           np.isfinite([r["test_cost_two"], r["test_cost_bin"]]).all()
                           and r["test_cost_bin"] > 0))
            checks.append((f"{scenario}/seed{seed}: calibration lowers ECE",
                           r["ece_cal"] < r["ece_raw"]))
        summary[scenario] = {"mean_rel_regret": float(np.mean(rels)),
                             "mean_abs_regret": float(np.mean(abss)),
                             "per_bin_not_two_threshold_shape": int(sum(not s for s in shapes)),
                             "mean_roc_extra_cost_vs_exact_two_threshold": float(np.mean(roc_rel_vs_exact)),
                             "n_seeds": len(seed_list)}
        print(f" -> ROC test cost vs exact best two-threshold: {100*np.mean(roc_rel_vs_exact):+.1f}% "
              f"(mean over seeds)")
        print(f" -> mean relative regret {100*np.mean(rels):+.2f}%  "
              f"(per-bin policy left the two-threshold shape in "
              f"{sum(not s for s in shapes)}/{len(shapes)} seeds)")

    # scenario-level expectations (the known answers)
    checks.append(("constant: |mean relative regret| < 2%",
                   abs(summary["constant"]["mean_rel_regret"]) < 0.02))
    checks.append(("blind_spot: mean relative regret > 3%",
                   summary["blind_spot"]["mean_rel_regret"] > 0.03))
    checks.append(("blind_spot: per-bin policy is fragmented in every seed",
                   summary["blind_spot"]["per_bin_not_two_threshold_shape"]
                   == summary["blind_spot"]["n_seeds"]))

    checks.append(("constant: ROC within 3% of exact two-threshold (constant-cost model is right)",
                   abs(summary["constant"]["mean_roc_extra_cost_vs_exact_two_threshold"]) < 0.03))
    checks.append(("blind_spot: ROC costs > 10% more than exact two-threshold (constant-cost model is wrong)",
                   summary["blind_spot"]["mean_roc_extra_cost_vs_exact_two_threshold"] > 0.10))

    print("\n=== checks ===")
    ok = True
    for name, passed in checks:
        print(f" [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "smoke_results.json"), "w") as f:
        json.dump({"config": {"K": K, "method": METHOD, "min_pos": MIN_POS,
                              "n_boot": N_BOOT},
                   "summary": summary,
                   "checks": {n: bool(p) for n, p in checks}}, f, indent=2)

    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
