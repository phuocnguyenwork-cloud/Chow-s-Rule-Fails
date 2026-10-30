"""Tests for the ROC-optimal baseline. Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from threeaction.calibrate import PlattCalibrator                        # noqa: E402
from threeaction.costs import per_txn_costs                              # noqa: E402
from threeaction.pipeline import run_policies                            # noqa: E402
from threeaction.roc_baseline import (class_escalation_costs,            # noqa: E402
                                      closed_form_thresholds,
                                      cost_from_thresholds,
                                      exact_two_threshold,
                                      roc_optimal_thresholds)
from threeaction.synthetic import SyntheticConfig, analytic_costs, generate  # noqa: E402


def make_parts(scenario, seed, n=200_000, y_te_override=None):
    d = generate(SyntheticConfig(scenario=scenario, n=n), seed=seed)
    idx = np.arange(d["y"].size)
    tr, rest = train_test_split(idx, test_size=0.4, stratify=d["y"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=d["y"][rest], random_state=seed)
    s = PlattCalibrator().fit(d["p_raw"][va], d["y"][va]).transform(d["p_raw"])
    parts = [{"s": s[i], "y": d["y"][i], "tier2_block": d["tier2_block"][i]} for i in (tr, va, te)]
    if y_te_override is not None:
        parts[2]["y"] = y_te_override(parts[2]["y"])
    return parts


class TestRocImplementation(unittest.TestCase):
    def test1_perfect_tier2_gives_chow_thresholds(self):
        """Tier 2 never errs: thresholds ~ 0.002 and 0.8, and cost ~ cost at those values."""
        d = generate(SyntheticConfig(scenario="perfect", n=400_000), seed=0)
        s, y, b = d["s_true"], d["y"], d["tier2_block"]
        a_plus, a_minus = class_escalation_costs(y, b)
        self.assertAlmostEqual(a_plus, 0.2)
        self.assertAlmostEqual(a_minus, 0.2)
        t_low, t_high, info = roc_optimal_thresholds(s, y, a_plus, a_minus)
        self.assertTrue(info["escalates"])
        self.assertTrue(0.0012 < t_low < 0.0035, t_low)
        self.assertTrue(0.65 < t_high < 0.93, t_high)
        costs = per_txn_costs(y, b)
        c_roc = cost_from_thresholds(s, t_low, t_high, costs).mean()
        c_chow = cost_from_thresholds(s, 0.002, 0.8, costs).mean()
        self.assertLessEqual(c_roc, c_chow * 1.01)

    def test2_tangent_search_equals_brute_force_of_its_own_model(self):
        """The ROC shortcut must find the best pair for the constant-cost model."""
        rng = np.random.default_rng(0)
        n = 1200
        s = rng.random(n) ** 3
        y = (rng.random(n) < s).astype(int)
        a_plus, a_minus, c_fn, c_fp = 12.0, 0.4, 100.0, 1.0
        t_low, t_high, info = roc_optimal_thresholds(s, y, a_plus, a_minus)
        self.assertTrue(info["escalates"])
        pp, pm = y.mean(), 1 - y.mean()

        def model_cost(tl, th):
            pos, neg = s[y == 1], s[y == 0]
            p_pass, p_blk = (pos < tl).mean(), (pos >= th).mean()
            n_pass, n_blk = (neg < tl).mean(), (neg >= th).mean()
            return (pp * (c_fn * p_pass + a_plus * (1 - p_pass - p_blk))
                    + pm * (c_fp * n_blk + a_minus * (1 - n_pass - n_blk)))

        cuts = np.concatenate([[-np.inf], np.unique(s), [np.inf]])
        best = min(model_cost(a, b) for i, a in enumerate(cuts) for b in cuts[i:])
        self.assertAlmostEqual(model_cost(t_low, t_high), best, places=10)

    def test3_extreme_escalation_costs(self):
        d = generate(SyntheticConfig(scenario="perfect", n=60_000), seed=1)
        s, y, b = d["s_true"], d["y"], d["tier2_block"]
        # free escalation + perfect tier 2 -> total cost 0
        costs0 = per_txn_costs(y, b, c_escalate=0.0)
        a_plus, a_minus = class_escalation_costs(y, b, c_escalate=0.0)
        t_low, t_high, info = roc_optimal_thresholds(s, y, a_plus, a_minus)
        self.assertTrue(info["escalates"])
        self.assertAlmostEqual(cost_from_thresholds(s, t_low, t_high, costs0).mean(), 0.0)
        # very expensive escalation -> never escalate
        a_plus, a_minus = class_escalation_costs(y, b, c_escalate=50.0)
        t_low, t_high, info = roc_optimal_thresholds(s, y, a_plus, a_minus)
        self.assertFalse(info["escalates"])
        self.assertEqual(t_low, t_high)

    def test_closed_form_matches_chow(self):
        t_low, t_high = closed_form_thresholds(0.2, 0.2)
        self.assertAlmostEqual(t_low, 0.002)
        self.assertAlmostEqual(t_high, 0.8)


class TestExactTwoThreshold(unittest.TestCase):
    def test_matches_brute_force_on_real_costs(self):
        rng = np.random.default_rng(2)
        n = 300
        s = rng.random(n)
        costs = rng.random((n, 3)) * np.array([5, 3, 2])
        t_low, t_high, c = exact_two_threshold(s, costs)
        cuts = np.concatenate([[-np.inf], np.unique(s), [np.inf]])
        best = min(cost_from_thresholds(s, a, b, costs).mean()
                   for i, a in enumerate(cuts) for b in cuts[i:])
        self.assertAlmostEqual(c, best, places=10)
        self.assertAlmostEqual(cost_from_thresholds(s, t_low, t_high, costs).mean(), best, places=10)


class TestAgainstAnalyticAnswer(unittest.TestCase):
    def test4_blind_spot_costs_match_analytic(self):
        an = analytic_costs(SyntheticConfig(scenario="blind_spot"))
        roc_c, ex_c = [], []
        for seed in (0, 1, 2):
            r = run_policies(*make_parts("blind_spot", seed), K=16, n_boot=20, seed=seed)
            roc_c.append(r["roc"]["test_cost"])
            ex_c.append(r["exact_two_threshold"]["test_cost"])
        self.assertLess(abs(np.mean(roc_c) / an["C_roc"] - 1), 0.04)
        self.assertLess(abs(np.mean(ex_c) / an["C_two"] - 1), 0.04)
        # the headline numbers of the analytic answer
        self.assertAlmostEqual(an["C_two"] / an["C_star"] - 1, 0.0907, places=3)
        self.assertAlmostEqual(an["C_roc"] / an["C_star"] - 1, 0.6279, places=3)

    def test_constant_scenario_all_policies_equal_analytically(self):
        an = analytic_costs(SyntheticConfig(scenario="constant"))
        self.assertAlmostEqual(an["C_roc"] / an["C_star"], 1.0, places=4)
        self.assertAlmostEqual(an["C_two"] / an["C_star"], 1.0, places=4)


class TestInvariantsAndLeakage(unittest.TestCase):
    def test5_invariants_on_validation(self):
        for scenario in ("perfect", "constant", "blind_spot"):
            for seed in (0, 1):
                r = run_policies(*make_parts(scenario, seed, n=120_000), K=16, n_boot=20, seed=seed)
                tol = 1e-12
                msg = f"{scenario}/{seed}"
                # ROC is one feasible pair -> cannot beat the exact optimum
                self.assertGreaterEqual(r["roc"]["val_cost"] + tol, r["exact_two_threshold"]["val_cost"], msg)
                # exact optimum over all cuts <= best pair restricted to bin edges
                self.assertLessEqual(r["exact_two_threshold"]["val_cost"], r["val_cost_two"] + tol, msg)
                # snapped ROC is a two-threshold policy on bin edges
                self.assertGreaterEqual(r["roc"]["snapped_val_cost"] + tol, r["val_cost_two"], msg)
                self.assertGreaterEqual(r["val_cost_two"] + tol, r["val_cost_bin"], msg)

    def test6_shuffling_test_labels_changes_nothing_that_is_chosen(self):
        base = run_policies(*make_parts("blind_spot", 0, n=120_000), K=16, n_boot=20, seed=0)
        rng = np.random.default_rng(0)
        shuf = run_policies(*make_parts("blind_spot", 0, n=120_000,
                                        y_te_override=lambda y: rng.permutation(y)),
                            K=16, n_boot=20, seed=0)
        for key in ("a_plus", "a_minus", "t_low", "t_high"):
            self.assertEqual(base["roc"][key], shuf["roc"][key], key)
        self.assertEqual(base["exact_two_threshold"]["t_low"], shuf["exact_two_threshold"]["t_low"])
        np.testing.assert_array_equal(base["edges"], shuf["edges"])
        np.testing.assert_array_equal(base["actions_per_bin"], shuf["actions_per_bin"])
        self.assertNotEqual(base["roc"]["test_cost"], shuf["roc"]["test_cost"])  # only scoring changes

    def test7_class_costs_use_only_the_data_given(self):
        rng = np.random.default_rng(3)
        y = (rng.random(5000) < 0.05).astype(int)
        b = np.where(y == 1, rng.random(5000) < 0.9, rng.random(5000) < 0.02).astype(int)
        a1 = class_escalation_costs(y, b)
        a2 = class_escalation_costs(y[:2500], b[:2500])
        self.assertNotEqual(a1, a2)                       # depends on the rows passed in
        self.assertEqual(class_escalation_costs(y, b), a1)  # deterministic


if __name__ == "__main__":
    unittest.main()
