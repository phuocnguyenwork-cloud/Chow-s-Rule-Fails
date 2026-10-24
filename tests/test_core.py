"""Unit tests. Run with:  python -m unittest discover -s tests -v   (or pytest)."""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from threeaction.bins import assign_bins, make_edges, merge_min_positives   # noqa: E402
from threeaction.calibrate import PlattCalibrator, ece                      # noqa: E402
from threeaction.costs import PASS, ESCALATE, BLOCK, per_txn_costs         # noqa: E402
from threeaction.policies import (best_two_threshold, bin_cost_table,       # noqa: E402
                                  is_two_threshold_shape, per_bin_actions,
                                  thresholds_from_ij)
from threeaction.synthetic import SyntheticConfig, generate                 # noqa: E402
from threeaction.tier2 import choose_t2                                     # noqa: E402


class TestCosts(unittest.TestCase):
    def test_cost_matrix(self):
        # rows: (y, tier2_block)
        y = [1, 1, 0, 0]
        b = [1, 0, 1, 0]
        c = per_txn_costs(y, b)
        np.testing.assert_allclose(c[:, PASS], [100, 100, 0, 0])
        np.testing.assert_allclose(c[:, BLOCK], [0, 0, 1, 1])
        # escalate: fraud caught 0.2 | fraud missed 100.2 | legit blocked 1.2 | legit passed 0.2
        np.testing.assert_allclose(c[:, ESCALATE], [0.2, 100.2, 1.2, 0.2])


class TestBins(unittest.TestCase):
    def test_assign_boundaries(self):
        edges = [0.1, 0.5]
        ids = assign_bins([0.0, 0.0999, 0.1, 0.4999, 0.5, 0.99], edges)
        np.testing.assert_array_equal(ids, [0, 0, 1, 1, 2, 2])   # edge belongs to upper bin

    def test_no_edges(self):
        np.testing.assert_array_equal(assign_bins([0.2, 0.9], []), [0, 0])

    def test_merge_enforces_min_positives(self):
        rng = np.random.default_rng(0)
        s = rng.random(5000) ** 4
        y = (rng.random(5000) < s).astype(int)
        edges = make_edges(s, y, 20, "logit")
        merged = merge_min_positives(edges, s, y, m=30)
        pos = np.bincount(assign_bins(s, merged), weights=y, minlength=len(merged) + 1)
        self.assertTrue((pos >= 30).all())

    def test_all_methods_run(self):
        rng = np.random.default_rng(1)
        s = rng.random(3000) ** 3
        y = (rng.random(3000) < s).astype(int)
        for m in ("quantile", "logit", "fraud_quantile"):
            e = make_edges(s, y, 8, m)
            self.assertTrue(np.all(np.diff(e) > 0))


class TestPolicies(unittest.TestCase):
    def test_two_threshold_matches_brute_force(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            K = int(rng.integers(2, 9))
            table = rng.random((K, 3)) * 10
            act, (i, j), cost = best_two_threshold(table)
            best = min(
                sum(table[b, PASS] for b in range(ii)) +
                sum(table[b, ESCALATE] for b in range(ii, jj)) +
                sum(table[b, BLOCK] for b in range(jj, K))
                for ii in range(K + 1) for jj in range(ii, K + 1))
            self.assertAlmostEqual(cost, best)
            self.assertTrue(is_two_threshold_shape(act))

    def test_per_bin_never_worse_than_two_threshold(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            table = rng.random((int(rng.integers(1, 10)), 3))
            c_bin = table[np.arange(len(table)), per_bin_actions(table)].sum()
            _, _, c_two = best_two_threshold(table)
            self.assertLessEqual(c_bin, c_two + 1e-12)

    def test_fragmented_table_gives_positive_regret(self):
        # escalate is cheap in bins 1 and 3 but terrible in bin 2 (tier 2 blind there)
        table = np.array([[0.1, 0.3, 1.0],
                          [2.0, 0.3, 1.0],
                          [3.0, 5.0, 0.9],
                          [4.0, 0.4, 0.8],
                          [9.0, 3.0, 0.1]])
        act_bin = per_bin_actions(table)
        np.testing.assert_array_equal(act_bin, [PASS, ESCALATE, BLOCK, ESCALATE, BLOCK])
        self.assertFalse(is_two_threshold_shape(act_bin))
        c_bin = table[np.arange(5), act_bin].sum()
        _, _, c_two = best_two_threshold(table)
        self.assertGreater(c_two, c_bin)

    def test_thresholds_from_ij(self):
        edges = np.array([0.1, 0.3, 0.6])
        self.assertEqual(thresholds_from_ij(edges, 1, 3), (0.1, 0.6))
        lo, hi = thresholds_from_ij(edges, 0, 4)
        self.assertEqual((lo, hi), (-np.inf, np.inf))


class TestHandWorkedExample(unittest.TestCase):
    """Reproduces the 3-bin example worked by hand in the project notes."""

    def test_three_bin_table(self):
        # bin0: n=1000, F=1, M=0, W=10 ; bin1: n=1000, F=20, M=2, W=15 ; bin2: n=200, F=150, M=10, W=5
        y, b, ids = [], [], []
        for k, (n, F, M, W) in enumerate([(1000, 1, 0, 10), (1000, 20, 2, 15), (200, 150, 10, 5)]):
            G = n - F
            # frauds: M passed by tier 2 (b=0), rest blocked (b=1)
            y += [1] * F;  b += [0] * M + [1] * (F - M)
            # legit: W blocked by tier 2 (b=1), rest passed
            y += [0] * G;  b += [1] * W + [0] * (G - W)
            ids += [k] * n
        costs = per_txn_costs(y, b)
        table, n = bin_cost_table(np.array(ids), costs, 3)
        mean = table / n[:, None]
        np.testing.assert_allclose(mean[:, PASS], [0.10, 2.00, 75.0])
        np.testing.assert_allclose(mean[:, BLOCK], [0.999, 0.98, 0.25])
        np.testing.assert_allclose(mean[:, ESCALATE], [0.21, 0.415, 5.225])
        act = per_bin_actions(table)
        np.testing.assert_array_equal(act, [PASS, ESCALATE, BLOCK])
        self.assertAlmostEqual(table[np.arange(3), act].sum() / n.sum(), 0.25682, places=4)


class TestTier2Threshold(unittest.TestCase):
    def test_choose_t2_matches_brute_force(self):
        rng = np.random.default_rng(5)
        p = rng.random(300)
        y = (rng.random(300) < p ** 2).astype(int)
        t2, cost = choose_t2(p, y)
        def c(t):
            blk = p >= t
            return (100 * np.sum((y == 1) & ~blk) + np.sum((y == 0) & blk)) / p.size
        best = min(c(t) for t in list(p) + [np.inf])
        self.assertAlmostEqual(cost, best)
        self.assertAlmostEqual(c(t2), best)


class TestCalibration(unittest.TestCase):
    def test_platt_fixes_distorted_score(self):
        d = generate(SyntheticConfig(n=60_000), seed=0)
        half = d["y"].size // 2
        cal = PlattCalibrator().fit(d["p_raw"][:half], d["y"][:half])
        s = cal.transform(d["p_raw"][half:])
        self.assertLess(ece(s, d["y"][half:]), 0.5 * ece(d["p_raw"][half:], d["y"][half:]))

    def test_platt_is_monotone(self):
        d = generate(SyntheticConfig(n=20_000), seed=1)
        cal = PlattCalibrator().fit(d["p_raw"], d["y"])
        grid = np.linspace(0.001, 0.999, 50)
        self.assertTrue(np.all(np.diff(cal.transform(grid)) > 0))


if __name__ == "__main__":
    unittest.main()
