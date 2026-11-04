"""Unit tests for T2: Search vs. Brute-force validation.

Validates:
  1. Bin-level two-threshold search (best_two_threshold) matches exhaustive brute-force.
  2. Continuous two-threshold search (exact_two_threshold) matches exhaustive valid-cut search.
  3. Score ties: identical scores are never split into different actions.
  4. Cost ties: multiple global optima yield identical minimum risk (< 1e-12).
  5. Empty escalation zone: when escalation is suboptimal, t_low == t_high (or i == j).
  6. Oracle (pointwise / per-bin) risk matches weighted sum of min risk and lower-bounds B2.

Run with:
  python -m pytest tests/test_search_vs_bruteforce.py -v
"""
import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from threeaction.costs import PASS, ESCALATE, BLOCK, per_txn_costs
from threeaction.policies import (
    best_two_threshold,
    bin_cost_table,
    is_two_threshold_shape,
    per_bin_actions,
)
from threeaction.roc_baseline import exact_two_threshold


# ---------------------------------------------------------------------------
# Independent Reference Brute-force Solvers
# ---------------------------------------------------------------------------

def brute_force_bin_table(table):
    """Exhaustive search over all (i, j) with 0 <= i <= j <= K.

    Cost = sum(PASS for b < i) + sum(ESCALATE for i <= b < j) + sum(BLOCK for b >= j).
    Returns (min_cost, list_of_all_optimal_pairs).
    """
    K = table.shape[0]
    best_cost = np.inf
    best_pairs = []

    for i in range(K + 1):
        for j in range(i, K + 1):
            cost = (
                table[:i, PASS].sum()
                + table[i:j, ESCALATE].sum()
                + table[j:, BLOCK].sum()
            )
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_pairs = [(i, j)]
            elif abs(cost - best_cost) <= 1e-12:
                best_pairs.append((i, j))

    return float(best_cost), best_pairs


def brute_force_continuous(s, costs, weights=None):
    """Exhaustive search over all VALID cut points between distinct scores.

    A valid cut cannot split samples with identical scores.
    Returns (min_risk, list_of_optimal_cut_indices).
    """
    s = np.asarray(s, float)
    costs = np.asarray(costs, float)
    n = len(s)
    if weights is None:
        w = np.ones(n, float)
    else:
        w = np.asarray(weights, float)
    w_sum = w.sum()

    order = np.argsort(s, kind="stable")
    sa = s[order]
    ca = costs[order]
    wa = w[order]

    # Valid cut indices: 0, n, or k where sa[k-1] != sa[k]
    valid_cuts = [0]
    for k in range(1, n):
        if sa[k] != sa[k - 1]:
            valid_cuts.append(k)
    valid_cuts.append(n)

    best_risk = np.inf
    best_pairs = []

    for idx_i, kL in enumerate(valid_cuts):
        for kH in valid_cuts[idx_i:]:
            # kL items get PASS, (kH - kL) items get ESCALATE, (n - kH) items get BLOCK
            cost_pass = (wa[:kL] * ca[:kL, PASS]).sum() if kL > 0 else 0.0
            cost_esc = (wa[kL:kH] * ca[kL:kH, ESCALATE]).sum() if kH > kL else 0.0
            cost_block = (wa[kH:] * ca[kH:, BLOCK]).sum() if kH < n else 0.0

            total = (cost_pass + cost_esc + cost_block) / w_sum
            if total < best_risk - 1e-12:
                best_risk = total
                best_pairs = [(kL, kH)]
            elif abs(total - best_risk) <= 1e-12:
                best_pairs.append((kL, kH))

    return float(best_risk), best_pairs, sa


def pointwise_oracle_risk(costs, weights=None):
    """Oracle selects the pointwise cheapest action for every transaction.

    Oracle risk = sum(w_i * min_a costs[i, a]) / sum(w_i).
    """
    costs = np.asarray(costs, float)
    min_costs = costs.min(axis=1)
    if weights is None:
        return float(min_costs.mean())
    w = np.asarray(weights, float)
    return float((w * min_costs).sum() / w.sum())


# ---------------------------------------------------------------------------
# Test Suite T2
# ---------------------------------------------------------------------------

class TestBinTableSearchVsBruteForce:
    """Test 1: best_two_threshold matches brute-force on discrete bin cost tables."""

    @pytest.mark.parametrize("K", [2, 3, 5, 8, 12, 16])
    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_random_tables_match_brute_force(self, K, seed):
        rng = np.random.default_rng(seed)
        table = rng.uniform(0.1, 10.0, size=(K, 3))

        actions, (i, j), cost = best_two_threshold(table)
        brute_cost, optimal_pairs = brute_force_bin_table(table)

        # Invariant 1: Minimum cost must match within float precision
        assert abs(cost - brute_cost) < 1e-11, (
            f"Cost mismatch: algo={cost}, brute={brute_cost}"
        )
        # Invariant 2: The chosen (i, j) must be in the set of global optima
        assert (i, j) in optimal_pairs
        # Invariant 3: Shape must be monotonic PASS* ESCALATE* BLOCK*
        assert is_two_threshold_shape(actions)
        # Invariant 4: Oracle lower-bound
        oracle_cost = table.min(axis=1).sum()
        assert oracle_cost <= cost + 1e-11

    def test_hand_crafted_toy_table(self):
        """Toy table with known unique optimal pair (i=1, j=3)."""
        table = np.array([
            [1.0, 5.0, 10.0],  # bin 0: PASS is cheapest
            [6.0, 2.0, 8.0],   # bin 1: ESCALATE is cheapest
            [7.0, 3.0, 9.0],   # bin 2: ESCALATE is cheapest
            [10.0, 6.0, 1.0],  # bin 3: BLOCK is cheapest
        ])
        actions, (i, j), cost = best_two_threshold(table)
        brute_cost, optimal_pairs = brute_force_bin_table(table)

        assert (i, j) == (1, 3)
        assert abs(cost - (1.0 + 2.0 + 3.0 + 1.0)) < 1e-12
        assert abs(cost - brute_cost) < 1e-12
        np.testing.assert_array_equal(actions, [PASS, ESCALATE, ESCALATE, BLOCK])


class TestContinuousSearchWithScoreTies:
    """Test 2: exact_two_threshold matches brute-force and respects score ties."""

    def test_score_ties_never_split_identical_scores(self):
        """Cluster of duplicate scores must always receive the identical action."""
        # 10 samples with only 3 distinct scores: 0.2 (x4), 0.5 (x3), 0.8 (x3)
        s = np.array([0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8])
        costs = np.array([
            [0.5, 3.0, 8.0],
            [0.6, 3.1, 7.9],
            [0.4, 2.9, 8.1],
            [0.5, 3.0, 8.0],   # s=0.2 clearly favors PASS
            [5.0, 1.0, 6.0],
            [4.8, 1.1, 6.2],
            [5.2, 0.9, 5.8],   # s=0.5 clearly favors ESCALATE
            [9.0, 4.0, 0.5],
            [9.2, 4.1, 0.4],
            [8.8, 3.9, 0.6],   # s=0.8 clearly favors BLOCK
        ])

        t_low, t_high, algo_risk = exact_two_threshold(s, costs)
        brute_risk, _, _ = brute_force_continuous(s, costs)

        # Risk must match brute force over all legal cut points
        assert abs(algo_risk - brute_risk) < 1e-12

        # Verify decisions: all s=0.2 get PASS, all s=0.5 get ESCALATE, all s=0.8 get BLOCK
        decisions = np.where(s < t_low, PASS, np.where(s < t_high, ESCALATE, BLOCK))
        assert len(set(decisions[:4])) == 1, "Duplicate score s=0.2 was split!"
        assert len(set(decisions[4:7])) == 1, "Duplicate score s=0.5 was split!"
        assert len(set(decisions[7:])) == 1, "Duplicate score s=0.8 was split!"

    @pytest.mark.parametrize("n_unique", [3, 5, 8])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_randomized_score_ties_match_brute_force(self, n_unique, seed):
        rng = np.random.default_rng(seed)
        unique_scores = np.sort(rng.uniform(0.01, 0.99, size=n_unique))
        # Repeat each score 2 to 5 times
        counts = rng.integers(2, 6, size=n_unique)
        s = np.repeat(unique_scores, counts)
        n = len(s)
        costs = rng.uniform(0.1, 20.0, size=(n, 3))

        t_low, t_high, algo_risk = exact_two_threshold(s, costs)
        brute_risk, _, _ = brute_force_continuous(s, costs)

        assert abs(algo_risk - brute_risk) < 1e-11, (
            f"Risk mismatch with score ties: algo={algo_risk}, brute={brute_risk}"
        )


class TestCostTiesAndMultipleOptima:
    """Test 3: Handles ties in cost where multiple pairs achieve identical minimum risk."""

    def test_cost_ties_equal_minimum_risk(self):
        """When multiple (i, j) have identical cost, risk must equal brute-force."""
        # Bins 1 and 2 have identical cost for PASS and ESCALATE
        table = np.array([
            [1.0, 5.0, 10.0],  # bin 0: PASS cheap
            [2.0, 2.0, 8.0],   # bin 1: PASS and ESCALATE tie!
            [2.0, 2.0, 8.0],   # bin 2: PASS and ESCALATE tie!
            [8.0, 5.0, 1.0],   # bin 3: BLOCK cheap
        ])
        actions, (i, j), cost = best_two_threshold(table)
        brute_cost, optimal_pairs = brute_force_bin_table(table)

        # There are multiple valid optimal pairs (e.g. (1, 3), (2, 3), (3, 3))
        assert len(optimal_pairs) > 1, "Fixture must have multiple optima"
        # Minimum cost must strictly match
        assert abs(cost - brute_cost) < 1e-12
        assert (i, j) in optimal_pairs

    def test_continuous_cost_ties(self):
        """Continuous cutpoints with multiple identical optima."""
        s = np.array([0.1, 0.3, 0.5, 0.7])
        # Two identical regions yielding multiple optimal cutpoints
        costs = np.array([
            [1.0, 5.0, 10.0],
            [3.0, 3.0, 10.0],  # tie between PASS and ESCALATE
            [10.0, 2.0, 2.0],  # tie between ESCALATE and BLOCK
            [10.0, 5.0, 1.0],
        ])
        t_low, t_high, algo_risk = exact_two_threshold(s, costs)
        brute_risk, optimal_cuts, _ = brute_force_continuous(s, costs)

        assert len(optimal_cuts) >= 1
        assert abs(algo_risk - brute_risk) < 1e-12


class TestEmptyEscalationZone:
    """Test 4: When escalation is dominated, t_low == t_high (empty middle zone)."""

    def test_bin_table_empty_escalation_when_expensive(self):
        table = np.array([
            [1.0, 50.0, 10.0],
            [2.0, 50.0, 8.0],
            [8.0, 50.0, 2.0],
            [10.0, 50.0, 1.0],
        ])
        actions, (i, j), cost = best_two_threshold(table)
        # Escalation zone must be empty (i == j)
        assert i == j, f"Expected empty escalation zone, got i={i}, j={j}"
        assert ESCALATE not in actions
        brute_cost, optimal_pairs = brute_force_bin_table(table)
        assert abs(cost - brute_cost) < 1e-12

    def test_continuous_empty_escalation_when_expensive(self):
        s = np.array([0.05, 0.15, 0.40, 0.70, 0.85, 0.95])
        costs = np.array([
            [0.5, 100.0, 10.0],
            [1.0, 100.0, 8.0],
            [2.0, 100.0, 6.0],
            [6.0, 100.0, 2.0],
            [8.0, 100.0, 1.0],
            [9.5, 100.0, 0.5],
        ])
        t_low, t_high, algo_risk = exact_two_threshold(s, costs)
        brute_risk, _, _ = brute_force_continuous(s, costs)

        assert t_low == t_high, f"Expected t_low == t_high, got {t_low} and {t_high}"
        assert abs(algo_risk - brute_risk) < 1e-12


class TestOracleAndWeightedGrid:
    """Test 5: Oracle matches weighted sum of min risk and lower-bounds B2."""

    def test_oracle_matches_rowwise_minimums(self):
        rng = np.random.default_rng(42)
        n = 50
        costs = rng.uniform(0.5, 50.0, size=(n, 3))
        weights = rng.uniform(1.0, 5.0, size=n)

        # Theoretical definition
        expected_oracle = (weights * costs.min(axis=1)).sum() / weights.sum()
        computed_oracle = pointwise_oracle_risk(costs, weights=weights)

        assert abs(computed_oracle - expected_oracle) < 1e-12

    def test_b2_risk_bounded_by_oracle_and_single_threshold(self):
        """Fundamental hierarchy: Oracle <= Best 2-Threshold <= Best 1-Threshold."""
        rng = np.random.default_rng(7)
        s = np.sort(rng.uniform(0.01, 0.99, size=60))
        costs = rng.uniform(1.0, 30.0, size=(60, 3))

        t_low, t_high, b2_risk = exact_two_threshold(s, costs)
        oracle = pointwise_oracle_risk(costs)

        # Compute best 1-threshold (pass/block only, i.e. t_low == t_high)
        brute_risk, all_cuts, _ = brute_force_continuous(s, costs)
        single_threshold_risk = min(
            (np.where(s < s_cut, costs[:, PASS], costs[:, BLOCK])).mean()
            for s_cut in np.concatenate([[-np.inf], s, [np.inf]])
        )

        assert oracle <= b2_risk + 1e-12, "Oracle must lower-bound B2 risk!"
        assert b2_risk <= single_threshold_risk + 1e-12, (
            "B2 risk cannot exceed single-threshold policy risk!"
        )
        assert abs(b2_risk - brute_risk) < 1e-12


class TestParameterizedStressValidation:
    """Test 6: Parameterized stress test across sample sizes and seeds."""

    @pytest.mark.parametrize("n", [15, 35, 70])
    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_stress_continuous_vs_brute_force(self, n, seed):
        rng = np.random.default_rng(seed)
        # Random scores with occasional duplicates
        raw_s = rng.uniform(0.0, 1.0, size=n)
        s = np.round(raw_s, 2)  # Rounding creates natural score ties
        costs = rng.uniform(0.1, 15.0, size=(n, 3))

        t_low, t_high, algo_risk = exact_two_threshold(s, costs)
        brute_risk, optimal_cuts, sa = brute_force_continuous(s, costs)

        assert abs(algo_risk - brute_risk) < 1e-11, (
            f"Failed on n={n}, seed={seed}: algo={algo_risk}, brute={brute_risk}"
        )

