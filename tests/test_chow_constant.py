"""T1: calibrated finite Bayes oracles; independent A1/B bound checks."""
import os
import sys
import subprocess
import unittest
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from threeaction.costs import PASS, ESCALATE, BLOCK, per_txn_costs
from threeaction.policies import thresholds_from_ij, per_bin_actions
from threeaction.roc_baseline import (closed_form_thresholds, actions_from_thresholds,
                                     exact_two_threshold, cost_from_thresholds, roc_optimal_thresholds)

ATOL, RTOL = 1e-12, 1e-12


def risks(s, ap, am, fn=100., fp=1.):
    s = np.asarray(s)
    return np.column_stack((fn*s, ap*s + am*(1-s), fp*(1-s)))


# Integer numerators encode empirical calibration exactly: k positives / denom.
CHOW_CASES = [
    pytest.param((8., 2., .5, 16, [0, 1, 2, 11, 12, 13, 16]), id="dyadic-ties"),
    pytest.param((100., 1., .2, 1000, [0, 1, 2, 3, 790, 800, 810, 1000]),
                 id="project-100-1-c0.2"),
    pytest.param((100., 1., .99, 20000, [0, 196, 198, 199, 200, 201, 20000]),
                 id="near-critical-c0.99"),
    pytest.param((100., 1., 1., 101, [0, 1, 2, 101]), id="empty-esc-c1"),
]


@pytest.fixture(params=CHOW_CASES)
def chow_case(request):
    """Uniform finite score law; labels are counts, never Bernoulli samples."""
    A, D, c, denom, numerators = request.param
    q = np.asarray(numerators, dtype=float) / denom
    scores = np.repeat(q, denom)
    labels = np.concatenate([np.r_[np.ones(k), np.zeros(denom-k)]
                             for k in numerators]).astype(int)
    conditional = np.column_stack((A*q, np.full(q.size, c), D*(1-q)))
    return A, D, c, q, scores, labels, conditional


def independent_grid_oracle(q, conditional):
    """Enumerate cuts and sum risks directly; no production policy/cost helper."""
    cuts = np.r_[-np.inf, q, np.inf]
    candidates = []
    for i, lo in enumerate(cuts):
        for hi in cuts[i:]:
            value = np.mean(np.where(q < lo, conditional[:, 0],
                            np.where(q < hi, conditional[:, 1], conditional[:, 2])))
            candidates.append(value)
    return min(candidates)


class TestChowBaselines:
    def test_t1a_b1_pietraszek_constant_abstention(self, chow_case):
        A, D, c, q, scores, labels, conditional = chow_case
        lo, hi, _ = roc_optimal_thresholds(scores, labels, c, c,
                                          c_miss=A, c_false_block=D)
        assert lo <= hi
        assert lo in np.r_[-np.inf, q, np.inf]
        assert hi in np.r_[-np.inf, q, np.inf]
        selected = np.where(q < lo, conditional[:, 0],
                           np.where(q < hi, conditional[:, 1], conditional[:, 2]))
        np.testing.assert_allclose(selected, conditional.min(axis=1), atol=ATOL, rtol=RTOL)
        if A == 8. and D == 2.:
            # All four equivalent pairs are accepted; no fixed ROC tie priority.
            assert lo in {1/16, 1/8}
            assert hi in {3/4, 13/16}
        if c >= A*D/(A+D):
            assert lo == hi  # empty escalation is part of the two-threshold class
        critical = A*D/(A+D)
        if c < critical:
            np.testing.assert_allclose(closed_form_thresholds(c, c, A, D),
                                       (c/A, 1-c/D), atol=ATOL, rtol=RTOL)

    @pytest.mark.xfail(strict=True, reason="closed_form_thresholds does not collapse an empty escalation region")
    def test_closed_form_empty_escalation_region(self):
        expected = 1/101
        np.testing.assert_allclose(closed_form_thresholds(1., 1.),
                                   (expected, expected), atol=ATOL, rtol=RTOL)

    def test_t1b_b2_grid_same_risk(self, chow_case):
        _, _, _, q, _, _, conditional = chow_case
        expected = independent_grid_oracle(q, conditional)
        lo, hi, value = exact_two_threshold(q, conditional)
        chosen = np.where(q < lo, conditional[:, 0],
                          np.where(q < hi, conditional[:, 1], conditional[:, 2])).mean()
        np.testing.assert_allclose([value, chosen], expected, atol=ATOL, rtol=RTOL)

    def test_t1c_independent_oracle_regret_zero(self, chow_case):
        A, D, c, q, scores, labels, conditional = chow_case
        risk_s = np.minimum(np.minimum(A*q, c), D*(1-q)).mean()
        risk_two = independent_grid_oracle(q, conditional)
        np.testing.assert_allclose(risk_two-risk_s, 0., atol=ATOL, rtol=RTOL)
        lo, hi, _ = roc_optimal_thresholds(scores, labels, c, c,
                                          c_miss=A, c_false_block=D)
        b1_risk = np.where(q < lo, A*q, np.where(q < hi, c, D*(1-q))).mean()
        _, _, b2_risk = exact_two_threshold(q, conditional)
        np.testing.assert_allclose([b1_risk-risk_s, b2_risk-risk_s], 0., atol=ATOL, rtol=RTOL)


class TestChowConstant(unittest.TestCase):
    def test_asymmetric_cost_matrix(self):
        actual = per_txn_costs([0, 0, 1, 1], [0, 1, 0, 1],
                               c_miss=7., c_false_block=2., c_escalate=.3)
        expected = [[0, .3, 2], [0, 2.3, 2], [7, 7.3, 0], [7, .3, 0]]
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)

    def test_textbook_asymmetric_chow(self):
        np.testing.assert_allclose(closed_form_thresholds(.2, .2), [.002, .8],
                                   atol=ATOL, rtol=RTOL)

    def test_class_dependent_constant_costs(self):
        # e(s)=12*s+.4*(1-s); not the textbook constant abstention risk.
        lo, hi = closed_form_thresholds(12., .4)
        np.testing.assert_allclose([lo, hi], [.4/88.4, .6/12.6], atol=ATOL, rtol=RTOL)
        s = np.unique(np.r_[np.linspace(0, 1, 10001), lo, hi])
        r = risks(s, 12., .4)
        a = actions_from_thresholds(s, lo, hi)
        np.testing.assert_allclose(r[np.arange(len(s)), a], r.min(axis=1), atol=ATOL, rtol=RTOL)

    def test_threshold_ties_and_neighbors(self):
        lo, hi = closed_form_thresholds(.2, .2)
        s = [np.nextafter(lo, -np.inf), lo, np.nextafter(lo, np.inf),
             np.nextafter(hi, -np.inf), hi, np.nextafter(hi, np.inf)]
        np.testing.assert_array_equal(actions_from_thresholds(s, lo, hi),
                                     [PASS, ESCALATE, ESCALATE, ESCALATE, BLOCK, BLOCK])

    def test_triple_tie_collapses_to_binary(self):
        lo, hi = closed_form_thresholds(.5, .5, c_miss=1., c_false_block=1.)
        self.assertEqual((lo, hi), (.5, .5))
        np.testing.assert_array_equal(actions_from_thresholds([.49, .5, .51], lo, hi), [PASS, BLOCK, BLOCK])

    @unittest.expectedFailure  # known closed_form_thresholds empty-region defect
    def test_expensive_escalation_has_empty_region(self):
        lo, hi = closed_form_thresholds(50., 50.)
        self.assertEqual(lo, hi)
        self.assertAlmostEqual(lo, 1/101)
        self.assertNotIn(ESCALATE, actions_from_thresholds(np.linspace(0, 1, 1001), lo, hi))

    def test_zero_cost_escalation(self):
        self.assertEqual(closed_form_thresholds(0., 0.), (0., 1.))
        s = np.array([0., .5, 1.])
        r = risks(s, 0., 0.)
        a = actions_from_thresholds(s, 0., 1.)
        np.testing.assert_allclose(r[np.arange(3), a], 0., atol=ATOL)

    @unittest.expectedFailure  # known thresholds_from_ij boundary defect
    def test_all_empty_bin_regions_roundtrip(self):
        edges = np.array([.1, .5])
        s = np.array([0., .1, .3, .5, 1.])
        ids = np.searchsorted(edges, s, side='right')
        for i in range(4):
            for j in range(i, 4):
                expected = np.array([PASS]*i + [ESCALATE]*(j-i) + [BLOCK]*(3-j))
                lo, hi = thresholds_from_ij(edges, i, j)
                np.testing.assert_array_equal(actions_from_thresholds(s, lo, hi), expected[ids])
        self.assertEqual(thresholds_from_ij([], 0, 0), (-np.inf, -np.inf))
        self.assertEqual(thresholds_from_ij([], 1, 1), (np.inf, np.inf))

    @unittest.expectedFailure  # known exact_two_threshold all-PASS boundary defect
    def test_exact_solver_all_pass(self):
        c = np.full((3, 3), 2.)
        c[:, PASS] = 0.
        lo, hi, value = exact_two_threshold([.1, .5, .9], c)
        self.assertEqual(value, 0.)
        np.testing.assert_array_equal(actions_from_thresholds([.1, .5, .9], lo, hi), PASS)

    def test_exact_solver_all_escalate_and_all_block(self):
        for preferred in [ESCALATE, BLOCK]:
            c = np.full((3, 3), 2.)
            c[:, preferred] = 0.
            lo, hi, value = exact_two_threshold([.1, .5, .9], c)
            self.assertEqual(value, 0.)
            np.testing.assert_array_equal(actions_from_thresholds([.1, .5, .9], lo, hi), preferred)

    def test_duplicate_scores_permutation_and_input_integrity(self):
        s = np.array([.2, .2, .8, .8])
        c = np.array([[0., 1, 2], [2., 0, 1], [3., 2, 0], [1., 0, 3]])
        s0, c0 = s.copy(), c.copy()
        lo, hi, value = exact_two_threshold(s, c)
        a = actions_from_thresholds(s, lo, hi)
        self.assertEqual(a[0], a[1]); self.assertEqual(a[2], a[3])
        self.assertAlmostEqual(value, cost_from_thresholds(s, lo, hi, c).mean())
        p = [3, 0, 2, 1]
        self.assertAlmostEqual(value, exact_two_threshold(s[p], c[p])[2])
        np.testing.assert_array_equal(s, s0); np.testing.assert_array_equal(c, c0)
        np.testing.assert_array_equal(per_bin_actions(np.zeros((2, 3))), [PASS, PASS])

    @unittest.expectedFailure  # input validation is not yet part of the API contract
    def test_invalid_closed_form_parameters(self):
        for ap, am, fn, fp in [(-1, .2, 100, 1), (.2, .2, 0, 1), (np.nan, .2, 100, 1)]:
            with self.assertRaises(ValueError):
                closed_form_thresholds(ap, am, fn, fp)


class TestBoundA1AndB(unittest.TestCase):
    @unittest.expectedFailure  # depends on the known empty-region defect
    def test_collapsed_surrogate_still_satisfies_bound(self):
        s = np.linspace(0., 1., 101)
        w = np.full(s.size, 1/s.size)
        surrogate = risks(s, 50., 50.)
        lo, hi = closed_form_thresholds(50., 50.)
        self.assertEqual(lo, hi)
        a0 = actions_from_thresholds(s, lo, hi)
        true = risks(s, .2, .2)
        star = true.min(axis=1)
        delta = true[:, ESCALATE]-surrogate[:, ESCALATE]
        r0 = np.dot(w, true[np.arange(s.size), a0]-star)
        _, _, best_two = exact_two_threshold(s, true)
        regret = best_two-np.dot(w, star)
        self.assertGreaterEqual(regret, -ATOL)
        self.assertLessEqual(regret, r0+ATOL)
        self.assertLessEqual(r0, np.dot(w, abs(delta))+ATOL)

    def test_escalation_only_perturbation_bound(self):
        # Exact finite probability space; no Monte Carlo label noise or bins.
        s = np.linspace(0, 1, 1001)
        w = np.arange(1, len(s)+1, dtype=float); w /= w.sum()
        surrogate = risks(s, .2, .2)
        lo, hi = closed_form_thresholds(.2, .2)
        a0 = actions_from_thresholds(s, lo, hi)
        for amp in [0., .01, .5, 2.]:
            delta = amp*np.sin(19*s)
            true = surrogate.copy()
            true[:, ESCALATE] = np.maximum(0, .2 + delta)
            delta = true[:, ESCALATE] - surrogate[:, ESCALATE]
            star = true.min(axis=1)
            r0 = np.dot(w, true[np.arange(len(s)), a0] - star)
            # optimum within two-threshold class <= feasible surrogate policy
            _, _, value = exact_two_threshold(s, true*w[:, None]*len(s))
            rtwo = value - np.dot(w, star)
            bound = np.dot(w, abs(delta))
            # A1 is independently enumerated, PASS/BLOCK only; ESC may be empty.
            cuts = np.r_[-np.inf, s, np.inf]
            risk_a1 = min(np.dot(w, np.where(s < t, true[:, PASS], true[:, BLOCK]))
                          for t in cuts)
            preliminary = risk_a1 - np.dot(w, star)
            self.assertGreaterEqual(rtwo, -ATOL)
            self.assertLessEqual(rtwo, r0+ATOL)
            self.assertLessEqual(r0, bound+ATOL)
            self.assertLessEqual(rtwo, min(bound, preliminary)+ATOL)
            if amp == 0:
                self.assertAlmostEqual(rtwo, 0., places=12)


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    command = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider',
               '-q', '-rx', os.path.abspath(__file__)]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    log = ('$ ' + ' '.join(command) + '\n' + result.stdout + result.stderr
           + f'\nExit code: {result.returncode}\n')
    log_path = os.path.join(repo_root, 'outputs', 'test_chow_constant.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as output:
        output.write(log)
    print(log, end='')
    print(f'Log saved to: {log_path}')
    raise SystemExit(result.returncode)
