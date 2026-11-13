import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT,"src"))

from threeaction.costs import PASS, ESCALATE, BLOCK, per_txn_costs
from threeaction.policies import (best_two_threshold, bin_cost_table, is_two_threshold_shape, per_bin_actions)


# tham so
TOTAL = 1_000_000
W = [0.80, 0.10, 0.05, 0.03, 0.02] # tỉ lệ số gd mỗi bin
ETA = [0.001, 0.01, 0.03, 0.10, 0.50] # Tỷ lệ gian lận thật trong từng bin
M_BLIND = [0.05, 0.05, 0.90, 0.05, 0.05]   # xs tầng 2 bỏ sót gian lận
M_FLAT = [0.05] * 5                        # đối chứng: tầng 2 bỏ sót đều
F = 0.01 # xs tầng 2 chặn nhầm gd hợp lệ

# tính tay
TABLE_MEAN_HAND = np.array([
    [0.1,  0.21499, 0.999],   # bin 0: rE = 0.2 + 0.05*100*0.001 + 1*0.01*(1-0.001)
    [1.0,  0.2599,  0.99],    # bin 1: rE = 0.2 + 0.05  + 0.0099
    [3.0,  2.9097,  0.97],    # bin 2: rE = 0.2 + 2.7   + 0.0097  (blind spot)
    [10.0, 0.709,   0.9],     # bin 3: rE = 0.2 + 0.5   + 0.009
    [50.0, 2.705,   0.5],     # bin 4: rE = 0.2 + 2.5   + 0.005
])
ORACLE_ACTIONS_HAND = [PASS, ESCALATE, BLOCK, ESCALATE, BLOCK]
TWO_THRESHOLD_IJ_HAND = (1, 2)    # PASS bin 0, ESC bin 1, BLOCK bin 2..4

RS_HAND = 0.18576    # 0.8·0.1 + 0.1·0.2599 + 0.05·0.97 + 0.03·0.709 + 0.02·0.5
R2T_HAND = 0.19149   # 0.8·0.1 + 0.1·0.2599 + 0.05·0.97 + 0.03·0.9   + 0.02·0.5
R0_HAND = 0.26450    # 0.8·0.1 + 0.1·0.99   + 0.05·0.97 + 0.03·0.9   + 0.02·0.5
DELTA_HAND = R2T_HAND - RS_HAND   # 0.00573
B_HAND = R0_HAND - RS_HAND        # 0.07874

EPS = 1e-9

def transactions(m_per_bin, total=TOTAL):
    # m_per_bin: xs gian lận thật bị bỏ sót 
    bin_ids, y,b  = [],[],[]
    for k, (w, eta, m) in enumerate(zip(W, ETA, m_per_bin)):
        n = round(total * w) #số giao dịhc trong 1 bin 
        n_fraud = round(eta*n) # giao dich gian lan that trong bin
        n_leg = n - n_fraud
        n_miss = round(n_fraud*m) # giao dch gian lan bị bỏ sót tầng 2
        n_fp = round(F*n_leg) # giao dich hợp lệ bị chặn ở tầng 2 

        # Kiểm tra số đếm thật sự nguyên, nếu không đáp án tính tay sẽ lệch
        assert np.isclose(eta * n, n_fraud) and np.isclose(m * n_fraud, n_miss)
        assert np.isclose(F * n_leg, n_fp)

        y += [1] * n_fraud + [0]*n_leg #label
        b += [0] * n_miss + [1] * (n_fraud - n_miss) + [1] * n_fp + [0] * (n_leg - n_fp) # b=1: tầng 2 chặn, b=0: pass
        bin_ids += [k] * n
    return np.array(bin_ids), np.array(y), np.array(b)

def evaluate_fixture_policies(m_per_bin):
    bin_ids, y, b = transactions(m_per_bin)
    cost = per_txn_costs(y,b)  #  chi phí từng giao dịch (n,3)
    table, n = bin_cost_table(bin_ids, cost, len(W))
    N = n.sum()

    oracle_action = per_bin_actions(table) # rẻ nhất trong từng bin 
    r_s = table[np.arange(len(W)), oracle_action].sum()/N

    two_actions, ij, two_cost = best_two_threshold(table)
    r_2t = two_cost/N

    return {"table_mean": table / n[:, None], "oracle_actions": oracle_action,
            "two_actions": two_actions, "ij": ij, "r_s": r_s, "r_2t": r_2t}

@pytest.fixture(scope="module")
def blind():
    return evaluate_fixture_policies(M_BLIND)

# tầng 2 có blind spot -> ko thể áp dụng 2 ngưỡng
class TestFragmentRegret:
    def test_cost_table(self, blind):
        np.testing.assert_allclose(blind["table_mean"], TABLE_MEAN_HAND, rtol=0, atol=EPS)

    def test_oracle_actions(self, blind):
        np.testing.assert_array_equal(blind["oracle_actions"], ORACLE_ACTIONS_HAND)
        assert not is_two_threshold_shape(blind["oracle_actions"])

    def test_risks(self, blind):
        assert abs(blind["r_s"] - RS_HAND) < EPS, f"oracle: {blind['r_s']} != {RS_HAND}"
        assert abs(blind["r_2t"] - R2T_HAND) < EPS, f"2 ngưỡng: {blind['r_2t']} != {R2T_HAND}"
        assert blind["ij"] == TWO_THRESHOLD_IJ_HAND
        assert is_two_threshold_shape(blind["two_actions"])

    def test_two_thresholds_strictly_lose(self, blind):
        """r_2t - r_s > 0 do phải 2 ngưỡng"""
        delta = blind["r_2t"] - blind["r_s"]
        assert delta > 0
        assert abs(delta - DELTA_HAND) < EPS

    def test_regret_bound(self, blind):
        # r_2t - r_s <= r0 - r_s
        delta = blind["r_2t"] - blind["r_s"]
        assert delta <= B_HAND + EPS

# regret xuất hiện khi bin có blind spot
class TestWithoutBlindSpot:
    def test_no_regret_without_blind_spot(self):
        risk = evaluate_fixture_policies(M_FLAT)
        np.testing.assert_array_equal(risk["oracle_actions"],[PASS, ESCALATE, ESCALATE, ESCALATE, BLOCK])
        assert is_two_threshold_shape(risk["oracle_actions"])
        assert abs(risk["r_2t"] - risk["r_s"]) < EPS

# chi phí khi 1 ngưỡng 
def brute_force_one_threshold(table):
    m  = table.shape[0]
    return min(table[:t, PASS].sum() + table[t:, BLOCK].sum() for t in range(m+1))

# test trên nhìu table
@pytest.mark.parametrize("seed", range(20))
def test_regret_bound_on_random_tables(seed):
    rng = np.random.default_rng(seed)

    K = rng.integers(2, 12)
    table = rng.uniform(0, 10, size=(K, 3))

    r_s = table.min(axis=1).sum()
    _, _, r_2t = best_two_threshold(table)
    r_0 = brute_force_one_threshold(table)

    delta = r_2t - r_s
    tol = 1e-12

    assert delta >= -tol
    assert r_2t <= r_0 + tol
    assert delta <= r_0 - r_s + tol