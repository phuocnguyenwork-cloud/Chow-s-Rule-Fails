"""Choosing the tier-2 decision threshold t2 (fixed once, shared by all policies).

Block if p2 >= t2. t2 minimises  c_miss * (#frauds passed) + c_fb * (#legit blocked)
on TRAIN, using out-of-fold p2 so the model has not seen the rows it is scored on.
"""
import numpy as np

from .costs import C_MISS, C_FALSE_BLOCK


def choose_t2(p2, y, c_miss=C_MISS, c_false_block=C_FALSE_BLOCK):
    """Exact minimiser over all thresholds (ties in p2 handled)."""
    p2 = np.asarray(p2, float)
    y = np.asarray(y, int)
    order = np.argsort(p2, kind="stable")
    ps, ys = p2[order], y[order]
    n = ps.size
    cum_pos = np.concatenate([[0], np.cumsum(ys)])            # frauds among first k
    cum_neg = np.concatenate([[0], np.cumsum(1 - ys)])
    neg_total = cum_neg[-1]
    cost = c_miss * cum_pos + c_false_block * (neg_total - cum_neg)  # k = 0..n
    valid = np.ones(n + 1, bool)
    valid[1:n] = ps[1:] != ps[:-1]          # cut only between distinct values
    cost = np.where(valid, cost, np.inf)
    k = int(np.argmin(cost))
    t2 = np.inf if k == n else float(ps[k])
    return t2, float(cost[k] / n)
