"""The two policies and their evaluation.

Both policies are chosen from the SAME validation cost table
(K bins x 3 actions of summed costs):

  * per-bin policy : argmin over the 3 actions in every bin independently.
  * two-threshold  : pass for bins < i, escalate for i <= bin < j, block for
                     bin >= j  (i <= j). Thresholds sit on bin edges, so the
                     two-threshold class is a strict subset of the per-bin class.
"""
import numpy as np

from .costs import PASS, ESCALATE, BLOCK


def bin_cost_table(bin_ids, costs, K):
    """Sum of costs per (bin, action) and the number of transactions per bin."""
    table = np.zeros((K, 3))
    for a in range(3):
        table[:, a] = np.bincount(bin_ids, weights=costs[:, a], minlength=K)
    n = np.bincount(bin_ids, minlength=K)
    return table, n


def per_bin_actions(table):
    """Cheapest action in each bin (ties go to the lowest action index)."""
    return table.argmin(axis=1)


def best_two_threshold(table):
    """Exhaustive search over (i, j). Returns (actions, (i, j), total_cost)."""
    K = table.shape[0]
    cp = np.concatenate([[0.0], np.cumsum(table[:, PASS])])
    ce = np.concatenate([[0.0], np.cumsum(table[:, ESCALATE])])
    cb = np.concatenate([[0.0], np.cumsum(table[:, BLOCK])])
    best_cost, best_ij = np.inf, None
    for i in range(K + 1):
        for j in range(i, K + 1):
            total = cp[i] + (ce[j] - ce[i]) + (cb[K] - cb[j])
            if total < best_cost - 1e-12:
                best_cost, best_ij = total, (i, j)
    i, j = best_ij
    actions = np.array([PASS] * i + [ESCALATE] * (j - i) + [BLOCK] * (K - j))
    return actions, best_ij, best_cost


def thresholds_from_ij(edges, i, j):
    """Translate (i, j) into score thresholds (t_L, t_H)."""
    K = len(edges) + 1
    t_low = -np.inf if i == 0 else float(edges[i - 1])
    t_high = np.inf if j == K else float(edges[j - 1])
    return t_low, t_high


def is_two_threshold_shape(actions):
    """True if the action sequence is PASS* ESCALATE* BLOCK* (non-decreasing)."""
    return bool(np.all(np.diff(np.asarray(actions)) >= 0))


def per_txn_policy_cost(actions, bin_ids, costs):
    """Cost of each transaction under a bin->action map."""
    chosen = np.asarray(actions)[bin_ids]
    return costs[np.arange(costs.shape[0]), chosen]


def paired_bootstrap_regret(cost_two, cost_bin, n_boot=1000, seed=0):
    """Paired bootstrap over transactions; both policies share each resample.

    Returns dict with absolute and relative regret (point estimate + 95% CI).
    Regret = mean(cost_two) - mean(cost_bin); relative = regret / mean(cost_bin).
    """
    cost_two = np.asarray(cost_two, float)
    cost_bin = np.asarray(cost_bin, float)
    n = cost_two.size
    rng = np.random.default_rng(seed)
    abs_r = np.empty(n_boot)
    rel_r = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ct, cb = cost_two[idx].mean(), cost_bin[idx].mean()
        abs_r[b] = ct - cb
        rel_r[b] = (ct - cb) / cb if cb > 0 else np.nan
    point_abs = cost_two.mean() - cost_bin.mean()
    point_rel = point_abs / cost_bin.mean() if cost_bin.mean() > 0 else np.nan
    return {
        "abs": point_abs,
        "abs_ci": tuple(np.nanpercentile(abs_r, [2.5, 97.5])),
        "rel": point_rel,
        "rel_ci": tuple(np.nanpercentile(rel_r, [2.5, 97.5])),
    }
