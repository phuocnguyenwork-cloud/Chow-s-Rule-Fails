"""ROC-optimal abstaining classifier (Pietraszek, ICML 2005) = the 'traditional' baseline.

Assumption of the method: escalating costs a CONSTANT per class:
    a_plus  = cost of escalating a fraud       (c_escalate + c_miss * tier-2 miss rate)
    a_minus = cost of escalating a legitimate  (c_escalate + c_fb   * tier-2 false-block rate)
Under that assumption the cost of a threshold pair (t_L < t_H) is linear in two ROC
points and the two thresholds decouple:

    minimise over point B (t_L):  -p+ (C_FN - a+) TPR + p- a-        FPR
    minimise over point A (t_H):  -p+  a+         TPR + p- (C_FP-a-) FPR

(the minimiser of a linear function over the ROC points is a vertex of the ROC
convex hull, so no explicit hull is needed). If tier 2's error depends on the score
this is only a MODEL of the real cost, which is exactly what the project measures.

Policy: s < t_L -> PASS ; t_L <= s < t_H -> ESCALATE ; s >= t_H -> BLOCK.
"""
import numpy as np

from .costs import C_MISS, C_FALSE_BLOCK, C_ESCALATE, PASS, ESCALATE, BLOCK
from .bins import _logit


def class_escalation_costs(y, tier2_block, c_miss=C_MISS, c_false_block=C_FALSE_BLOCK,
                           c_escalate=C_ESCALATE):
    """(a_plus, a_minus) estimated from data. Use TRAIN (out-of-fold) data only."""
    y = np.asarray(y)
    b = np.asarray(tier2_block, float)
    miss = 1.0 - b[y == 1].mean()
    false_block = b[y == 0].mean()
    return c_escalate + c_miss * miss, c_escalate + c_false_block * false_block


def _roc(s, y):
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    n = ss.size
    tpr = np.concatenate([[0], np.cumsum(yy)]) / yy.sum()
    fpr = np.concatenate([[0], np.cumsum(1 - yy)]) / (1 - yy).sum()
    valid = np.ones(n + 1, bool)                 # cut only between distinct scores
    valid[1:n] = ss[:-1] != ss[1:]
    return ss, tpr, fpr, valid


def _threshold(ss, i):
    """Score threshold that blocks/escalates exactly the top-i ranked items."""
    if i <= 0:
        return np.inf
    if i >= ss.size:
        return -np.inf
    return float(ss[i - 1])


def roc_optimal_thresholds(s, y, a_plus, a_minus, c_miss=C_MISS, c_false_block=C_FALSE_BLOCK,
                           prior_pos=None):
    """Return (t_low, t_high, info). Falls back to one threshold (no escalation)
    when escalation is never worth it under the constant-cost model."""
    ss, tpr, fpr, valid = _roc(s, y)
    pp = float(np.mean(y)) if prior_pos is None else float(prior_pos)
    pm = 1.0 - pp
    inf = np.inf

    def argmin_valid(cost):
        return int(np.argmin(np.where(valid, cost, inf)))

    feasible = (a_plus < c_miss) and (a_minus < c_false_block)
    if feasible:
        iB = argmin_valid(-pp * (c_miss - a_plus) * tpr + pm * a_minus * fpr)      # t_low
        iA = argmin_valid(-pp * a_plus * tpr + pm * (c_false_block - a_minus) * fpr)  # t_high
        feasible = iB >= iA
    if not feasible:
        i = argmin_valid(-pp * c_miss * tpr + pm * c_false_block * fpr)
        t = _threshold(ss, i)
        return t, t, {"escalates": False, "a_plus": a_plus, "a_minus": a_minus}
    return _threshold(ss, iB), _threshold(ss, iA), {
        "escalates": True, "a_plus": a_plus, "a_minus": a_minus}


def closed_form_thresholds(a_plus, a_minus, c_miss=C_MISS, c_false_block=C_FALSE_BLOCK):
    """Population thresholds for a CALIBRATED score under the same constant-cost model."""
    t_low = a_minus / (c_miss - a_plus + a_minus)
    t_high = (c_false_block - a_minus) / (c_false_block - a_minus + a_plus)
    return t_low, t_high


def actions_from_thresholds(s, t_low, t_high):
    s = np.asarray(s, float)
    return np.where(s < t_low, PASS, np.where(s < t_high, ESCALATE, BLOCK))


def cost_from_thresholds(s, t_low, t_high, costs):
    """Per-transaction cost of a threshold policy (no bins involved)."""
    a = actions_from_thresholds(s, t_low, t_high)
    return costs[np.arange(costs.shape[0]), a]


def snap_to_edges(t, edges):
    """Index in 0..K of the bin edge closest (logit scale) to t.
    0 means -inf and K means +inf, as in policies.thresholds_from_ij."""
    K = len(edges) + 1
    if t == -np.inf:
        return 0
    if t == np.inf:
        return K
    if len(edges) == 0:
        return 0
    d = np.abs(_logit(np.asarray(edges, float)) - _logit(np.array(t)))
    return int(np.argmin(d)) + 1


def snapped_actions(t_low, t_high, edges):
    """Bin -> action map of the ROC policy after snapping thresholds to bin edges."""
    K = len(edges) + 1
    i = snap_to_edges(t_low, edges)
    j = max(i, snap_to_edges(t_high, edges))
    return np.array([PASS] * i + [ESCALATE] * (j - i) + [BLOCK] * (K - j)), (i, j)


def exact_two_threshold(s, costs):
    """Best (t_low, t_high) over ALL cut points for the REAL per-transaction costs, O(n).

    Total cost = f(kL) + g(kH) + const with f = cumPass - cumEsc, g = cumEsc - cumBlock,
    so the pair decouples (same trick as the ROC method, but with real costs).
    Returns (t_low, t_high, mean_cost).
    """
    s = np.asarray(s, float)
    o = np.argsort(s, kind="stable")
    sa, ca = s[o], np.asarray(costs)[o]
    n = sa.size
    cum = np.vstack([np.zeros(3), np.cumsum(ca, axis=0)])
    valid = np.ones(n + 1, bool)
    valid[1:n] = sa[1:] != sa[:-1]
    f = np.where(valid, cum[:, PASS] - cum[:, ESCALATE], np.inf)
    g = np.where(valid, cum[:, ESCALATE] - cum[:, BLOCK], np.inf)
    run_min = np.minimum.accumulate(f)
    run_arg = np.zeros(n + 1, int)
    best = 0
    for k in range(n + 1):
        if f[k] < f[best]:
            best = k
        run_arg[k] = best
    tot = run_min + g
    kH = int(np.argmin(tot))
    kL = int(run_arg[kH])
    cost = (tot[kH] + cum[n, BLOCK]) / n
    t_low = -np.inf if kL == 0 else float(sa[kL])
    t_high = np.inf if kH == n else float(sa[kH])
    return t_low, t_high, float(cost)
