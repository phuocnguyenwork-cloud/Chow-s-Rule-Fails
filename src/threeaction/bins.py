"""Score bins: how the tier-1 score axis is cut into K intervals.

An edge vector `edges` (sorted, length K-1) defines bins
  bin 0:  s < edges[0]
  bin i:  edges[i-1] <= s < edges[i]
  bin K-1: s >= edges[K-2]
The same edges (computed on TRAIN only) are reused for validation and test.
"""
import numpy as np

METHODS = ("quantile", "logit", "fraud_quantile")


def _logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_edges(s_train, y_train, K, method="logit"):
    """Return interior edges (<= K-1 of them, duplicates removed)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return np.array([])
    qs = np.linspace(0, 1, K + 1)[1:-1]
    s_train = np.asarray(s_train, float)
    y_train = np.asarray(y_train)
    if method == "quantile":          # equal number of transactions per bin
        e = np.quantile(s_train, qs)
    elif method == "logit":           # equal width on the logit axis
        z = _logit(s_train)
        lo, hi = np.quantile(z, [0.01, 0.99])
        e = _sigmoid(np.linspace(lo, hi, K + 1)[1:-1])
    elif method == "fraud_quantile":  # equal number of frauds per bin
        e = np.quantile(s_train[y_train == 1], qs)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return np.unique(e)


def assign_bins(s, edges):
    edges = np.asarray(edges, float)
    s = np.asarray(s, float)
    if edges.size == 0:
        return np.zeros(s.shape, dtype=int)
    return np.digitize(s, edges)


def merge_min_positives(edges, s, y, m):
    """Merge neighbouring bins (low -> high) until each has >= m frauds.

    Uses the (s, y) given (normally TRAIN). Returns the reduced edge vector.
    """
    edges = list(np.asarray(edges, float))
    y = np.asarray(y, float)
    while edges:
        ids = assign_bins(s, edges)
        K = len(edges) + 1
        pos = np.bincount(ids, weights=y, minlength=K)
        bad = np.where(pos < m)[0]
        if bad.size == 0:
            break
        b = bad[0]
        edges.pop(b if b < K - 1 else b - 1)   # drop the edge shared with a neighbour
    return np.array(edges)
