"""Platt scaling and a simple calibration error (ECE).

Platt scaling is an increasing map p_raw -> s, learned on VALIDATION labels.
Test labels are never used to learn it; test is only used to check it.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .bins import _logit


class PlattCalibrator:
    def fit(self, p_raw, y):
        z = _logit(np.asarray(p_raw, float)).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y))
        return self

    def transform(self, p_raw):
        z = _logit(np.asarray(p_raw, float)).reshape(-1, 1)
        return self.lr_.predict_proba(z)[:, 1]


def ece(s, y, n_bins=10):
    """Expected calibration error with equal-count bins.

    Noisy when frauds are rare; read together with a reliability table.
    """
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    order = np.argsort(s)
    chunks = np.array_split(order, n_bins)
    err = 0.0
    for c in chunks:
        if c.size == 0:
            continue
        err += c.size * abs(s[c].mean() - y[c].mean())
    return err / s.size


def reliability_table(s, y, n_bins=10):
    """Rows: (mean score, observed fraud rate, n, n_fraud) per equal-count bin."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    order = np.argsort(s)
    rows = []
    for c in np.array_split(order, n_bins):
        if c.size:
            rows.append((s[c].mean(), y[c].mean(), c.size, int(y[c].sum())))
    return rows
