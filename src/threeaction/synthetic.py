"""Synthetic two-tier system with a KNOWN structure (for tests and smoke test).

A latent logit z gives the true fraud probability s = sigmoid(z); labels are
Bernoulli(s), so the clean score s is calibrated by construction. A distorted
raw score p_raw = sigmoid(a*z + b) lets us test the calibration step.

Tier 2 is a black box with error rates that may depend on z:
  miss rate m(z)  = P(tier 2 passes | fraud, z)
  false-block f(z) = P(tier 2 blocks | legitimate, z)

Scenarios
  "constant"   : m, f constant       -> escalation cost is linear in s, so a
                                        three-interval (two-threshold) policy
                                        is optimal in the population (Chow-like).
  "perfect"    : tier 2 never errs -> escalation cost is exactly c_escalate, the
                 textbook Chow case (thresholds 0.002 and 0.8 for 100/1/0.2).
  "blind_spot" : tier 2 is nearly blind to frauds in a window of z -> escalation
                 cost is non-convex and the optimal policy can be fragmented.
"""
from dataclasses import dataclass

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SyntheticConfig:
    n: int = 200_000
    mu: float = -3.5           # mean of latent logit
    sd: float = 2.0
    scenario: str = "blind_spot"
    blind_center: float = -3.5  # logit where tier 2 is blind
    blind_width: float = 0.35
    blind_peak: float = 0.95    # extra miss rate at the centre of the window
    base_miss: float = 0.03
    base_false_block: float = 0.03
    raw_slope: float = 1.4      # distortion of the raw tier-1 score
    raw_shift: float = 0.7


def tier2_rates(z, cfg):
    z = np.asarray(z, float)
    if cfg.scenario == "perfect":
        m = np.zeros_like(z)
        f = np.zeros_like(z)
    elif cfg.scenario == "constant":
        m = np.full_like(z, 0.10)
        f = np.full_like(z, cfg.base_false_block)
    elif cfg.scenario == "blind_spot":
        bump = np.exp(-((z - cfg.blind_center) ** 2) / (2 * cfg.blind_width ** 2))
        m = np.clip(cfg.base_miss + cfg.blind_peak * bump, 0, 1)
        f = np.full_like(z, cfg.base_false_block)
    else:
        raise ValueError(f"unknown scenario {cfg.scenario!r}")
    return m, f


def generate(cfg=None, seed=0):
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(seed)
    z = rng.normal(cfg.mu, cfg.sd, cfg.n)
    s_true = _sigmoid(z)
    y = (rng.random(cfg.n) < s_true).astype(int)
    p_raw = _sigmoid(cfg.raw_slope * z + cfg.raw_shift)
    m, f = tier2_rates(z, cfg)
    u = rng.random(cfg.n)
    # fraud: tier 2 blocks with prob 1-m ; legitimate: blocks with prob f
    tier2_block = np.where(y == 1, u >= m, u < f).astype(int)
    return {"z": z, "s_true": s_true, "p_raw": p_raw, "y": y,
            "tier2_block": tier2_block}


def analytic_costs(cfg=None, c_miss=100.0, c_false_block=1.0, c_escalate=0.2):
    """Population costs by numerical integration over z (no sampling noise).

    Returns dict with
      C_star : cost of the pointwise-optimal policy (best action at every score)
      C_two  : cost of the best two-threshold policy for the REAL cost
      C_roc  : cost of the ROC-optimal two-threshold design (class-average costs a+, a-)
      plus a_plus, a_minus and the ROC thresholds.
    Valid because the clean score s = sigmoid(z) is calibrated by construction.
    """
    cfg = cfg or SyntheticConfig()
    z = np.linspace(-16, 12, 400_001)
    dz = z[1] - z[0]
    w = np.exp(-((z - cfg.mu) ** 2) / (2 * cfg.sd ** 2)) / (cfg.sd * np.sqrt(2 * np.pi)) * dz
    s = _sigmoid(z)
    m, f = tier2_rates(z, cfg)
    c_pass = c_miss * s
    c_block = c_false_block * (1 - s)
    c_esc = c_escalate + c_miss * s * m + c_false_block * (1 - s) * f
    C_star = float((w * np.minimum(np.minimum(c_pass, c_esc), c_block)).sum())
    cp = np.concatenate([[0], np.cumsum(w * c_pass)])
    ce = np.concatenate([[0], np.cumsum(w * c_esc)])
    cb = np.concatenate([[0], np.cumsum(w * c_block)])
    C_two = float((np.minimum.accumulate(cp - ce) + (ce - cb) + cb[-1]).min())
    miss = float((w * s * m).sum() / (w * s).sum())
    fblk = float((w * (1 - s) * f).sum() / (w * (1 - s)).sum())
    a_plus = c_escalate + c_miss * miss
    a_minus = c_escalate + c_false_block * fblk
    t_low = a_minus / (c_miss - a_plus + a_minus)
    t_high = (c_false_block - a_minus) / (c_false_block - a_minus + a_plus)
    kL, kH = np.searchsorted(s, t_low), np.searchsorted(s, t_high)
    C_roc = float(cp[kL] + (ce[kH] - ce[kL]) + (cb[-1] - cb[kH]))
    return {"C_star": C_star, "C_two": C_two, "C_roc": C_roc, "a_plus": a_plus,
            "a_minus": a_minus, "t_low_roc": float(t_low), "t_high_roc": float(t_high)}
