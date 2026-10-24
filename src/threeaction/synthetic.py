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
    if cfg.scenario == "constant":
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
