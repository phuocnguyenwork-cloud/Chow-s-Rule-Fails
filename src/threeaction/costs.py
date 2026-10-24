"""Cost matrix and per-transaction costs.

Action order is fixed everywhere: 0 = PASS, 1 = ESCALATE, 2 = BLOCK.

Cost of one transaction (y = 1 means fraud):
  PASS      -> c_miss if fraud else 0
  BLOCK     -> c_false_block if legitimate else 0
  ESCALATE  -> c_escalate, plus c_miss if tier 2 passes a fraud,
               plus c_false_block if tier 2 blocks a legitimate one.
"""
import numpy as np

PASS, ESCALATE, BLOCK = 0, 1, 2
ACTION_NAMES = ("PASS", "ESCALATE", "BLOCK")

C_MISS = 100.0
C_FALSE_BLOCK = 1.0
C_ESCALATE = 0.2


def per_txn_costs(y, tier2_block, c_miss=C_MISS, c_false_block=C_FALSE_BLOCK,
                  c_escalate=C_ESCALATE):
    """Return an (N, 3) array: column a = cost if action a is applied.

    y            : 0/1 labels (1 = fraud)
    tier2_block  : 0/1 final decision of tier 2 for every transaction
                   (1 = block). Computed for all rows in simulation; it only
                   matters for rows that are actually escalated.
    """
    y = np.asarray(y, dtype=float)
    b = np.asarray(tier2_block, dtype=float)
    cost_pass = c_miss * y
    cost_block = c_false_block * (1.0 - y)
    cost_esc = c_escalate + c_miss * y * (1.0 - b) + c_false_block * (1.0 - y) * b
    return np.stack([cost_pass, cost_esc, cost_block], axis=1)
