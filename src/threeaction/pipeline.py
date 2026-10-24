"""One full experiment on one train/val/test split.

Inputs are already-computed arrays (tier-1 score s, labels y, tier-2 final
decision) for each part, so the same code serves synthetic data and the real
Credit Card pipeline.
"""
import numpy as np

from .bins import make_edges, merge_min_positives, assign_bins
from .costs import per_txn_costs
from .policies import (bin_cost_table, per_bin_actions, best_two_threshold,
                       thresholds_from_ij, is_two_threshold_shape,
                       per_txn_policy_cost, paired_bootstrap_regret)


def run_policies(train, val, test, K=10, method="logit", min_pos=5,
                 n_boot=1000, seed=0):
    """Each of train/val/test is a dict with keys s, y, tier2_block.

    Edges come from TRAIN only; the cost table and both policies are chosen on
    VAL only; the reported cost/regret are measured on TEST only.
    """
    edges = make_edges(train["s"], train["y"], K, method)
    edges = merge_min_positives(edges, train["s"], train["y"], min_pos)
    K_eff = len(edges) + 1

    c_val = per_txn_costs(val["y"], val["tier2_block"])
    c_test = per_txn_costs(test["y"], test["tier2_block"])
    b_val = assign_bins(val["s"], edges)
    b_test = assign_bins(test["s"], edges)

    table, n_val = bin_cost_table(b_val, c_val, K_eff)
    act_bin = per_bin_actions(table)
    act_two, (i, j), _ = best_two_threshold(table)
    t_low, t_high = thresholds_from_ij(edges, i, j)

    val_two = per_txn_policy_cost(act_two, b_val, c_val)
    val_bin = per_txn_policy_cost(act_bin, b_val, c_val)
    test_two = per_txn_policy_cost(act_two, b_test, c_test)
    test_bin = per_txn_policy_cost(act_bin, b_test, c_test)

    regret = paired_bootstrap_regret(test_two, test_bin, n_boot=n_boot, seed=seed)
    return {
        "K_effective": K_eff,
        "edges": edges,
        "table_mean": table / np.maximum(n_val, 1)[:, None],
        "n_val_per_bin": n_val,
        "actions_per_bin": act_bin,
        "actions_two_threshold": act_two,
        "t_low": t_low, "t_high": t_high,
        "per_bin_is_two_threshold_shape": is_two_threshold_shape(act_bin),
        "val_cost_two": float(val_two.mean()),
        "val_cost_bin": float(val_bin.mean()),
        "test_cost_two": float(test_two.mean()),
        "test_cost_bin": float(test_bin.mean()),
        "test_regret": regret,
    }
