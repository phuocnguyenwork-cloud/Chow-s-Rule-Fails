"""Group-safe 60/20/20 splits; exact float64 features define groups."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def feature_groups(X):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Features must be a finite two-dimensional float64 table')
    return pd.DataFrame(values).groupby(list(range(values.shape[1])), sort=False,
                                      dropna=False).ngroup().to_numpy()


def validate_split(y, groups, tr, va, te):
    arrays = [np.asarray(a) for a in (tr, va, te)]
    if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
        raise ValueError('Split indices must be one-dimensional integer arrays')
    if not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(len(y))):
        raise ValueError('Splits must cover every row exactly once; stale archive?')
    sets = [set(groups[a]) for a in arrays]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Feature group crosses train/validation/test')
    if any(len(np.unique(np.asarray(y)[a])) != 2 for a in arrays):
        raise ValueError('Each partition must contain both classes')


def _group_labels(y, groups):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary')
    unique, inv = np.unique(groups, return_inverse=True)
    # Group identity uses features only. Any-positive is used only for stratification.
    labels = np.zeros(len(unique), dtype=int)
    np.maximum.at(labels, inv, y)
    return unique, inv, labels


def group_oof_splits(y, groups, seed):
    unique, inv, labels = _group_labels(y, groups)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return [(np.flatnonzero(np.isin(inv, tr)), np.flatnonzero(np.isin(inv, te)))
            for tr, te in cv.split(unique, labels)]


def grouped_split(y, groups, seed):
    unique, inv, labels = _group_labels(y, groups)
    idx = np.arange(len(unique))
    tr, rest = train_test_split(idx, test_size=.4, stratify=labels, random_state=seed)
    va, te = train_test_split(rest, test_size=.5, stratify=labels[rest], random_state=seed)
    # Exact 60/20/20 of groups up to rounding; row ratios can differ by group sizes.
    arrays = tuple(np.flatnonzero(np.isin(inv, part)) for part in (tr, va, te))
    validate_split(y, groups, *arrays)
    return arrays
