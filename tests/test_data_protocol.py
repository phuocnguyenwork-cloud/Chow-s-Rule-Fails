import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from threeaction.data_split import feature_groups, grouped_split, validate_split, group_oof_splits


def test_duplicates_conflicting_labels_stay_together_and_rows_retained():
    X = np.repeat(np.arange(100)[:, None], 3, axis=0).astype(float)
    y = np.repeat(np.arange(100) % 2, 3); y[0] = 1
    groups = feature_groups(X)
    split = grouped_split(y, groups, 0)
    validate_split(y, groups, *split)
    assert sum(map(len, split)) == 300
    assert all(np.array_equal(a,b) for a,b in zip(split,grouped_split(y,groups,0)))
    for tr, te in group_oof_splits(y,groups,0):
        assert not set(groups[tr]) & set(groups[te])


def test_reject_stale_and_cross_group_archive():
    y=np.tile([0,1],6);groups=np.arange(12);groups[4]=groups[0]
    with pytest.raises(ValueError,match='crosses'):
        validate_split(y,groups,np.arange(4),np.arange(4,8),np.arange(8,12))
    with pytest.raises(ValueError,match='cover'):
        validate_split(y,np.arange(12),np.arange(4),np.arange(4,8),np.arange(8,11))


def test_float_identity_without_rounding():
    assert feature_groups([[1.0], [1], [np.nextafter(1.0, 2)] ]).tolist()==[0,0,1]
    with pytest.raises(ValueError):feature_groups([[np.nan]])


def test_calibrator_uses_only_oof_training_rows(monkeypatch):
    path=Path(__file__).parents[1]/'scripts/run_creditcard.py'
    spec=importlib.util.spec_from_file_location('creditcard_protocol_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(4);X=pd.DataFrame(rng.normal(size=(500,29)),columns=mod.ALL_FEATURES)
    y=np.tile([0,1],250);groups=feature_groups(X);tr,va,te=grouped_split(y,groups,0)
    archive={'train_0':tr,'val_0':va,'test_0':te}
    seen=[];original=mod.PlattCalibrator
    class Spy(original):
        def fit(self,p,y):
            seen.append((p.copy(),y.copy()));return super().fit(p,y)
    monkeypatch.setattr(mod,'PlattCalibrator',Spy)
    monkeypatch.setattr(mod,'run_policies',lambda *a,**k: {})
    mod.one_split(X,y,groups,0,10,'logit',5,5,archive)
    altered=y.copy();altered[va]=1-altered[va];altered[te]=1-altered[te]
    mod.one_split(X,altered,groups,0,10,'logit',5,5,archive)
    assert len(seen) == 4  # tier 1 and tier 2, before and after perturbation
    for first, second in ((seen[0], seen[2]), (seen[1], seen[3])):
        np.testing.assert_array_equal(first[1], y[tr])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
