#!/usr/bin/env python
"""Freeze retained rows and feature-group splits for the locked week-5 protocol."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from threeaction.data_split import feature_groups, grouped_split


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', default=str(ROOT/'data'))
    a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.csv);df.columns=[c.strip() for c in df.columns]
    features=['Amount']+[f'V{i}' for i in range(1,29)]
    y=df['Class'].to_numpy()
    if not np.isin(y,[0,1]).all():raise ValueError('Labels must be binary')
    groups=feature_groups(df[features]);sizes=np.bincount(groups);arrays={};stats=[]
    for seed in range(20):
        row={'seed':seed}
        for part,idx in zip(('train','val','test'),grouped_split(y,groups,seed)):
            arrays[f'{part}_{seed}']=idx
            row[part]={'rows':len(idx),'frauds':int(y[idx].sum()),'groups':len(np.unique(groups[idx])),
                       'sha256':hashlib.sha256(idx.tobytes()).hexdigest()}
        stats.append(row)
    path=out/'splits_week5_grouped.npz';np.savez_compressed(path,**arrays)
    manifest={'protocol':'retained rows; feature float64 groups; any-positive group stratification',
              'csv_sha256':hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
              'splits_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'rows':len(y),'groups':len(sizes),'duplicate_groups':int(sum(sizes>1)),
              'rows_in_duplicate_groups':int(sizes[sizes>1].sum()),'redundant_rows':int((sizes-1).sum()),
              'ratio_note':'60/20/20 of groups, row ratios approximate; no group crosses partitions',
              'splits':stats}
    (out/'manifest_week5_grouped.json').write_text(json.dumps(manifest,indent=2))
    print(f'Frozen {len(y)} rows in {len(sizes)} groups; 20 seeds. No rows dropped.')

if __name__=='__main__':main()
