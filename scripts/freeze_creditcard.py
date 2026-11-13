#!/usr/bin/env python
"""Freeze Credit Card Fraud dataset: deduplication, 20 stratified splits, and manifest.

Produces:
  - data/manifest.json     : Complete dataset provenance, SHA-256 hashes, split stats
  - data/splits_20.npz     : Frozen indices for 20 splits (seeds 0..19)
  - data/creditcard_clean.csv (optional): Deduplicated dataset

Usage:
  python scripts/freeze_creditcard.py
  python scripts/freeze_creditcard.py --verify
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CSV_PATH = os.path.join(ROOT, "data", "creditcard.csv")
MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.json")
SPLITS_PATH = os.path.join(ROOT, "data", "splits_20.npz")
CLEAN_CSV_PATH = os.path.join(ROOT, "data", "creditcard_clean.csv")

ALL_FEATURES = ["Amount"] + [f"V{i}" for i in range(1, 29)]
LABEL_COL = "Class"


def compute_sha256(filepath):
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_array_sha256(arr):
    """Compute SHA-256 of a numpy array's contiguous bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_and_inspect_raw(csv_path):
    """Load raw CSV and perform integrity inspection."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Credit card data file not found at {csv_path}")

    raw_sha = compute_sha256(csv_path)
    file_size_bytes = os.path.getsize(csv_path)

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.rename(columns={c: c.strip() for c in df_raw.columns})

    n_raw = len(df_raw)
    cols_raw = list(df_raw.columns)
    frauds_raw = int(df_raw[LABEL_COL].sum())
    legit_raw = n_raw - frauds_raw

    # Check for presence of features
    missing_feats = [c for c in ALL_FEATURES if c not in cols_raw]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}")

    # Check NaN/Inf in feature columns
    feat_data = df_raw[ALL_FEATURES]
    if feat_data.isnull().any().any():
        raise ValueError("NaN values found in feature columns — check data source.")
    if np.isinf(feat_data.values.astype(float)).any():
        raise ValueError("Inf values found in feature columns — check data source.")

    # Build group key: float64 feature vector, no label, no rounding
    group_ids = df_raw.groupby(ALL_FEATURES, sort=False).ngroup().to_numpy(dtype=np.int64)
    n_groups = int(group_ids.max()) + 1

    # Exact duplicates vs feature duplicates
    exact_dups = int(df_raw.duplicated().sum())
    group_sizes = df_raw.groupby(ALL_FEATURES).size()
    n_duplicate_rows = int((group_sizes[group_sizes > 1]).sum())   # rows in multi-row groups
    excess_rows = int((group_sizes - 1).clip(lower=0).sum())       # Σ(nₕ - 1)
    max_group_size = int(group_sizes.max())

    # Conflicting labels within feature groups
    group_classes = df_raw.groupby(ALL_FEATURES)[LABEL_COL].nunique()
    conflicting_groups = int((group_classes > 1).sum())

    return {
        "df_raw": df_raw,
        "group_ids": group_ids,
        "n_groups": n_groups,
        "raw_sha256": raw_sha,
        "raw_size_bytes": file_size_bytes,
        "n_raw": n_raw,
        "cols_raw": cols_raw,
        "frauds_raw": frauds_raw,
        "legit_raw": legit_raw,
        "exact_duplicates": exact_dups,
        "n_duplicate_rows": n_duplicate_rows,
        "excess_rows": excess_rows,
        "max_group_size": max_group_size,
        "conflicting_groups": conflicting_groups,
    }


def deduplicate_dataset(df_raw):
    """Standardize representation and deduplicate exact and feature duplicates.

    Default behaviour: drop exact feature-label duplicates (keep=first).
    Rows with same features but different labels are KEPT (conflicting groups).
    Use as a sensitivity analysis only; not the default for split generation.
    """
    df_clean = df_raw.drop_duplicates(subset=ALL_FEATURES + [LABEL_COL]).reset_index(drop=True)
    n_clean = len(df_clean)
    frauds_clean = int(df_clean[LABEL_COL].sum())
    legit_clean = n_clean - frauds_clean
    return df_clean, n_clean, frauds_clean, legit_clean


def generate_splits(df, group_ids, n_splits=20, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Generate 20 deterministic group-aware stratified 60/20/20 splits.

    Groups (identical feature vectors) are kept intact — no group spans two
    different partitions. Stratification is by label at the GROUP level.

    Strategy:
      Unique groups are partitioned using 5-fold StratifiedKFold(shuffle=True, random_state=seed).
      For seed s: fold (s % 5) -> test, fold ((s + 1) % 5) -> val, rest -> train.
      Rotation across seeds ensures all 20 splits are statistically independent and distinct.
    """
    y = df[LABEL_COL].to_numpy(dtype=int)
    n = len(df)
    idx = np.arange(n, dtype=np.int32)

    unique_groups, group_first_idx = np.unique(group_ids, return_index=True)
    group_y = y[group_first_idx]
    n_groups = len(unique_groups)

    splits_dict = {}
    splits_meta = []
    # 2D matrix of shape (n_samples, n_splits): 0=train, 1=val, 2=test
    split_labels_matrix = np.empty((n, n_splits), dtype=np.uint8)

    for seed in range(n_splits):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(skf.split(unique_groups, group_y))

        test_fold = seed % 5
        val_fold = (seed + 1) % 5

        group_fold = np.empty(n_groups, dtype=np.int8)
        for f_idx, (_, test_grp_indices) in enumerate(folds):
            group_fold[unique_groups[test_grp_indices]] = f_idx

        sample_fold = group_fold[group_ids]
        # Assign: 0 = train, 1 = val, 2 = test
        sl = np.zeros(n, dtype=np.uint8)
        sl[sample_fold == val_fold] = 1
        sl[sample_fold == test_fold] = 2
        split_labels_matrix[:, seed] = sl

        tr = idx[sl == 0].astype(np.int32)
        va = idx[sl == 1].astype(np.int32)
        te = idx[sl == 2].astype(np.int32)

        # Invariant checks
        assert len(set(tr) & set(va)) == 0, f"Overlap train/val seed {seed}"
        assert len(set(tr) & set(te)) == 0, f"Overlap train/test seed {seed}"
        assert len(set(va) & set(te)) == 0, f"Overlap val/test seed {seed}"
        assert len(tr) + len(va) + len(te) == n, f"Size mismatch seed {seed}"

        # Group integrity: no group spans two partitions
        for part_a, part_b in [(tr, va), (tr, te), (va, te)]:
            groups_a = set(group_ids[part_a])
            groups_b = set(group_ids[part_b])
            assert len(groups_a & groups_b) == 0, (
                f"Group leakage between partitions at seed {seed}"
            )

        hash_tr = compute_array_sha256(tr)
        hash_va = compute_array_sha256(va)
        hash_te = compute_array_sha256(te)

        splits_dict[f"train_{seed}"] = tr
        splits_dict[f"val_{seed}"] = va
        splits_dict[f"test_{seed}"] = te

        splits_meta.append({
            "seed": seed,
            "train": {
                "n": int(len(tr)),
                "frauds": int(y[tr].sum()),
                "legit": int((1 - y[tr]).sum()),
                "fraud_rate": float(y[tr].mean()),
                "hash_sha256": hash_tr,
            },
            "val": {
                "n": int(len(va)),
                "frauds": int(y[va].sum()),
                "legit": int((1 - y[va]).sum()),
                "fraud_rate": float(y[va].mean()),
                "hash_sha256": hash_va,
            },
            "test": {
                "n": int(len(te)),
                "frauds": int(y[te].sum()),
                "legit": int((1 - y[te]).sum()),
                "fraud_rate": float(y[te].mean()),
                "hash_sha256": hash_te,
            },
        })

    # Also store compact split_labels matrix (uint8 per sample per seed)
    splits_dict["split_labels"] = split_labels_matrix

    return splits_dict, splits_meta


def freeze_data(save_clean_csv=True):
    """Run full freeze procedure, export manifest and splits.

    Splits are GROUP-AWARE: rows with identical feature vectors are assigned
    to the same partition.  No deduplication is applied by default; pass
    --dedupe for the sensitivity analysis.
    """
    t0 = time.time()
    print("=== ĐÓNG BĂNG DỮ LIỆU CREDIT CARD FRAUD ===")
    print(f"File nguồn: {CSV_PATH}")

    raw_info = load_and_inspect_raw(CSV_PATH)
    df_raw = raw_info.pop("df_raw")
    group_ids = raw_info.pop("group_ids")
    n_groups = raw_info.pop("n_groups")

    print(f"Kích thước gốc: {raw_info['n_raw']} dòng, {len(raw_info['cols_raw'])} cột")
    print(f"Số gian lận gốc: {raw_info['frauds_raw']} ({100*raw_info['frauds_raw']/raw_info['n_raw']:.3f}%)")
    print(f"Số nhóm đặc trưng: {n_groups}  |  bản trùng (Σ(nₕ-1)): {raw_info['excess_rows']}")
    print(f"Số nhóm xung đột nhãn: {raw_info['conflicting_groups']}")
    print(f"SHA-256 gốc: {raw_info['raw_sha256']}")

    # Splits on RAW data (all rows kept, groups intact)
    print("\nĐang sinh 20 splits group-aware phân tầng 60/20/20...")
    splits_dict, splits_meta = generate_splits(df_raw, group_ids, n_splits=20)

    # Save splits to compressed npz
    np.savez_compressed(SPLITS_PATH, **splits_dict)
    splits_sha256 = compute_sha256(SPLITS_PATH)
    print(f"Đã lưu các split vào: {SPLITS_PATH} (SHA-256: {splits_sha256})")

    # Optional: also save a deduplicated CSV for sensitivity analysis
    clean_sha = None
    if save_clean_csv:
        df_clean, n_clean, frauds_clean, _ = deduplicate_dataset(df_raw)
        df_clean.to_csv(CLEAN_CSV_PATH, index=False)
        clean_sha = compute_sha256(CLEAN_CSV_PATH)
        print(f"Đã xuất file sạch (sensitivity only): {CLEAN_CSV_PATH}")
        print(f"SHA-256 file sạch: {clean_sha}")
    else:
        n_clean, frauds_clean = None, None

    # Build manifest
    manifest = {
        "metadata": {
            "dataset_name": "Credit Card Fraud Detection (ULB / Machine Learning Group)",
            "openml_data_id": 1597,
            "version": "2.0-frozen-week4-group-aware",
            "date_frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
            "author": "Nguyen Huu Phuoc (Nhom 01 - CO3101)",
            "preprocessing_version": "group_aware_v2_stratified_60_20_20",
        },
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "raw_file": {
            "path": os.path.relpath(CSV_PATH, ROOT),
            "size_bytes": raw_info["raw_size_bytes"],
            "sha256": raw_info["raw_sha256"],
            "rows": raw_info["n_raw"],
            "columns": len(raw_info["cols_raw"]),
            "feature_columns": ALL_FEATURES,
            "label_column": LABEL_COL,
            "frauds": raw_info["frauds_raw"],
            "legitimate": raw_info["legit_raw"],
            "n_groups": n_groups,
            "n_duplicate_rows": raw_info["n_duplicate_rows"],
            "excess_rows": raw_info["excess_rows"],
            "max_group_size": raw_info["max_group_size"],
            "conflicting_label_groups": raw_info["conflicting_groups"],
        },
        "sensitivity_deduplicated": {
            "path": os.path.relpath(CLEAN_CSV_PATH, ROOT) if save_clean_csv else None,
            "sha256": clean_sha,
            "rows": n_clean,
            "frauds": frauds_clean,
            "note": "Sensitivity analysis only. Not used for split generation.",
        },
        "splits_archive": {
            "path": os.path.relpath(SPLITS_PATH, ROOT),
            "sha256": splits_sha256,
            "n_splits": 20,
            "split_basis": "raw rows (no deduplication); groups kept intact",
            "split_method": "StratifiedGroupKFold(5): fold(seed%5)=test, fold((seed+1)%5)=val, rest=train",
            "ratios_approx": {"train": 0.60, "val": 0.20, "test": 0.20},
            "group_aware": True,
        },
        "splits": splits_meta,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Đã lưu manifest: {MANIFEST_PATH}")
    print(f"Hoàn thành trong {time.time() - t0:.2f}s!")
    return manifest


def verify_manifest():
    """Verify that current files match manifest checksums and invariant properties."""
    print("=== KIỂM TRA TÍNH TOÀN VẸN (VERIFY INTEGRITY) ===")
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify raw CSV hash matches manifest
    raw_path = os.path.join(ROOT, manifest["raw_file"]["path"])
    raw_sha = compute_sha256(raw_path)
    assert raw_sha == manifest["raw_file"]["sha256"], (
        f"Raw SHA mismatch: {raw_sha} vs {manifest['raw_file']['sha256']}"
    )
    print(f"✓ Raw CSV SHA-256 match: {raw_sha}")

    # 2. Verify splits npz
    splits_path = os.path.join(ROOT, manifest["splits_archive"]["path"])
    splits_sha = compute_sha256(splits_path)
    assert splits_sha == manifest["splits_archive"]["sha256"], "Splits archive SHA mismatch!"
    print(f"✓ Splits archive SHA-256 match: {splits_sha}")

    # 3. Verify sensitivity clean CSV if present
    sens = manifest.get("sensitivity_deduplicated") or manifest.get("processed_file", {})
    if sens.get("path") and sens.get("sha256"):
        clean_path = os.path.join(ROOT, sens["path"])
        if os.path.exists(clean_path):
            clean_sha = compute_sha256(clean_path)
            assert clean_sha == sens["sha256"], "Clean CSV SHA mismatch!"
            print(f"✓ Clean CSV SHA-256 match: {clean_sha}")

    # 4. Reload raw to get group IDs for group integrity check
    raw_info = load_and_inspect_raw(raw_path)
    group_ids = raw_info["group_ids"]

    # 5. Verify every split: sizes, hashes, disjointness, group integrity
    loaded_splits = np.load(splits_path)
    for s_info in manifest["splits"]:
        seed = s_info["seed"]
        tr = loaded_splits[f"train_{seed}"]
        va = loaded_splits[f"val_{seed}"]
        te = loaded_splits[f"test_{seed}"]

        assert len(tr) == s_info["train"]["n"], f"seed {seed}: train size mismatch"
        assert len(va) == s_info["val"]["n"],   f"seed {seed}: val size mismatch"
        assert len(te) == s_info["test"]["n"],  f"seed {seed}: test size mismatch"

        assert compute_array_sha256(tr) == s_info["train"]["hash_sha256"], f"seed {seed}: train hash"
        assert compute_array_sha256(va) == s_info["val"]["hash_sha256"],   f"seed {seed}: val hash"
        assert compute_array_sha256(te) == s_info["test"]["hash_sha256"],  f"seed {seed}: test hash"

        assert len(set(tr) & set(va)) == 0, f"seed {seed}: train/val overlap"
        assert len(set(tr) & set(te)) == 0, f"seed {seed}: train/test overlap"
        assert len(set(va) & set(te)) == 0, f"seed {seed}: val/test overlap"

        # Group integrity: no group spans two partitions
        for part_a, part_b, label_ab in [
            (tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")
        ]:
            shared_groups = set(group_ids[part_a]) & set(group_ids[part_b])
            assert len(shared_groups) == 0, (
                f"seed {seed}: group leakage between {label_ab}: {len(shared_groups)} groups"
            )

    print(f"✓ Toàn bộ 20 splits đều khớp hash, không giao nhau, và không rò nhóm!")
    print("INTEGRITY CHECK PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze dataset and generate manifest")
    parser.add_argument("--verify", action="store_true", help="Verify integrity against existing manifest")
    parser.add_argument("--no-clean-csv", action="store_true", help="Skip saving cleaned CSV file")
    args = parser.parse_args()

    if args.verify:
        verify_manifest()
    else:
        freeze_data(save_clean_csv=not args.no_clean_csv)
