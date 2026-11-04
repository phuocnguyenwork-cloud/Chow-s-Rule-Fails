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
from sklearn.model_selection import train_test_split

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

    # Inspect exact duplicates vs feature duplicates
    exact_dups = int(df_raw.duplicated().sum())
    feat_dups = int(df_raw.duplicated(subset=ALL_FEATURES, keep=False).sum())

    # Check conflicting labels within feature duplicates
    group_classes = df_raw.groupby(ALL_FEATURES)[LABEL_COL].nunique()
    conflicting_groups = int((group_classes > 1).sum())

    return {
        "df_raw": df_raw,
        "raw_sha256": raw_sha,
        "raw_size_bytes": file_size_bytes,
        "n_raw": n_raw,
        "cols_raw": cols_raw,
        "frauds_raw": frauds_raw,
        "legit_raw": legit_raw,
        "exact_duplicates": exact_dups,
        "feature_duplicates_total_rows": feat_dups,
        "conflicting_groups": conflicting_groups,
    }


def deduplicate_dataset(df_raw):
    """Standardize representation and deduplicate exact and feature duplicates."""
    # Ensure float64 precision and deduplicate by features + label
    df_clean = df_raw.drop_duplicates(subset=ALL_FEATURES + [LABEL_COL]).reset_index(drop=True)
    n_clean = len(df_clean)
    frauds_clean = int(df_clean[LABEL_COL].sum())
    legit_clean = n_clean - frauds_clean
    return df_clean, n_clean, frauds_clean, legit_clean


def generate_splits(df_clean, n_splits=20, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Generate 20 deterministic stratified 60/20/20 train/val/test splits."""
    y = df_clean[LABEL_COL].to_numpy(dtype=int)
    n = len(y)
    idx = np.arange(n, dtype=np.int32)

    splits_dict = {}
    splits_meta = []

    for seed in range(n_splits):
        # 60% train, 40% rest
        tr, rest = train_test_split( 
            idx,
            test_size=(val_ratio + test_ratio),
            stratify=y,
            random_state=seed,
        )
        # 20% val, 20% test (50/50 of rest)
        va, te = train_test_split(
            rest,
            test_size=0.5,
            stratify=y[rest],
            random_state=seed,
        )

        tr = np.asarray(tr, dtype=np.int32)
        va = np.asarray(va, dtype=np.int32)
        te = np.asarray(te, dtype=np.int32)

        # Invariant checks
        assert len(set(tr).intersection(va)) == 0, f"Overlap in seed {seed} train/val"
        assert len(set(tr).intersection(te)) == 0, f"Overlap in seed {seed} train/test"
        assert len(set(va).intersection(te)) == 0, f"Overlap in seed {seed} val/test"
        assert len(tr) + len(va) + len(te) == n, f"Size mismatch in seed {seed}"

        # Hash individual index arrays
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

    return splits_dict, splits_meta


def freeze_data(save_clean_csv=True):
    """Run full freeze procedure, export manifest and splits."""
    t0 = time.time()
    print("=== ĐÓNG BĂNG DỮ LIỆU CREDIT CARD FRAUD ===")
    print(f"File nguồn: {CSV_PATH}")

    raw_info = load_and_inspect_raw(CSV_PATH)
    df_raw = raw_info.pop("df_raw")

    print(f"Kích thước gốc: {raw_info['n_raw']} dòng, {len(raw_info['cols_raw'])} cột")
    print(f"Số gian lận gốc: {raw_info['frauds_raw']} ({100*raw_info['frauds_raw']/raw_info['n_raw']:.3f}%)")
    print(f"Số bản trùng chính xác (exact duplicates): {raw_info['exact_duplicates']}")
    print(f"Số nhóm đặc trưng xung đột nhãn: {raw_info['conflicting_groups']}")
    print(f"SHA-256 gốc: {raw_info['raw_sha256']}")

    df_clean, n_clean, frauds_clean, legit_clean = deduplicate_dataset(df_raw)
    print(f"\nSau khi gom bản trùng: {n_clean} dòng, {frauds_clean} gian lận ({100*frauds_clean/n_clean:.3f}%)")
    print(f"Đã loại bỏ/gom: {raw_info['n_raw'] - n_clean} dòng trùng lặp.")

    clean_sha = None
    if save_clean_csv:
        df_clean.to_csv(CLEAN_CSV_PATH, index=False)
        clean_sha = compute_sha256(CLEAN_CSV_PATH)
        print(f"Đã xuất file sạch: {CLEAN_CSV_PATH}")
        print(f"SHA-256 file sạch: {clean_sha}")

    print("\nĐang sinh 20 splits phân tầng 60/20/20...")
    splits_dict, splits_meta = generate_splits(df_clean, n_splits=20)

    # Save splits to compressed npz
    np.savez_compressed(SPLITS_PATH, **splits_dict)
    splits_sha256 = compute_sha256(SPLITS_PATH)
    print(f"Đã lưu các split vào: {SPLITS_PATH} (SHA-256: {splits_sha256})")

    # Build manifest
    manifest = {
        "metadata": {
            "dataset_name": "Credit Card Fraud Detection (ULB / Machine Learning Group)",
            "openml_data_id": 1597,
            "version": "1.0-frozen-week3",
            "date_frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
            "author": "Nguyen Huu Phuoc (Nhom 01 - CO3101)",
            "preprocessing_version": "dedupe_v1_stratified_60_20_20",
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
            "frauds": raw_info["frauds_raw"],
            "legitimate": raw_info["legit_raw"],
            "exact_duplicate_rows": raw_info["exact_duplicates"],
            "conflicting_groups": raw_info["conflicting_groups"],
        },
        "processed_file": {
            "path": os.path.relpath(CLEAN_CSV_PATH, ROOT) if save_clean_csv else None,
            "sha256": clean_sha,
            "rows": n_clean,
            "columns": len(df_clean.columns),
            "frauds": frauds_clean,
            "legitimate": legit_clean,
            "rows_removed_or_merged": raw_info["n_raw"] - n_clean,
            "policy_for_duplicates": "Group-level deduplication: drop identical feature-label duplicates, preserving true positive class ratio.",
        },
        "splits_archive": {
            "path": os.path.relpath(SPLITS_PATH, ROOT),
            "sha256": splits_sha256,
            "n_splits": 20,
            "ratios": {"train": 0.60, "val": 0.20, "test": 0.20},
            "stratified": True,
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

    # 1. Verify raw CSV
    raw_path = os.path.join(ROOT, manifest["raw_file"]["path"])
    raw_sha = compute_sha256(raw_path)
    assert raw_sha == manifest["raw_file"]["sha256"], f"Raw SHA mismatch: {raw_sha} vs {manifest['raw_file']['sha256']}"
    print(f"✓ Raw CSV SHA-256 match: {raw_sha}")

    # 2. Verify splits npz
    splits_path = os.path.join(ROOT, manifest["splits_archive"]["path"])
    splits_sha = compute_sha256(splits_path)
    assert splits_sha == manifest["splits_archive"]["sha256"], "Splits archive SHA mismatch!"
    print(f"✓ Splits archive SHA-256 match: {splits_sha}")

    # 3. Verify clean CSV if exists
    if manifest["processed_file"]["path"]:
        clean_path = os.path.join(ROOT, manifest["processed_file"]["path"])
        clean_sha = compute_sha256(clean_path)
        assert clean_sha == manifest["processed_file"]["sha256"], "Clean CSV SHA mismatch!"
        print(f"✓ Clean CSV SHA-256 match: {clean_sha}")

    # 4. Verify splits structure and indices
    loaded_splits = np.load(splits_path)
    for s_info in manifest["splits"]:
        seed = s_info["seed"]
        tr = loaded_splits[f"train_{seed}"]
        va = loaded_splits[f"val_{seed}"]
        te = loaded_splits[f"test_{seed}"]

        assert len(tr) == s_info["train"]["n"]
        assert len(va) == s_info["val"]["n"]
        assert len(te) == s_info["test"]["n"]

        assert compute_array_sha256(tr) == s_info["train"]["hash_sha256"]
        assert compute_array_sha256(va) == s_info["val"]["hash_sha256"]
        assert compute_array_sha256(te) == s_info["test"]["hash_sha256"]

        assert len(set(tr).intersection(va)) == 0
        assert len(set(tr).intersection(te)) == 0
        assert len(set(va).intersection(te)) == 0

    print(f"✓ Toàn bộ 20 splits đều khớp hash và không giao nhau!")
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
