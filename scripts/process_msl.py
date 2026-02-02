# -*- coding: utf-8 -*-
"""scripts/process_msl.py

Telemanom-style MSL preprocessing.

Outputs:
- PKL mode (repo-compatible):
  - data/MSL/MSL_train.pkl: (T_train, N)
  - data/MSL/MSL_test.pkl:  (T_test, N+1)  last col is label

- CSV mode (SWaT/WADI-style under dataset/MSL):
  - dataset/MSL/msl_train.csv: features only
  - dataset/MSL/msl_test.csv: features + attack label column
  - dataset/MSL/list.txt: node names (channel_id)
"""

import os
import ast
import argparse
import pickle
import typing
from typing import List, Tuple, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _downsample_features_median(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    if x.shape[0] < factor:
        return x[0:0]
    usable = (x.shape[0] // factor) * factor
    x = x[:usable]
    x = x.reshape(-1, factor, x.shape[1])
    return np.median(x, axis=1)


def _downsample_labels_max(y: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return y
    if y.shape[0] < factor:
        return y[0:0]
    usable = (y.shape[0] // factor) * factor
    y = y[:usable]
    y = y.reshape(-1, factor)
    return np.max(y, axis=1)


def _skip_head(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return arr
    if arr.shape[0] <= n:
        return arr[0:0]
    return arr[n:]


def _truncate(arr: np.ndarray, max_rows: Optional[int]) -> np.ndarray:
    if max_rows is None:
        return arr
    if max_rows <= 0:
        return arr[0:0]
    return arr[: min(arr.shape[0], max_rows)]


def load_meta(raw_root: str) -> pd.DataFrame:
    """Read labeled_anomalies.csv and keep spacecraft == 'MSL'."""
    csv_path = os.path.join(raw_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labeled_anomalies.csv: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Telemanom usually uses channel_id; some forks use chan_id in docs.
    col_map = {
        "channel_id": "channel_id",
        "chan_id": "channel_id",
    }
    if "channel_id" not in df.columns and "chan_id" in df.columns:
        df = df.rename(columns={"chan_id": "channel_id"})

    required = {"channel_id", "spacecraft", "anomaly_sequences"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"labeled_anomalies.csv missing columns: {required}, got: {list(df.columns)}")

    df_msl = df[df["spacecraft"].astype(str).str.upper() == "MSL"].copy()
    if df_msl.empty:
        raise ValueError("No rows with spacecraft == MSL in labeled_anomalies.csv")

    return df_msl


def parse_anomaly_sequences(seq_str) -> List[Tuple[int, int]]:
    """Parse anomaly_sequences cell like '[[2149, 2409], [2581, 2707]]'."""
    if seq_str is None:
        return []
    if isinstance(seq_str, float) and np.isnan(seq_str):
        return []

    text = str(seq_str).strip()
    if not text:
        return []

    try:
        seqs = ast.literal_eval(text)
    except Exception:
        # fallback: extract integers
        import re
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
        return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]

    out: List[Tuple[int, int]] = []
    if isinstance(seqs, (list, tuple)):
        for seg in seqs:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                try:
                    out.append((int(seg[0]), int(seg[1])))
                except Exception:
                    continue
    return out


def _load_channel_array(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape((-1, arr.shape[-1]))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape={arr.shape}")
    # Replace NaN/inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def preprocess_msl(
    raw_root: str,
    out_data_dir: str,
    out_dataset_dir: str,
    use_global_scaler: bool = False,
    telemetry_col: int = 0,
    output_format: str = "pkl",
    *,
    downsample_factor: int = 1,
    skip_head: int = 0,
    max_train_rows_per_channel: Optional[int] = None,
    max_test_rows_per_channel: Optional[int] = None,
    channels: Optional[List[str]] = None,
    max_channels: Optional[int] = None,
):
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_dataset_dir, exist_ok=True)

    fmt = str(output_format).lower()
    if fmt not in {"pkl", "csv", "both"}:
        raise ValueError("output_format must be one of: pkl, csv, both")

    train_dir = os.path.join(raw_root, "train")
    test_dir = os.path.join(raw_root, "test")

    meta = load_meta(raw_root)
    all_chan_ids = sorted(meta["channel_id"].astype(str).tolist())

    # Keep only channels that exist in both train/test
    all_chan_ids = [
        cid
        for cid in all_chan_ids
        if os.path.exists(os.path.join(train_dir, f"{cid}.npy"))
        and os.path.exists(os.path.join(test_dir, f"{cid}.npy"))
    ]
    if not all_chan_ids:
        raise FileNotFoundError("No channels found with both train/<cid>.npy and test/<cid>.npy")

    # filter channels (SMD-like)
    if channels:
        channels = [str(c).strip() for c in channels if str(c).strip()]
        missing = [c for c in channels if c not in all_chan_ids]
        if missing:
            raise ValueError(f"Unknown channel ids: {missing}. Available (first 20): {all_chan_ids[:20]}")
        chan_ids = channels
    else:
        chan_ids = list(all_chan_ids)

    if max_channels is not None:
        chan_ids = chan_ids[: int(max_channels)]

    if not chan_ids:
        raise RuntimeError("No channels selected.")

    print(f"[MSL] channels found={len(all_chan_ids)}")
    print(f"[MSL] processing channels={len(chan_ids)}: {chan_ids[:5]}{'...' if len(chan_ids) > 5 else ''}")

    # Index meta by channel id to locate anomaly_sequences
    meta = meta.set_index("channel_id")

    # Optional global scaler fitted on *all channels train data* (all inputs)
    global_scaler: Optional[StandardScaler] = None
    if use_global_scaler:
        print("[MSL] fitting global StandardScaler on all channels train data (55 features)...")
        all_train = []
        for cid in chan_ids:
            train_raw = _load_channel_array(os.path.join(train_dir, f"{cid}.npy"))
            all_train.append(train_raw)
        all_train_cat = np.concatenate(all_train, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(all_train_cat)

    # In this repo, MSL uses (samples, features) with features=55.
    # Each <cid>.npy has shape (T_i, 55). We concatenate segments along time.
    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    test_label_blocks: List[np.ndarray] = []

    for cid in chan_ids:
        train_raw = _load_channel_array(os.path.join(train_dir, f"{cid}.npy"))
        test_raw = _load_channel_array(os.path.join(test_dir, f"{cid}.npy"))

        if use_global_scaler:
            scaler = global_scaler
        else:
            scaler = StandardScaler()
            scaler.fit(train_raw)

        train_norm = scaler.transform(train_raw).astype(np.float32)
        test_norm = scaler.transform(test_raw).astype(np.float32)

        # per-timestep label for this segment on test
        T = test_norm.shape[0]
        y = np.zeros((T,), dtype=np.int64)

        if cid in meta.index:
            intervals = parse_anomaly_sequences(meta.loc[cid, "anomaly_sequences"])
            for s, e in intervals:
                # Telemanom: usually inclusive [s,e]
                s = max(0, int(s))
                e = min(T - 1, int(e))
                if s <= e:
                    y[s : e + 1] = 1

        # downsample like SWaT/WADI/SMD
        if downsample_factor and downsample_factor > 1:
            train_norm = _downsample_features_median(train_norm, int(downsample_factor))
            test_norm = _downsample_features_median(test_norm, int(downsample_factor))
            y = _downsample_labels_max(y, int(downsample_factor))

        # optional skip_head
        if skip_head and skip_head > 0:
            train_norm = _skip_head(train_norm, int(skip_head))
            test_norm = _skip_head(test_norm, int(skip_head))
            y = _skip_head(y, int(skip_head))

        # optional truncate
        train_norm = _truncate(train_norm, max_train_rows_per_channel)
        test_norm = _truncate(test_norm, max_test_rows_per_channel)
        y = _truncate(y, max_test_rows_per_channel)

        # align test/label length
        if test_norm.shape[0] != y.shape[0]:
            m = min(test_norm.shape[0], y.shape[0])
            test_norm = test_norm[:m]
            y = y[:m]

        train_blocks.append(train_norm)
        test_blocks.append(test_norm)
        test_label_blocks.append(y)

        print(f"[MSL:{cid}] train={train_norm.shape} test={test_norm.shape} label={y.shape}")

    x_train = np.concatenate(train_blocks, axis=0).astype(np.float32)
    x_test = np.concatenate(test_blocks, axis=0).astype(np.float32)
    y_test = np.concatenate(test_label_blocks, axis=0).astype(np.int64)

    print(f"[MSL] concatenated lengths: train={x_train.shape[0]}, test={x_test.shape[0]}, features={x_train.shape[1]}")

    # list.txt in dataset/MSL for segment names (channel_id)
    list_path = os.path.join(out_dataset_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for cid in chan_ids:
            f.write(str(cid) + "\n")

    # PKL outputs
    if fmt in {"pkl", "both"}:
        train_pkl = os.path.join(out_data_dir, "MSL_train.pkl")
        test_pkl = os.path.join(out_data_dir, "MSL_test.pkl")

        test_with_label = np.concatenate([x_test, y_test.reshape(-1, 1).astype(np.float32)], axis=1)

        with open(train_pkl, "wb") as f:
            pickle.dump(x_train, cast(typing.IO[bytes], f))
        with open(test_pkl, "wb") as f:
            pickle.dump(test_with_label, cast(typing.IO[bytes], f))

        print("[OK] wrote PKL:")
        print(" ", train_pkl, x_train.shape)
        print(" ", test_pkl, test_with_label.shape, "(last col=label)")

    # CSV outputs
    if fmt in {"csv", "both"}:
        # CSV contract (align with SWaT/WADI loaders):
        # - train: features only, shape (T_train, 55)
        # - test: features + 'attack' label column, shape (T_test, 56)
        feat_cols = [f"f{i}" for i in range(x_train.shape[1])]
        train_df = pd.DataFrame(x_train, columns=feat_cols)
        test_df = pd.DataFrame(x_test, columns=feat_cols)
        test_df["attack"] = y_test.astype(int)

        train_csv = os.path.join(out_dataset_dir, "msl_train.csv")
        test_csv = os.path.join(out_dataset_dir, "msl_test.csv")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print("[OK] wrote CSV:")
        print(" ", train_csv, train_df.shape)
        print(" ", test_csv, test_df.shape)

    print("[OK] wrote list:")
    print(" ", list_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MSL (Telemanom-style) into PKL and/or CSV files")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=os.path.join("data", "MSL"),
        help="raw MSL root containing train/ test/ labeled_anomalies.csv",
    )
    # Follow README: pkl lives under data/MSL, CSV lives under dataset/MSL
    parser.add_argument(
        "--out_data_dir",
        type=str,
        default=os.path.join("dataset", "MSL"),
        help="output dir for PKL files (default: data/MSL)",
    )
    parser.add_argument(
        "--out_dataset_dir",
        type=str,
        default=os.path.join("dataset", "MSL"),
        help="output dir for CSV/list.txt (default: dataset/MSL)",
    )
    parser.add_argument("--output_format", type=str, default="csv", choices=["pkl", "csv", "both"], help="output format")
    parser.add_argument("--use_global_scaler", action="store_true", help="fit a single StandardScaler on all segments train data")
    # telemetry_col kept for backward-compat; no longer used when we keep all 55 dims
    parser.add_argument("--telemetry_col", type=int, default=0, help="(deprecated) kept for backward-compat; MSL uses all 55 features")

    parser.add_argument("--downsample_factor", type=int, default=10, help="median-pool downsample factor in preprocess (>1 enables)")
    parser.add_argument("--skip_head", type=int, default=0, help="drop first N rows for each channel segment after downsampling")

    parser.add_argument("--max_train_rows_per_channel", type=int, default=None, help="truncate train length per channel")
    parser.add_argument("--max_test_rows_per_channel", type=int, default=None, help="truncate test length per channel")

    parser.add_argument("--channels", type=str, nargs="*", default=None, help="subset of channel_ids to process")
    parser.add_argument("--max_channels", type=int, default=None, help="cap number of channels (after --channels)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_msl(
        raw_root=args.raw_root,
        out_data_dir=args.out_data_dir,
        out_dataset_dir=args.out_dataset_dir,
        use_global_scaler=args.use_global_scaler,
        telemetry_col=args.telemetry_col,
        output_format=args.output_format,
        downsample_factor=args.downsample_factor,
        skip_head=args.skip_head,
        max_train_rows_per_channel=args.max_train_rows_per_channel,
        max_test_rows_per_channel=args.max_test_rows_per_channel,
        channels=args.channels,
        max_channels=args.max_channels,
    )
