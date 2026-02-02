# -*- coding: utf-8 -*-
"""scripts/process_smap.py

Telemanom-style SMAP preprocessing.

SMAP and MSL are from the same NASA Telemanom dataset family, and share the same
raw dataset layout and metadata format.

Expected raw layout under --raw_root:
- labeled_anomalies.csv
- train/<channel_id>.npy
- test/<channel_id>.npy

Outputs:
- PKL mode (repo-compatible, used by lib/dataloader_msl_smap.py):
  - data/SMAP/SMAP_train.pkl: (T_train, N)
  - data/SMAP/SMAP_test.pkl:  (T_test, N+1)  last col is label

- CSV mode (SWaT/WADI-style under dataset/SMAP):
  - dataset/SMAP/smap_train.csv: features only
  - dataset/SMAP/smap_test.csv: features + attack label column
  - dataset/SMAP/list.txt: node names (channel_id)
"""

import os
import ast
import argparse
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_meta(raw_root: str) -> pd.DataFrame:
    """Read labeled_anomalies.csv and keep spacecraft == 'SMAP'."""
    csv_path = os.path.join(raw_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labeled_anomalies.csv: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Telemanom usually uses channel_id; some forks use chan_id in docs.
    if "channel_id" not in df.columns and "chan_id" in df.columns:
        df = df.rename(columns={"chan_id": "channel_id"})

    required = {"channel_id", "spacecraft", "anomaly_sequences"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"labeled_anomalies.csv missing columns: {required}, got: {list(df.columns)}")

    df_smap = df[df["spacecraft"].astype(str).str.upper() == "SMAP"].copy()
    if df_smap.empty:
        raise ValueError("No rows with spacecraft == SMAP in labeled_anomalies.csv")

    return df_smap


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


def preprocess_smap(
    raw_root: str,
    out_data_dir: str,
    out_dataset_dir: str,
    use_global_scaler: bool = False,
    telemetry_col: int = 0,
    output_format: str = "pkl",
):
    """Preprocess SMAP from Telemanom raw format into PKL/CSV used by this repo.

    NOTE:
    - SMAP 和 MSL 同源（Telemanom/NASA），每个 channel 的 npy 通常是 (T, 25) 或 (T, 55)。
    - 旧实现为了做多变量图结构，把每个 channel 压成 1 维（telemetry_col），再堆成 (T, #channels)。
      这会导致 label union 口径与论文表（55*25）不一致，也容易出现异常率被放大。

    现在的实现与 MSL 保持一致：保留每个 channel 的全部输入维度，并沿时间拼接所有 channel。
    输出：
      - x_train: (sum T_train_i, n_inputs)
      - x_test:  (sum T_test_i,  n_inputs)
      - y_test:  (sum T_test_i,)  每个样本点的异常标签
    """

    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_dataset_dir, exist_ok=True)

    fmt = str(output_format).lower()
    if fmt not in {"pkl", "csv", "both"}:
        raise ValueError("output_format must be one of: pkl, csv, both")

    train_dir = os.path.join(raw_root, "train")
    test_dir = os.path.join(raw_root, "test")

    meta = load_meta(raw_root)
    chan_ids = sorted(meta["channel_id"].astype(str).tolist())

    # Keep only channels that exist in both train/test
    chan_ids = [
        cid
        for cid in chan_ids
        if os.path.exists(os.path.join(train_dir, f"{cid}.npy"))
        and os.path.exists(os.path.join(test_dir, f"{cid}.npy"))
    ]
    if not chan_ids:
        raise FileNotFoundError("No channels found with both train/<cid>.npy and test/<cid>.npy")

    print(f"[SMAP] channels found: {len(chan_ids)}")

    # Index meta by channel id to locate anomaly_sequences
    meta = meta.set_index("channel_id")

    # Optional global scaler fitted on *all channels train data* (all inputs)
    global_scaler: Optional[StandardScaler] = None
    if use_global_scaler:
        print("[SMAP] fitting global StandardScaler on all channels train data...")
        all_train = []
        for cid in chan_ids:
            train_raw = _load_channel_array(os.path.join(train_dir, f"{cid}.npy"))
            all_train.append(train_raw)
        all_train_cat = np.concatenate(all_train, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(all_train_cat)

    # New: keep FULL input dims like MSL, and concatenate along time
    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    test_label_blocks: List[np.ndarray] = []

    n_inputs: Optional[int] = None

    for cid in chan_ids:
        train_raw = _load_channel_array(os.path.join(train_dir, f"{cid}.npy"))
        test_raw = _load_channel_array(os.path.join(test_dir, f"{cid}.npy"))

        if n_inputs is None:
            n_inputs = int(train_raw.shape[1])
        elif int(train_raw.shape[1]) != int(n_inputs):
            raise ValueError(f"Input dim mismatch for {cid}: {train_raw.shape[1]} vs {n_inputs}")

        if use_global_scaler:
            scaler = global_scaler
        else:
            scaler = StandardScaler()
            scaler.fit(train_raw)

        train_norm = scaler.transform(train_raw).astype(np.float32)
        test_norm = scaler.transform(test_raw).astype(np.float32)

        # per-timestep label for this channel on test
        T = test_norm.shape[0]
        y = np.zeros((T,), dtype=np.int64)
        if cid in meta.index:
            intervals = parse_anomaly_sequences(meta.loc[cid, "anomaly_sequences"])
            for s, e in intervals:
                s = max(0, int(s))
                e = min(T - 1, int(e))
                if s <= e:
                    y[s : e + 1] = 1

        # Align (just in case)
        if test_norm.shape[0] != y.shape[0]:
            m = min(test_norm.shape[0], y.shape[0])
            test_norm = test_norm[:m]
            y = y[:m]

        train_blocks.append(train_norm)
        test_blocks.append(test_norm)
        test_label_blocks.append(y)

        print(f"[SMAP:{cid}] train={train_norm.shape} test={test_norm.shape} label={y.shape}")

    if n_inputs is None:
        raise RuntimeError("Failed to infer input dim")

    x_train = np.concatenate(train_blocks, axis=0).astype(np.float32)
    x_test = np.concatenate(test_blocks, axis=0).astype(np.float32)
    y_test = np.concatenate(test_label_blocks, axis=0).astype(np.int64)

    print(f"[SMAP] concatenated lengths: train={x_train.shape[0]}, test={x_test.shape[0]}, features={x_train.shape[1]}")

    # list.txt in dataset/SMAP for segment names (channel_id)
    list_path = os.path.join(out_dataset_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for cid in chan_ids:
            f.write(str(cid) + "\n")

    # PKL outputs
    if fmt in {"pkl", "both"}:
        train_pkl = os.path.join(out_data_dir, "SMAP_train.pkl")
        test_pkl = os.path.join(out_data_dir, "SMAP_test.pkl")

        test_with_label = np.concatenate([x_test, y_test.reshape(-1, 1).astype(np.float32)], axis=1)

        with open(train_pkl, "wb") as f:
            pickle.dump(x_train, f)
        with open(test_pkl, "wb") as f:
            pickle.dump(test_with_label, f)

        print("[OK] wrote PKL:")
        print(" ", train_pkl, x_train.shape)
        print(" ", test_pkl, test_with_label.shape, "(last col=label)")

    # CSV outputs
    if fmt in {"csv", "both"}:
        feat_cols = [f"f{i}" for i in range(x_train.shape[1])]
        train_df = pd.DataFrame(x_train, columns=feat_cols)
        test_df = pd.DataFrame(x_test, columns=feat_cols)
        test_df["attack"] = y_test.astype(int)

        train_csv = os.path.join(out_dataset_dir, "smap_train.csv")
        test_csv = os.path.join(out_dataset_dir, "smap_test.csv")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print("[OK] wrote CSV:")
        print(" ", train_csv, train_df.shape)
        print(" ", test_csv, test_df.shape)

    print("[OK] wrote list:")
    print(" ", list_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SMAP (Telemanom-style) into PKL and/or CSV files")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=os.path.join("data", "SMAP"),
        help="raw SMAP root containing train/ test/ labeled_anomalies.csv",
    )
    parser.add_argument(
        "--out_data_dir",
        type=str,
        default=os.path.join("data", "SMAP"),
        help="output dir for PKL files (default: data/SMAP)",
    )
    parser.add_argument(
        "--out_dataset_dir",
        type=str,
        default=os.path.join("dataset", "SMAP"),
        help="output dir for CSV/list.txt (default: dataset/SMAP)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="both",
        choices=["pkl", "csv", "both"],
        help="output format (default: pkl)",
    )
    parser.add_argument("--use_global_scaler", action="store_true", help="fit a single StandardScaler on all channels train data")
    parser.add_argument(
        "--telemetry_col",
        type=int,
        default=0,
        help="which column of each channel npy to use as the node value (default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_smap(
        raw_root=args.raw_root,
        out_data_dir=args.out_data_dir,
        out_dataset_dir=args.out_dataset_dir,
        use_global_scaler=args.use_global_scaler,
        telemetry_col=args.telemetry_col,
        output_format=args.output_format,
    )
