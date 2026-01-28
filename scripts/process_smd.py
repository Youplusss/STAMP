# tools/preprocess_smd.py
# -*- coding: utf-8 -*-
"""scripts/process_smd.py

SMD (Server Machine Dataset) preprocessing.

Goals in this repo:
- Align with SWaT/WADI outputs: write CSVs under dataset/SMD
  - smd_train.csv: features only (+ optional attack=0 column)
  - smd_test.csv:  features + attack label column
  - list.txt: feature names (one per line)
- Align with MSL/SMAP philosophy: treat each machine as one sequence, scale using
  training statistics, then concatenate all machines.
- Add an explicit downsampling option (median pool) like SWaT/WADI scripts.

Expected raw layout (matches the official SMD release):
  <raw_root>/train/*.txt
  <raw_root>/test/*.txt
  <raw_root>/test_label/*.txt

Each *.txt is comma-separated numeric values.

Downsampling:
- Features: median over each window of length downsample_factor
- Labels: max over each window (if any anomaly in the window -> 1)

Example:
  python scripts/process_smd.py --raw_root ./data/SMD --out_root ./dataset/SMD --downsample_factor 10

Notes:
- If you want to mimic SWaT/WADI behavior of dropping initial transients, use --skip_head.
"""

import os
import argparse
from typing import List, Optional, Tuple

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


def _list_machine_ids(raw_root: str) -> List[str]:
    train_dir = os.path.join(raw_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    machine_ids = []
    for fn in os.listdir(train_dir):
        if fn.lower().endswith(".txt"):
            machine_ids.append(os.path.splitext(fn)[0])
    machine_ids.sort()
    return machine_ids


def _load_machine(raw_root: str, machine_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(raw_root, "train", f"{machine_id}.txt")
    test_path = os.path.join(raw_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(raw_root, "test_label", f"{machine_id}.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    train = np.loadtxt(train_path, delimiter=",")
    test = np.loadtxt(test_path, delimiter=",")
    labels = np.loadtxt(label_path, delimiter=",")

    # Ensure 2D features, 1D labels
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    return train, test, labels


def preprocess_smd(
    raw_root: str,
    out_root: str,
    *,
    use_global_scaler: bool = False,
    downsample_factor: int = 1,
    skip_head: int = 0,
    max_train_rows_per_machine: Optional[int] = None,
    max_test_rows_per_machine: Optional[int] = None,
    machines: Optional[List[str]] = None,
    max_machines: Optional[int] = None,
    include_attack_in_train: bool = True,
    scale_mode: str = "none",
):
    os.makedirs(out_root, exist_ok=True)

    all_machine_ids = _list_machine_ids(raw_root)

    # filter
    if machines:
        machines = [m.strip() for m in machines if str(m).strip()]
        missing = [m for m in machines if m not in all_machine_ids]
        if missing:
            raise ValueError(f"Unknown machine ids: {missing}. Available: {all_machine_ids}")
        machine_ids = machines
    else:
        machine_ids = list(all_machine_ids)

    if max_machines is not None:
        machine_ids = machine_ids[: int(max_machines)]

    if not machine_ids:
        raise RuntimeError("No machines selected.")

    print(f"[SMD] found machines={len(all_machine_ids)}")
    print(f"[SMD] processing machines={len(machine_ids)}: {machine_ids[:5]}{'...' if len(machine_ids) > 5 else ''}")

    scale_mode = str(scale_mode).lower().strip()
    if scale_mode not in {"none", "standard", "global_standard"}:
        raise ValueError("scale_mode must be one of: none, standard, global_standard")

    # Back-compat: if --use_global_scaler is set, treat it like global_standard
    if use_global_scaler:
        scale_mode = "global_standard"

    # optional global scaler
    global_scaler: Optional[StandardScaler] = None
    if scale_mode == "global_standard":
        print("[SMD] fitting global StandardScaler on ALL machines' train data...")
        trains = []
        for mid in machine_ids:
            tr, _, _ = _load_machine(raw_root, mid)
            trains.append(tr)
        concat = np.concatenate(trains, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(concat)

    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    label_blocks: List[np.ndarray] = []

    feature_dim: Optional[int] = None

    for mid in machine_ids:
        tr_raw, te_raw, y_raw = _load_machine(raw_root, mid)

        if feature_dim is None:
            feature_dim = int(tr_raw.shape[1])
        elif int(tr_raw.shape[1]) != int(feature_dim):
            raise ValueError(f"Feature dim mismatch for {mid}: {tr_raw.shape[1]} vs {feature_dim}")

        # scale per-machine (default) or globally; or disable scaling to match SWaT/WADI loaders
        if scale_mode == "none":
            tr = tr_raw.astype(np.float32)
            te = te_raw.astype(np.float32)
        else:
            if scale_mode == "global_standard":
                scaler = global_scaler
            else:
                scaler = StandardScaler()
                scaler.fit(tr_raw)

            tr = scaler.transform(tr_raw).astype(np.float32)
            te = scaler.transform(te_raw).astype(np.float32)

        y = y_raw.astype(np.int64)

        # downsample like SWaT/WADI
        if downsample_factor and downsample_factor > 1:
            tr = _downsample_features_median(tr, int(downsample_factor))
            te = _downsample_features_median(te, int(downsample_factor))
            y = _downsample_labels_max(y, int(downsample_factor))

        # optional skip_head
        if skip_head and skip_head > 0:
            tr = _skip_head(tr, int(skip_head))
            te = _skip_head(te, int(skip_head))
            y = _skip_head(y, int(skip_head))

        # optional truncate
        tr = _truncate(tr, max_train_rows_per_machine)
        te = _truncate(te, max_test_rows_per_machine)
        y = _truncate(y, max_test_rows_per_machine)

        # align test/label length
        if te.shape[0] != y.shape[0]:
            m = min(te.shape[0], y.shape[0])
            te = te[:m]
            y = y[:m]

        train_blocks.append(tr)
        test_blocks.append(te)
        label_blocks.append(y)

        print(f"[SMD:{mid}] train={tr.shape} test={te.shape} label={y.shape}")

    if feature_dim is None:
        raise RuntimeError("Failed to detect feature_dim")

    train_all = np.concatenate(train_blocks, axis=0) if train_blocks else np.empty((0, feature_dim), dtype=np.float32)
    test_all = np.concatenate(test_blocks, axis=0) if test_blocks else np.empty((0, feature_dim), dtype=np.float32)
    labels_all = np.concatenate(label_blocks, axis=0) if label_blocks else np.empty((0,), dtype=np.int64)

    # write CSVs
    feat_cols = [f"f{i}" for i in range(int(feature_dim))]

    train_df = pd.DataFrame(train_all, columns=feat_cols)
    if include_attack_in_train:
        train_df["attack"] = 0

    test_df = pd.DataFrame(test_all, columns=feat_cols)
    test_df["attack"] = labels_all.astype(int)

    train_csv = os.path.join(out_root, "smd_train.csv")
    test_csv = os.path.join(out_root, "smd_test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # list.txt
    list_txt = os.path.join(out_root, "list.txt")
    with open(list_txt, "w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(c + "\n")

    print("[OK] wrote:")
    print(" ", train_csv, train_df.shape)
    print(" ", test_csv, test_df.shape)
    print(" ", list_txt)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess SMD into CSVs under dataset/SMD")
    p.add_argument("--raw_root", type=str, default=os.path.join("data", "SMD"), help="raw SMD root (train/test/test_label)")
    p.add_argument("--out_root", type=str, default=os.path.join("dataset", "SMD"), help="output root (dataset/SMD)")

    # Match SWaT/WADI default behavior: do downsampling at training-time via --down_len.
    p.add_argument("--downsample_factor", type=int, default=1, help="median-pool downsample factor in preprocess (>1 enables)")
    p.add_argument("--skip_head", type=int, default=0, help="drop the first N rows for each machine (default: 0)")

    # Keep for backward compatibility; maps to scale_mode=global_standard.
    p.add_argument("--use_global_scaler", action="store_true", help="(deprecated) use one StandardScaler using all machines' train data")
    p.add_argument(
        "--scale_mode",
        type=str,
        default="none",
        choices=["none", "standard", "global_standard"],
        help="scaling in preprocess: none (recommended; match SWaT/WADI loaders), standard (per-machine), global_standard",
    )

    p.add_argument("--max_train_rows_per_machine", type=int, default=None, help="truncate train length per machine")
    p.add_argument("--max_test_rows_per_machine", type=int, default=None, help="truncate test length per machine")

    p.add_argument("--machines", type=str, nargs="*", default=None, help="subset of machines to process")
    p.add_argument("--max_machines", type=int, default=None, help="cap number of machines (after --machines)")

    p.add_argument(
        "--no_attack_in_train",
        action="store_true",
        help="do not include the 'attack' column in train CSV (some pipelines expect features-only)",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_smd(
        raw_root=args.raw_root,
        out_root=args.out_root,
        use_global_scaler=args.use_global_scaler,
        downsample_factor=args.downsample_factor,
        skip_head=args.skip_head,
        max_train_rows_per_machine=args.max_train_rows_per_machine,
        max_test_rows_per_machine=args.max_test_rows_per_machine,
        machines=args.machines,
        max_machines=args.max_machines,
        include_attack_in_train=not args.no_attack_in_train,
        scale_mode=getattr(args, "scale_mode", "none"),
    )
