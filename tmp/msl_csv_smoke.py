"""Smoke test for MSL CSV pipeline.

Creates tiny synthetic MSL train/test CSVs under dataset/MSL/ and validates that
lib.dataloader_msl_smap.load_data_csv/load_data2_csv can build dataloaders.

Run:
  python tmp/msl_csv_smoke.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    out_dir = os.path.join(_REPO_ROOT, "dataset", "MSL")
    os.makedirs(out_dir, exist_ok=True)

    D = 55
    window_size = 15

    # two sequences
    seqs_train = {"C-1": 80, "C-2": 90}
    seqs_test = {"C-1": 70, "C-2": 60}

    rng = np.random.RandomState(0)
    feat_cols = [f"f{i}" for i in range(D)]

    # train
    train_rows = []
    for seq_id, T in seqs_train.items():
        x = rng.randn(T, D).astype(np.float32)
        df = pd.DataFrame(x, columns=feat_cols)
        df.insert(0, "t", np.arange(T, dtype=np.int64))
        df.insert(0, "seq_id", seq_id)
        train_rows.append(df)
    pd.concat(train_rows, ignore_index=True).to_csv(os.path.join(out_dir, "msl_train.csv"), index=False)

    # test
    test_rows = []
    for seq_id, T in seqs_test.items():
        x = rng.randn(T, D).astype(np.float32)
        y = np.zeros(T, dtype=int)
        # create an anomaly segment
        if T > 30:
            y[10:15] = 1
        df = pd.DataFrame(x, columns=feat_cols)
        df.insert(0, "t", np.arange(T, dtype=np.int64))
        df.insert(0, "seq_id", seq_id)
        df["attack"] = y
        test_rows.append(df)
    pd.concat(test_rows, ignore_index=True).to_csv(os.path.join(out_dir, "msl_test.csv"), index=False)

    with open(os.path.join(out_dir, "list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_cols) + "\n")

    from lib.dataloader_msl_smap import load_data_csv, load_data2_csv

    print("[No MAS]...")
    train_loader, val_loader, test_loader, y_test_labels, _ = load_data_csv(
        os.path.join(out_dir, "msl_train.csv"),
        os.path.join(out_dir, "msl_test.csv"),
        device="cpu",
        window_size=window_size,
        val_ratio=0.2,
        batch_size=8,
        is_down_sample=False,
        down_len=1,
    )
    print("batches:", len(train_loader), len(val_loader), len(test_loader), "labels:", len(y_test_labels), "sum:", sum(y_test_labels))

    print("[MAS]...")
    train_loader, val_loader, test_loader, y_test_labels, _ = load_data2_csv(
        os.path.join(out_dir, "msl_train.csv"),
        os.path.join(out_dir, "msl_test.csv"),
        device="cpu",
        window_size=window_size,
        val_ratio=0.2,
        batch_size=8,
        is_down_sample=False,
        down_len=1,
    )
    print("batches:", len(train_loader), len(val_loader), len(test_loader), "labels:", len(y_test_labels), "sum:", sum(y_test_labels))

    print("[OK] MSL CSV pipeline smoke test passed")


if __name__ == "__main__":
    main()

