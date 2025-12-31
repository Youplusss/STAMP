"""Smoke test for MSL CSV output from scripts/process_msl.py.

Ensures the generated msl_train.csv contains only numeric feature columns (no seq_id/t),
and msl_test.csv contains an 'attack' column.

Run:
  python tmp/msl_csv_smoke2.py
"""

from __future__ import annotations

import os
import sys
import shutil

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    raw_root = os.path.join(_REPO_ROOT, "tmp", "_msl_raw")
    out_data = os.path.join(_REPO_ROOT, "tmp", "_msl_out_data")
    out_dataset = os.path.join(_REPO_ROOT, "tmp", "_msl_out_dataset")

    shutil.rmtree(raw_root, ignore_errors=True)
    shutil.rmtree(out_data, ignore_errors=True)
    shutil.rmtree(out_dataset, ignore_errors=True)

    os.makedirs(os.path.join(raw_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(raw_root, "test"), exist_ok=True)

    rng = np.random.RandomState(0)
    channels = ["C-1", "C-2", "C-3"]
    n_inputs = 4
    T_train = 30
    T_test = 25

    for cid in channels:
        np.save(os.path.join(raw_root, "train", f"{cid}.npy"), rng.randn(T_train, n_inputs).astype(np.float32))
        np.save(os.path.join(raw_root, "test", f"{cid}.npy"), rng.randn(T_test, n_inputs).astype(np.float32))

    df = pd.DataFrame(
        {
            "channel_id": channels,
            "spacecraft": ["MSL"] * len(channels),
            "anomaly_sequences": ["[]", "[[1,2]]", "[]"],
        }
    )
    df.to_csv(os.path.join(raw_root, "labeled_anomalies.csv"), index=False)

    from scripts.process_msl import preprocess_msl

    preprocess_msl(
        raw_root=raw_root,
        out_data_dir=out_data,
        out_dataset_dir=out_dataset,
        output_format="csv",
        telemetry_col=0,
    )

    train_csv = os.path.join(out_dataset, "msl_train.csv")
    test_csv = os.path.join(out_dataset, "msl_test.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    assert "attack" not in train_df.columns
    assert "attack" in test_df.columns
    assert "seq_id" not in train_df.columns
    assert "t" not in train_df.columns

    # all feature columns should be numeric
    assert train_df.shape == (T_train, len(channels))
    assert test_df.shape == (T_test, len(channels) + 1)

    print("[OK] MSL CSV output smoke test passed")


if __name__ == "__main__":
    main()

