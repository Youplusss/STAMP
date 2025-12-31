"""Smoke test for MSL PKL pipeline.

Creates a tiny synthetic Telemanom-style MSL folder structure under tmp/_msl_raw/
(train/test per-channel .npy + labeled_anomalies.csv), then runs scripts/process_msl.py
API to generate data/MSL/MSL_train.pkl and data/MSL/MSL_test.pkl (with labels in last col)
plus dataset/MSL/list.txt.

Run:
  python tmp/msl_pkl_smoke.py
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

    # two channels, each npy is (T, n_inputs)
    rng = np.random.RandomState(0)
    channels = ["C-1", "C-2"]
    n_inputs = 4
    T_train = 50
    T_test = 40

    for cid in channels:
        train_arr = rng.randn(T_train, n_inputs).astype(np.float32)
        test_arr = rng.randn(T_test, n_inputs).astype(np.float32)
        np.save(os.path.join(raw_root, "train", f"{cid}.npy"), train_arr)
        np.save(os.path.join(raw_root, "test", f"{cid}.npy"), test_arr)

    # labeled anomalies: only C-1 has anomalies in [5,7] and [20,21]
    df = pd.DataFrame(
        {
            "channel_id": channels,
            "spacecraft": ["MSL", "MSL"],
            "anomaly_sequences": ["[[5,7],[20,21]]", "[]"],
        }
    )
    df.to_csv(os.path.join(raw_root, "labeled_anomalies.csv"), index=False)

    from scripts.process_msl import preprocess_msl

    preprocess_msl(
        raw_root=raw_root,
        out_data_dir=out_data,
        out_dataset_dir=out_dataset,
        use_global_scaler=False,
        telemetry_col=0,
    )

    # verify files exist
    import pickle

    with open(os.path.join(out_data, "MSL_train.pkl"), "rb") as f:
        x_train = pickle.load(f)
    with open(os.path.join(out_data, "MSL_test.pkl"), "rb") as f:
        test_with_label = pickle.load(f)

    assert x_train.shape == (T_train, len(channels))
    assert test_with_label.shape == (T_test, len(channels) + 1)

    x_test = test_with_label[:, :len(channels)]
    y_test = test_with_label[:, len(channels)]

    assert x_test.shape == (T_test, len(channels))
    assert y_test.shape == (T_test,)
    assert int(np.sum(y_test)) > 0

    print("[OK] MSL PKL smoke test passed")


if __name__ == "__main__":
    main()

