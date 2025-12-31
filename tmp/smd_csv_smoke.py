"""Smoke test for SMD CSV pipeline.

Creates a tiny synthetic SMD CSV pair under dataset/SMD/ and ensures
lib.dataloader_smd.load_data_csv/load_data2_csv can build dataloaders.

Run:
  python tmp/smd_csv_smoke.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    out_dir = os.path.join(os.getcwd(), "dataset", "SMD")
    _ensure_dir(out_dir)

    # synthetic: 200 timesteps, 38 features
    T_train = 200
    T_test = 120
    D = 38

    rng = np.random.RandomState(0)
    train = rng.randn(T_train, D).astype(np.float32)
    test = rng.randn(T_test, D).astype(np.float32)

    # simple anomaly segments
    labels = np.zeros(T_test, dtype=int)
    labels[30:40] = 1
    labels[90:100] = 1

    cols = [f"f{i}" for i in range(D)]
    pd.DataFrame(train, columns=cols).to_csv(os.path.join(out_dir, "smd_train.csv"), index=False)
    df_test = pd.DataFrame(test, columns=cols)
    df_test["attack"] = labels
    df_test.to_csv(os.path.join(out_dir, "smd_test.csv"), index=False)

    with open(os.path.join(out_dir, "list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cols) + "\n")

    from lib.dataloader_smd import load_data_csv, load_data2_csv

    device = "cpu"
    print("\n[No MAS] building dataloaders...")
    train_loader, val_loader, test_loader, y_test_labels, _ = load_data_csv(
        os.path.join(out_dir, "smd_train.csv"),
        os.path.join(out_dir, "smd_test.csv"),
        device=device,
        window_size=15,
        val_ratio=0.2,
        batch_size=16,
        is_down_sample=False,
        down_len=1,
    )
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))
    print("y_test_labels:", len(y_test_labels), "sum:", sum(y_test_labels))

    print("\n[MAS] building dataloaders...")
    train_loader, val_loader, test_loader, y_test_labels, _ = load_data2_csv(
        os.path.join(out_dir, "smd_train.csv"),
        os.path.join(out_dir, "smd_test.csv"),
        device=device,
        window_size=15,
        val_ratio=0.2,
        batch_size=16,
        is_down_sample=False,
        down_len=1,
    )
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))
    print("y_test_labels:", len(y_test_labels), "sum:", sum(y_test_labels))

    print("\n[OK] SMD CSV pipeline smoke test passed")


if __name__ == "__main__":
    main()

