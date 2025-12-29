import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p)
    elif suf == ".npz":
        z = np.load(p, allow_pickle=True)
        for k in ["data", "x", "arr_0"]:
            if k in z:
                arr = z[k]
                break
        else:
            raise KeyError(f"npz has keys {list(z.keys())}, expected one of data/x/arr_0")
    elif suf == ".pkl":
        with open(p, "rb") as f:
            arr = pickle.load(f)
    elif suf == ".csv" or suf == ".txt":
        try:
            arr = np.loadtxt(p, delimiter=",", dtype=np.float32)
        except Exception:
            arr = np.loadtxt(p, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file type: {p}")

    return np.asarray(arr)


def _load_label(path: str) -> np.ndarray:
    y = _load_array(path)
    y = np.asarray(y).reshape(-1)
    if np.isin(y, [-1, 1]).all() and not np.isin(y, [0, 1]).all():
        y = np.asarray(y == -1).astype(np.float32)
    return y.astype(np.float32)


def _coerce_2d(x: np.ndarray, x_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        raise ValueError("MSL data must be 2D (T, 55)")
    if x.ndim > 2:
        x = x.reshape((-1, x.shape[-1]))
    if x.shape[1] != x_dim:
        raise ValueError(f"Expected feature dim={x_dim}, got {x.shape[1]} (shape={x.shape})")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _save_pkl(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(arr, f)


def process_msl(input_dir: str, output_dir: str, x_dim: int = 55) -> Tuple[str, str, str]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    train_candidates = [in_dir / "MSL_train.npy", in_dir / "train.npy", in_dir / "train.pkl", in_dir / "train.csv"]
    test_candidates = [in_dir / "MSL_test.npy", in_dir / "test.npy", in_dir / "test.pkl", in_dir / "test.csv"]
    label_candidates = [
        in_dir / "MSL_test_label.npy",
        in_dir / "test_label.npy",
        in_dir / "test_label.pkl",
        in_dir / "test_label.csv",
        in_dir / "label.npy",
        in_dir / "label.pkl",
        in_dir / "label.csv",
    ]

    def _pick(cands):
        for c in cands:
            if c.exists():
                return str(c)
        raise FileNotFoundError(f"None of the candidates exist: {[str(c) for c in cands]}")

    train_path = _pick(train_candidates)
    test_path = _pick(test_candidates)
    label_path = _pick(label_candidates)

    x_train = _coerce_2d(_load_array(train_path), x_dim)
    x_test = _coerce_2d(_load_array(test_path), x_dim)
    y_test = _load_label(label_path)

    if len(y_test) != len(x_test):
        raise ValueError(f"Label length mismatch: len(y_test)={len(y_test)} vs len(x_test)={len(x_test)}")

    out_train = str(out_dir / "MSL_train.pkl")
    out_test = str(out_dir / "MSL_test.pkl")
    out_label = str(out_dir / "MSL_test_label.pkl")

    _save_pkl(out_train, x_train)
    _save_pkl(out_test, x_test)
    _save_pkl(out_label, y_test)

    return out_train, out_test, out_label


def main():
    parser = argparse.ArgumentParser(description="Preprocess MSL into STAMP expected data/MSL/*.pkl")
    parser.add_argument("--input_dir", type=str, required=True, help="Raw MSL directory containing train/test/label arrays")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "MSL"))
    parser.add_argument("--x_dim", type=int, default=55)
    args = parser.parse_args()

    out_train, out_test, out_label = process_msl(args.input_dir, args.output_dir, x_dim=args.x_dim)
    print(f"[OK] MSL: {out_train}, {out_test}, {out_label}")


if __name__ == "__main__":
    main()
