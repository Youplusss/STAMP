import argparse
    main()
if __name__ == "__main__":


        print(f"[OK] {args.group_name}: {out_train}, {out_test}, {out_label}")
        out_train, out_test, out_label = process_one_group(args.group_name, args.input_dir, args.output_dir, x_dim=args.x_dim)
            raise ValueError("Either provide --group_name or use --all_groups")
        if not args.group_name:
    else:
            print(f"[OK] {g}: {out_train}, {out_test}, {out_label}")
            out_train, out_test, out_label = process_one_group(g, in_dir, args.output_dir, x_dim=args.x_dim)
            in_dir = str(root / g)
        for g in sorted(groups):
            raise ValueError("--all_groups set but no subdirectories found")
        if not groups:
        groups = [p.name for p in root.iterdir() if p.is_dir()]
        root = Path(args.input_dir)
    if args.all_groups:

    args = parser.parse_args()
    parser.add_argument("--x_dim", type=int, default=38)
    parser.add_argument("--all_groups", action="store_true", help="Process all subfolders in input_dir as groups")
    parser.add_argument("--group_name", type=str, default=None, help="machine-1-1 etc")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "SMD", "generalization"))
    parser.add_argument("--input_dir", type=str, required=True, help="Raw SMD root (either a group folder or a root containing group subfolders)")
    parser = argparse.ArgumentParser(description="Preprocess SMD into STAMP expected data/SMD/generalization/*.pkl")
def main():


    return out_train, out_test, out_label

    _save_pkl(out_label, y_test)
    _save_pkl(out_test, x_test)
    _save_pkl(out_train, x_train)

    out_label = str(out_dir / f"{group_name}_test_label.pkl")
    out_test = str(out_dir / f"{group_name}_test.pkl")
    out_train = str(out_dir / f"{group_name}_train.pkl")

        raise ValueError(f"Label length mismatch: len(y_test)={len(y_test)} vs len(x_test)={len(x_test)}")
    if len(y_test) != len(x_test):

    y_test = _load_label(label_path)
    x_test = _coerce_2d(_load_array(test_path), x_dim)
    x_train = _coerce_2d(_load_array(train_path), x_dim)

    label_path = _pick(label_candidates)
    test_path = _pick(test_candidates)
    train_path = _pick(train_candidates)

        raise FileNotFoundError(f"None of the candidates exist: {[str(c) for c in cands]}")
                return str(c)
            if c.exists():
        for c in cands:
    def _pick(cands):

    ]
        in_dir / "label.csv",
        in_dir / "label.pkl",
        in_dir / "label.npy",
        in_dir / "test_label.csv",
        in_dir / "test_label.pkl",
        in_dir / "test_label.npy",
        in_dir / f"{group_name}_test_label.csv",
        in_dir / f"{group_name}_test_label.pkl",
        in_dir / f"{group_name}_test_label.npy",
    label_candidates = [
    ]
        in_dir / "test.csv",
        in_dir / "test.pkl",
        in_dir / "test.npy",
        in_dir / f"{group_name}_test.csv",
        in_dir / f"{group_name}_test.pkl",
        in_dir / f"{group_name}_test.npy",
    test_candidates = [
    ]
        in_dir / "train.csv",
        in_dir / "train.pkl",
        in_dir / "train.npy",
        in_dir / f"{group_name}_train.csv",
        in_dir / f"{group_name}_train.pkl",
        in_dir / f"{group_name}_train.npy",
    train_candidates = [
    # default raw filenames (common conventions)

    out_dir = Path(output_dir)
    in_dir = Path(input_dir)
    """Process one SMD group into the repo's expected pkl layout."""
def process_one_group(group_name: str, input_dir: str, output_dir: str, x_dim: int = 38) -> Tuple[str, str, str]:


        pickle.dump(arr, f)
    with open(path, "wb") as f:
    os.makedirs(os.path.dirname(path), exist_ok=True)
def _save_pkl(path: str, arr: np.ndarray) -> None:


    return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # replace NaNs/Infs
        raise ValueError(f"Expected feature dim={x_dim}, got {x.shape[1]} (shape={x.shape})")
    if x.shape[1] != x_dim:
        x = x.reshape((-1, x.shape[-1]))
    if x.ndim > 2:
        raise ValueError("SMD data must be 2D (T, 38)")
    if x.ndim == 1:
    x = np.asarray(x, dtype=np.float32)
def _coerce_2d(x: np.ndarray, x_dim: int) -> np.ndarray:


    return y.astype(np.float32)
        y = (y == -1).astype(np.float32)
    if np.isin(y, [-1, 1]).all() and not np.isin(y, [0, 1]).all():
    # if labels are -1/1, map -1 -> 1 anomaly
    y = np.asarray(y).reshape(-1)
    y = _load_array(path)
def _load_label(path: str) -> np.ndarray:


    return arr
    arr = np.asarray(arr)

        raise ValueError(f"Unsupported file type: {p}")
    else:
            arr = np.loadtxt(p, dtype=np.float32)
        except Exception:
            arr = np.loadtxt(p, delimiter=",", dtype=np.float32)
        try:
        # try comma first, then whitespace
    elif suf == ".csv" or suf == ".txt":
            arr = pickle.load(f)
        with open(p, "rb") as f:
    elif suf == ".pkl":
            raise KeyError(f"npz has keys {list(z.keys())}, expected one of data/x/arr_0/a")
        else:
                break
                arr = z[k]
            if k in z:
        for k in ["data", "x", "arr_0", "a"]:
        z = np.load(p, allow_pickle=True)
    elif suf == ".npz":
        arr = np.load(p)
    if suf == ".npy":
    suf = p.suffix.lower()

        raise FileNotFoundError(path)
    if not p.exists():
    p = Path(path)
    """
    - .pkl: pickle containing a numpy array
    - .csv: numeric values only (delimiter auto)
    - .npz: must contain key 'data' or 'x' or 'arr_0'
    - .npy: numpy array

    """Load a numeric array from .npy/.npz/.csv/.pkl.
def _load_array(path: str) -> np.ndarray:


import numpy as np

from typing import Optional, Tuple
from pathlib import Path
import pickle
import os

