import numpy as np
import os
import pickle
from sklearn import preprocessing
import sys
from typing import Optional, Tuple, List
from lib.utils import *
#from utils import *
import pandas as pd
import torch

base_dir = os.getcwd()
prefix = os.path.join(base_dir, "data")
#prefix='/home/chenty/STAT-AD/data'

data_dim ={
    "SMD": 38,
    "SMAP": 25,
    "MSL": 55
}

def preprocess(df):
    """returns normalized and standardized data."""
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num(df)

    # normalize data
    df = preprocessing.MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def get_data(dataset, max_train_size=None, max_test_size=None, train_start=0, test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)

    x_dim = data_dim.get(dataset)

    # -------- train --------
    with open(os.path.join(prefix, dataset, dataset + '_train.pkl'), "rb") as f:
        train_loaded = pickle.load(f)

    train_loaded = np.asarray(train_loaded)

    # Most preprocess scripts store train as [T, N]. If it isn't 2D, attempt to reshape.
    if train_loaded.ndim == 2:
        train_data_2d = train_loaded
    else:
        if x_dim is None:
            raise ValueError(f"Unknown x_dim for dataset={dataset} and train.pkl is not 2D (shape={train_loaded.shape})")
        try:
            train_data_2d = train_loaded.reshape((-1, x_dim))
        except Exception:
            # Fallback: infer x_dim from total size if possible
            size = int(train_loaded.size)
            if size % int(x_dim) != 0:
                raise
            train_data_2d = train_loaded.reshape((-1, x_dim))

    # If x_dim was set but doesn't match actual columns, trust the file.
    if x_dim is None:
        x_dim = int(train_data_2d.shape[1])
    elif train_data_2d.shape[1] != int(x_dim):
        print(f"[Warn] data_dim[{dataset}]={x_dim} but train.pkl has {train_data_2d.shape[1]} columns; inferring x_dim from file")
        x_dim = int(train_data_2d.shape[1])

    train_data = train_data_2d[train_start:train_end, :]

    test_data = None

    test_label = None

    # -------- test --------
    try:
        with open(os.path.join(prefix, dataset, dataset + '_test.pkl'), "rb") as f:
            test_loaded = pickle.load(f)

        test_loaded = np.asarray(test_loaded)
        # If test.pkl already contains labels in the last column (shape: [T, x_dim+1])
        if test_loaded.ndim == 2 and test_loaded.shape[1] == x_dim + 1:
            test_data = test_loaded[:, :x_dim][test_start:test_end, :]
            test_label = test_loaded[:, x_dim][test_start:test_end]
        elif test_loaded.ndim == 2 and test_loaded.shape[1] == x_dim:
            test_data = test_loaded[test_start:test_end, :]
        else:
            test_data = test_loaded.reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None

    # 2) load explicit *_test_label.pkl if exists; override if present
    try:
        with open(os.path.join(prefix, dataset, dataset + "_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        # if missing and we didn't split from test.pkl, keep None
        pass

    return (train_data, None), (test_data, test_label)


def load_data(dataset, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    (normal, _), (attack, labels) = get_data(dataset, max_train_size=None, max_test_size=None, train_start=0,
                                                        test_start=0)

    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    # normal: (495000, 51)
    # attack: (449919, 51)
    # normal = normal[21600:,:]
    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

    # ## nomalization
    min = normal.min()##axis=0
    max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)
    # normal = min_max_scaler.transform(normal)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)

    attack = min_max_scaler.transform(attack)

    # windows_attack = min_max_scaler.transform(windows_attack)

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    ## train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    ## train
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_train).float().to(device))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_val).float().to(device))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_attack).float().to(device))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, min_max_scaler

def load_data2(dataset, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    (normal, _), (attack, labels) = get_data(dataset, max_train_size=None, max_test_size=None, train_start=0,
                                            test_start=0)
    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)
    

    ## Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    normal_mas = []
    attack_mas = []
    for w in window_sizes:
        normal_ma = np_ma(normal, w)
        normal_mas.append(normal_ma)

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)

    # normal: (495000, 45)
    # attack: (449919, 45)
    # normal = normal[21600:,:]
    # for i in range(len(window_sizes)):
    #     normal_mas[i] = normal_mas[i][21600:,:]

    W = np.max(window_sizes)
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    ## down sample
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

        for i in range(len(window_sizes)):
            normal_mas[i] = downsample(normal_mas[i], down_len=down_len, is_label=False)
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)

    ## nomalization
    min = normal.min()##axis=0
    max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)
    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    

    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    windows_normal_mas = np.stack(normal_mas, axis=-1)
    windows_attack_mas = np.stack(attack_mas, axis=-1)


    ## train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    windows_normal_mas_train = windows_normal_mas[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_mas_val = windows_normal_mas[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]


    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    print("windows_normal_train: ", windows_normal_train.shape)
    print("windows_normal_mas_train: ", windows_normal_mas_train.shape)
    print("windows_normal_val: ", windows_normal_val.shape)
    print("windows_normal_mas_val: ", windows_normal_mas_val.shape)
    print("windows_attack: ", windows_attack.shape)
    print("windows_attack_mas: ", windows_attack_mas.shape)

    ## train
    train_data_tensor = torch.from_numpy(windows_normal_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_normal_mas_train).float().to(device)

    train_dataset = MyDataset(train_data_tensor,train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data_tensor = torch.from_numpy(windows_normal_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_normal_mas_val).float().to(device)

    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data_tensor = torch.from_numpy(windows_attack).float().to(device)
    test_mas_data_tensor = torch.from_numpy(windows_attack_mas).float().to(device)

    test_dataset = MyDataset(test_data_tensor, test_mas_data_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, min_max_scaler


def _read_csv_robust(path: str, sep=None):
    """Read CSV with sane defaults across pandas versions.

    Notes:
      - We prefer engine='python' to avoid ParserWarning spam when sep is None.
      - Some pandas versions don't support low_memory with python engine, so we only pass
        low_memory when using the default C engine.
    """
    # Try fast path first (C engine). If sep=None triggers warnings/issues, fall back.
    try:
        # C engine path (supports low_memory)
        df = pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        # Python engine fallback (don't pass low_memory; not supported in some versions)
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    unnamed_like = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_like:
        df = df.drop(columns=unnamed_like)
    return df


def _extract_features_and_meta(df: pd.DataFrame):
    df = df.copy()
    # label column
    label_col = None
    for c in ["attack", "label", "Normal/Attack", "Attack"]:
        if c in df.columns:
            label_col = c
            break

    labels = None
    if label_col is not None:
        labels_raw = df[label_col]
        df = df.drop(columns=[label_col])
        labels_num = pd.to_numeric(labels_raw, errors="coerce").fillna(0).values
        labels = (labels_num == -1).astype(np.int64)
        if labels.max() == 0 and np.isin(labels_num, [0, 1]).all():
            labels = labels_num.astype(np.int64)

    # meta columns to keep for segmentation
    seq_id = df["seq_id"].astype(str).values if "seq_id" in df.columns else None
    if "seq_id" in df.columns:
        df = df.drop(columns=["seq_id"])

    # time index column (optional)
    if "t" in df.columns:
        df = df.drop(columns=["t"])

    # coerce features to numeric
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    x = df.values.astype(np.float32)
    return x, labels, seq_id


def _build_windows_by_seq(x: np.ndarray, labels: Optional[np.ndarray], seq_id: Optional[np.ndarray], window_size: int):
    """Build windows per seq_id if provided; else treat as one sequence.

    Return:
      windows_x: (num_windows, window_size, D)
      windows_labels: list of per-window label arrays (length window_size)
    """
    if seq_id is None:
        if x.shape[0] < window_size:
            return np.empty((0, window_size, x.shape[1]), dtype=np.float32), []
        windows_x = x[np.arange(window_size)[None, :] + np.arange(x.shape[0] - window_size + 1)[:, None]]
        if labels is None:
            return windows_x, []
        win_labs = [labels[i:i + window_size] for i in range(len(labels) - window_size + 1)]
        return windows_x, win_labs

    windows_list = []
    labels_list = []

    # group contiguous rows by seq_id (CSV is produced in blocks per seq_id)
    start = 0
    n = len(seq_id)
    while start < n:
        cur = seq_id[start]
        end = start + 1
        while end < n and seq_id[end] == cur:
            end += 1

        seg_x = x[start:end]
        seg_labels = labels[start:end] if labels is not None else None

        if seg_x.shape[0] >= window_size:
            seg_windows = seg_x[np.arange(window_size)[None, :] + np.arange(seg_x.shape[0] - window_size + 1)[:, None]]
            windows_list.append(seg_windows)
            if seg_labels is not None:
                for i in range(len(seg_labels) - window_size + 1):
                    labels_list.append(seg_labels[i:i + window_size])

        start = end

    if not windows_list:
        return np.empty((0, window_size, x.shape[1]), dtype=np.float32), []

    return np.concatenate(windows_list, axis=0), labels_list


def _raise_empty_windows_error(*, name: str, n_rows: int, n_features: int, window_size: int, is_mas: bool, is_down_sample: bool, down_len: int, extra: str = ""):
    msg = (
        f"{name}: produced 0 windows (so DataLoader can't sample). "
        f"rows={n_rows}, features={n_features}, window_size={window_size}, "
        f"is_mas={is_mas}, is_down_sample={is_down_sample}, down_len={down_len}."
    )
    if extra:
        msg += "\n" + extra
    msg += (
        "\nCommon fixes: reduce --window_size; set --is_down_sample False (or --down_len 1); "
        "and for MSL with MAS ensure series length after MA offset (20) is still >= window_size."
    )
    raise ValueError(msg)


def load_data_csv(train_csv: str, test_csv: str, device="gpu", window_size=12, val_ratio=0.2, batch_size=64, is_down_sample=False, down_len=10):
    train_df = _read_csv_robust(train_csv)
    test_df = _read_csv_robust(test_csv)

    normal, _, train_seq = _extract_features_and_meta(train_df)
    attack, labels, test_seq = _extract_features_and_meta(test_df)

    if labels is None:
        raise KeyError(f"MSL test CSV must include label column 'attack'. columns={list(test_df.columns)}")

    # Downsample per full concatenation (OK because segmentation is preserved via seq_id)
    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)
        # note: seq_id dropped in downsample mode (hard to keep exact mapping). recommend down_len=1 for MSL.
        train_seq = None
        test_seq = None

    scaler = preprocessing.MinMaxScaler()
    normal = scaler.fit_transform(normal)
    attack = scaler.transform(attack)

    windows_normal, _ = _build_windows_by_seq(normal, None, train_seq, window_size)
    windows_attack, windows_labels = _build_windows_by_seq(attack, labels.astype(np.int64), test_seq, window_size)

    if windows_normal.shape[0] == 0:
        _raise_empty_windows_error(
            name="MSL train",
            n_rows=int(normal.shape[0]),
            n_features=int(normal.shape[1]),
            window_size=int(window_size),
            is_mas=False,
            is_down_sample=bool(is_down_sample),
            down_len=int(down_len),
            extra=f"train_csv={train_csv}",
        )

    if windows_attack.shape[0] == 0:
        _raise_empty_windows_error(
            name="MSL test",
            n_rows=int(attack.shape[0]),
            n_features=int(attack.shape[1]),
            window_size=int(window_size),
            is_mas=False,
            is_down_sample=bool(is_down_sample),
            down_len=int(down_len),
            extra=f"test_csv={test_csv}",
        )

    ## train/val/test
    windows_normal_train = windows_normal[:int(np.floor((1-val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1-val_ratio) * windows_normal.shape[0])):]

    ## reshape: [B, T, N ,C]
    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    ## train
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_train).float().to(device))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_normal_val).float().to(device))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(windows_attack).float().to(device))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    ## test labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    print("train: ", type(windows_normal_train.shape), windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, scaler


def load_data2_csv(train_csv: str, test_csv: str, device="gpu", window_size=12, val_ratio=0.2, batch_size=64, is_down_sample=False, down_len=10):
    train_df = _read_csv_robust(train_csv)
    test_df = _read_csv_robust(test_csv)

    normal, _, train_seq = _extract_features_and_meta(train_df)
    attack, labels, test_seq = _extract_features_and_meta(test_df)

    if labels is None:
        raise KeyError(f"MSL test CSV must include label column 'attack'. columns={list(test_df.columns)}")

    window_sizes = [3, 5, 10, 20]
    normal_mas = [np_ma(normal, w) for w in window_sizes]
    attack_mas = [np_ma(attack, w) for w in window_sizes]

    W = int(np.max(window_sizes))
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    if is_down_sample:
        normal = downsample(normal, down_len=down_len, is_label=False)
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)
        for i in range(len(window_sizes)):
            normal_mas[i] = downsample(normal_mas[i], down_len=down_len, is_label=False)
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)
        train_seq = None
        test_seq = None

    scaler = preprocessing.MinMaxScaler()
    normal = scaler.fit_transform(normal)
    attack = scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = scaler.transform(normal_mas[i])
        attack_mas[i] = scaler.transform(attack_mas[i])

    windows_normal, _ = _build_windows_by_seq(normal, None, train_seq, window_size)
    windows_attack, windows_labels = _build_windows_by_seq(attack, labels.astype(np.int64), test_seq, window_size)

    if windows_normal.shape[0] == 0:
        _raise_empty_windows_error(
            name="MSL train (MAS)",
            n_rows=int(normal.shape[0]),
            n_features=int(normal.shape[1]),
            window_size=int(window_size),
            is_mas=True,
            is_down_sample=bool(is_down_sample),
            down_len=int(down_len),
            extra=f"train_csv={train_csv}",
        )

    if windows_attack.shape[0] == 0:
        _raise_empty_windows_error(
            name="MSL test (MAS)",
            n_rows=int(attack.shape[0]),
            n_features=int(attack.shape[1]),
            window_size=int(window_size),
            is_mas=True,
            is_down_sample=bool(is_down_sample),
            down_len=int(down_len),
            extra=(
                f"test_csv={test_csv}\n"
                f"Note: MAS mode drops the first {W} rows before windowing; ensure (T_test-{W}) >= window_size."
            ),
        )

    # MAS windows: if seq_id present we currently treat as single sequence to stay consistent
    # (keeping exact seq_id boundaries across MA offsets is tricky; recommend down_len=1 and accept this.)
    windows_normal_mas = []
    windows_attack_mas = []
    for i in range(len(window_sizes)):
        wn, _ = _build_windows_by_seq(normal_mas[i], None, train_seq, window_size)
        wa, _ = _build_windows_by_seq(attack_mas[i], None, test_seq, window_size)
        windows_normal_mas.append(wn)
        windows_attack_mas.append(wa)

    windows_normal_mas = np.stack(windows_normal_mas, axis=-1) if windows_normal_mas else None
    windows_attack_mas = np.stack(windows_attack_mas, axis=-1) if windows_attack_mas else None

    windows_normal_train = windows_normal[:int(np.floor((1 - val_ratio) * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor((1 - val_ratio) * windows_normal.shape[0])):]

    windows_normal_mas_train = windows_normal_mas[:int(np.floor((1 - val_ratio) * windows_normal.shape[0]))]
    windows_normal_mas_val = windows_normal_mas[int(np.floor((1 - val_ratio) * windows_normal.shape[0])):]

    windows_normal_train = windows_normal_train.reshape(windows_normal_train.shape + (1,))
    windows_normal_val = windows_normal_val.reshape(windows_normal_val.shape + (1,))
    windows_attack = windows_attack.reshape(windows_attack.shape + (1,))

    train_data_tensor = torch.from_numpy(windows_normal_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_normal_mas_train).float().to(device)
    train_dataset = MyDataset(train_data_tensor, train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_data_tensor = torch.from_numpy(windows_normal_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_normal_mas_val).float().to(device)
    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_data_tensor = torch.from_numpy(windows_attack).float().to(device)
    test_mas_data_tensor = torch.from_numpy(windows_attack_mas).float().to(device)
    test_dataset = MyDataset(test_data_tensor, test_mas_data_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_test_labels = []
    for w in windows_labels:
        y_test_labels.append(1.0 if (np.sum(w) > 0) else 0.0)

    print("train: ", windows_normal_train.shape)
    print("val: ", windows_normal_val.shape)
    print("test: ", windows_attack.shape)
    print("test labels: ", len(y_test_labels))

    return train_loader, val_loader, test_loader, y_test_labels, scaler
