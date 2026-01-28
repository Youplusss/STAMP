import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch

from lib.utils import *

def _read_csv_robust(path: str, sep=None) -> pd.DataFrame:
    """Read CSV robustly across separators/engines and normalize column names."""
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    # common: index column saved
    unnamed_like = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_like:
        df = df.drop(columns=unnamed_like)
    return df


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _extract_wadi_label_and_drop(df: pd.DataFrame):
    """Return (features_df, labels_array) for WADI test CSV."""
    # Known label column names/variants
    candidate_cols = [
        "label",
        "attack",
        "Attack",
        "Normal/Attack",
        "Attack LABLE (1:No Attack, -1:Attack)",
        "Attack  LABLE (1:No Attack, -1:Attack)",
    ]
    label_col = None
    for c in candidate_cols:
        if c in df.columns:
            label_col = c
            break
    # Fallback: find a column that contains both attack and label/lable
    if label_col is None:
        for c in df.columns:
            s = str(c).strip().lower()
            if "attack" in s and ("label" in s or "lable" in s):
                label_col = c
                break
    if label_col is None:
        raise KeyError(
            "Could not find WADI label column. Expected one of: "
            f"{candidate_cols}. Available columns (first 30): {list(df.columns)[:30]}"
        )

    labels_raw = df[label_col]
    df = df.drop(columns=[label_col])
    # Drop non-sensor columns if present
    df = df.drop(columns=["Row", "Date", "Time", "Date_Time", "Date time", "Timestamp"], errors="ignore")
    labels_num = pd.to_numeric(labels_raw, errors="coerce").fillna(0).values
    # Map common encodings:
    # -1 => attack (anomaly)
    #  1 => normal
    #  0/1 => sometimes already binary
    labels = (labels_num == -1).astype(float)
    if labels.max() == 0 and np.isin(labels_num, [0, 1]).all():
        labels = labels_num.astype(float)
    return df, labels


def preprocessTrainingData(file, sep=None, min_max_scaler = None, training = True):
    # === Normal period ====
    normal = _read_csv_robust(file, sep=sep)
    # Preprocessed wadi_train.csv may already have only sensors; be tolerant.
    normal = normal.drop(columns=['Row', 'Date', 'Time', 'Date_Time', 'Date time', 'Timestamp', 'label', 'attack', 'Attack'], errors='ignore')

    normal = _coerce_numeric_frame(normal)
    normal = normal.fillna(normal.mean(numeric_only=True))
    normal = normal.fillna(0)

    return normal.values, min_max_scaler

def preprocessTestingData(file, sep=None, min_max_scaler = None, training=False):
    # === Normal period ====
    attack = _read_csv_robust(file, sep=sep)

    attack, labels = _extract_wadi_label_and_drop(attack)

    attack = _coerce_numeric_frame(attack)
    attack = attack.fillna(attack.mean(numeric_only=True))
    attack = attack.fillna(0)

    return attack.values, labels


def load_data(train_filename, test_filename, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    normal, min_max_scaler = preprocessTrainingData(train_filename, sep=None, min_max_scaler=None, training=True)  # , nrows=1000)
    attack, labels = preprocessTestingData(test_filename, sep=None, min_max_scaler=min_max_scaler, training=False)  # , nrows=1000)
    normal = normal[21600:,:]
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
    # print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    # print(windows_attack.shape)  # (449907, 12, 51)

    ### train/val/test
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


def load_data2(train_filename, test_filename, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    ## EDA - Data Pre-Processing
    normal, min_max_scaler = preprocessTrainingData(train_filename, sep=None, min_max_scaler=None, training=True)  # , nrows=1000)
    attack, labels = preprocessTestingData(test_filename, sep=None, min_max_scaler=min_max_scaler, training=False)  # , nrows=1000)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
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
    normal = normal[21600:,:]
    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][21600:,:]

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

    # ## nomalization
    min = normal.min()##axis=0
    max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)
    # normal = min_max_scaler.transform(normal)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)

    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    # print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    # print(windows_attack.shape)  # (449907, 12, 51)

    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    windows_normal_mas = np.stack(normal_mas, axis=-1)
    windows_attack_mas = np.stack(attack_mas, axis=-1)


    ### train/val/test
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

def load_data3(normal, attack, labels, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):
    #用有标签数据（含异常）训练和测试

    labels = np.array(labels)
    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    normal_mas = []
    attack_mas = []
    for w in window_sizes:
        normal_ma = np_ma(normal, w)
        normal_mas.append(normal_ma)

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)
    '''
    normal = normal[21600:,:]
    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][21600:,:]
    '''
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

    # ## nomalization
    # min = normal.min()##axis=0
    # max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)
    # normal = min_max_scaler.transform(normal)

    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal)

    attack = min_max_scaler.transform(attack)

    for i in range(len(window_sizes)):
        normal_mas[i] = min_max_scaler.transform(normal_mas[i])
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])

    windows_normal = normal[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
    # print(windows_normal.shape)  # (494988, 12, 51)

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    # print(windows_attack.shape)  # (449907, 12, 51)

    for i in range(len(window_sizes)):
        normal_mas[i] = normal_mas[i][np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size + 1)[:, None]]
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    windows_normal_mas = np.stack(normal_mas, axis=-1)
    windows_attack_mas = np.stack(attack_mas, axis=-1)


    ### train/val/test
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


def load_data_unsup_train(attack, labels, device = "gpu", window_size = 12, val_ratio = 0.2, batch_size = 64, is_down_sample = False, down_len=10):

    labels = np.array(labels)
    print("attack_train: ", attack.shape)
    print("labels: ", labels.shape)

    ### Add Moving Average (MA)
    window_sizes = [3,5,10,20]
    attack_mas = []
    for w in window_sizes:

        attack_ma = np_ma(attack, w)
        attack_mas.append(attack_ma)
        
    W = np.max(window_sizes)
    attack = attack[W:, :]
    labels = labels[W:]
    for i in range(len(window_sizes)):
        attack_mas[i] = attack_mas[i][W:, :]

    ## down sample
    if is_down_sample:
        attack = downsample(attack, down_len=down_len, is_label=False)
        labels = downsample(labels, down_len=down_len, is_label=True)

        for i in range(len(window_sizes)):
            attack_mas[i] = downsample(attack_mas[i], down_len=down_len, is_label=False)

    # ## nomalization
    # min = normal.min()##axis=0
    # max = normal.max()##axis=0
    # min_max_scaler = MinMaxScaler(min, max)

    min_max_scaler = preprocessing.MinMaxScaler()
    attack = min_max_scaler.fit_transform(attack)

    for i in range(len(window_sizes)):
        attack_mas[i] = min_max_scaler.transform(attack_mas[i])
    

    windows_attack = attack[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]
    

    for i in range(len(window_sizes)):
        
        attack_mas[i] = attack_mas[i][np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size + 1)[:, None]]

    
    windows_attack_mas = np.stack(attack_mas, axis=-1)
    
    ## window labels
    windows_labels = []
    for i in range(len(labels) - window_size + 1):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_train_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
    
    
    ## select window
    windows_attack_ = []
    windows_attack_mas_ = []
    for i in range(len(y_train_labels)):
        if y_train_labels[i] == 0:
            windows_attack_.append(windows_attack[i])
            windows_attack_mas_.append(windows_attack_mas[i])
    windows_attack = np.array(windows_attack_)
    windows_attack_mas = np.array(windows_attack_mas_)


    ### train/val/test
    windows_train = windows_attack[:int(np.floor((1-val_ratio) * windows_attack.shape[0]))]
    windows_val = windows_attack[int(np.floor((1-val_ratio) * windows_attack.shape[0])):]

    windows_mas_train = windows_attack_mas[:int(np.floor((1-val_ratio) * windows_attack.shape[0]))]
    windows_mas_val = windows_attack_mas[int(np.floor((1-val_ratio) * windows_attack.shape[0])):]


    ## reshape: [B, T, N ,C]
    windows_train = windows_train.reshape(windows_train.shape + (1,))
    windows_val = windows_val.reshape(windows_val.shape + (1,))

    print("windows_train: ", windows_train.shape)
    print("windows_mas_train: ", windows_mas_train.shape)
    print("windows_val: ", windows_val.shape)
    print("windows_mas_val: ", windows_mas_val.shape)

    ## train
    train_data_tensor = torch.from_numpy(windows_train).float().to(device)
    train_mas_data_tensor = torch.from_numpy(windows_mas_train).float().to(device)

    train_dataset = MyDataset(train_data_tensor,train_mas_data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## val
    val_data_tensor = torch.from_numpy(windows_val).float().to(device)
    val_mas_data_tensor = torch.from_numpy(windows_mas_val).float().to(device)

    val_dataset = MyDataset(val_data_tensor, val_mas_data_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    print("train: ", type(windows_train.shape), windows_train.shape)
    print("val: ", windows_val.shape)

    return train_loader, val_loader, min_max_scaler

