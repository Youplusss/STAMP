import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric when possible; non-convertible values become NaN."""
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _fillna_with_column_means(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs using per-column means (numeric-only) and finally fill remaining NaNs with 0."""
    df = df.copy()
    means = df.mean(numeric_only=True)
    df = df.fillna(means)
    df = df.fillna(0)
    return df


def main():

    # WADI raw CSVs often contain mixed dtypes; low_memory=False avoids chunked inference surprises.
    train = pd.read_csv('./data/WADI/WADI_14days_new.csv', index_col=0, low_memory=False)
    test = pd.read_csv('./data/WADI/WADI_attackdataLABLE.csv', index_col=0, header=1, low_memory=False)


    train = train.iloc[:, 3:]
    test = test.iloc[:, 3:]

    # (debug) uncomment to inspect columns
    # print(len(test.columns), test.columns)

    # trim column names
    train = train.rename(columns=lambda x: str(x).strip())
    test = test.rename(columns=lambda x: str(x).strip())

    # normalize label column name to 'attack' if present
    def _is_attack_label_col(col) -> bool:
        s = str(col).strip()
        s_norm = re.sub(r"\s+", " ", s).lower()
        # fast path: your file prints exactly this
        if s_norm == "attack lable (1:no attack, -1:attack)":
            return True
        # robust path: contains words
        return ("attack" in s_norm) and ("lable" in s_norm or "label" in s_norm) and ("no attack" in s_norm)

    test = test.rename(columns=lambda x: 'attack' if _is_attack_label_col(x) else str(x).strip())

    # Fallback: if for any reason the rename didn't hit (e.g., header quirks), find the column explicitly.
    if 'attack' not in test.columns:
        for c in list(test.columns):
            if _is_attack_label_col(c):
                test = test.rename(columns={c: 'attack'})
                break

    # (debug) uncomment to inspect columns after renaming
    # print(len(test.columns), test.columns)

    # keep only numeric sensor columns; coerce anything weird to NaN
    # (we'll handle the label column separately)
    if 'attack' in test.columns:
        test_labels_raw = test['attack']
        test = test.drop(columns=['attack'])
    else:
        raise KeyError("Could not find attack label column in WADI attack CSV after renaming")

    train = _coerce_numeric_df(train)
    test = _coerce_numeric_df(test)

    train = _fillna_with_column_means(train)
    test = _fillna_with_column_means(test)

    train_labels = np.zeros(len(train))
    # WADI labels are commonly 1 for normal and -1 for attack.
    # Convert to numeric then map: (-1 -> 1 anomaly) else 0.
    test_labels_num = pd.to_numeric(test_labels_raw, errors='coerce').fillna(1).values
    test_labels = (test_labels_num == -1).astype(float)

    # remove column name prefixes (some WADI dumps use long prefixes)
    cols = [str(x)[46:] if len(str(x)) > 46 else str(x) for x in train.columns]
    train.columns = cols
    test.columns = cols


    x_train, x_test = norm(train.values, test.values)


    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]



    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)


    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]

    train_df.to_csv('./dataset/WADI/wadi_train.csv')
    test_df.to_csv('./dataset/WADI/wadi_test.csv')

    f = open('./dataset/WADI/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
