"""utils/dataset_stats.py

Generate dataset statistics table (Table 4.1) for this repo.

It reads the *processed* dataset files produced by scripts/process_*.py and reports:
- 训练集大小：<ds>_train.csv 行数
- 测试集大小：<ds>_test.csv 行数
- 维度：特征列数（不包含标签列）
- 异常率（%）：测试集中异常点占比

依赖约定（与本仓库 scripts/process_*.py 对齐）：
- dataset/<DS>/swat_train.csv, swat_test.csv (含 attack 列)
- dataset/<DS>/wadi_train.csv, wadi_test.csv (含 attack 列)
- dataset/<DS>/smd_train.csv,  smd_test.csv  (含 attack 列)
- dataset/<DS>/smap_train.csv, smap_test.csv (含 attack 列)
- dataset/<DS>/msl_train.csv,  msl_test.csv  (含 attack 列)

用法：
  python utils/dataset_stats.py --format md

你可以加 --debug_labels 查看每个测试集 attack 列取值分布（有助于排查 SMAP）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


DEFAULT_DATASETS: Tuple[str, ...] = ("SWaT", "WADI", "SMD", "SMAP", "MSL")


@dataclass
class DatasetStat:
    dataset: str
    train_size: int
    test_size: int
    dim_features: int
    anomaly_rate_pct: float
    dim_paper: str


def _to_bool_anomaly(v) -> bool:
    """把各种 label 约定映射成是否异常。

    目标：避免把“非 0”一概当异常，导致 SMAP 这类统计变成 100%。

    支持：
    - 0/1
    - -1/1（部分数据集 raw 用 -1 表示攻击）
    - 字符串：Attack/Normal, True/False 等
    """
    if v is None:
        return False

    s = str(v).strip()
    if s == "":
        return False

    try:
        num = float(s)
        if num == 1.0:
            return True
        if num == 0.0:
            return False
        if num == -1.0:
            return True
        # 其他非零数值比较少见，谨慎：只把正数当异常
        return num > 0
    except Exception:
        s_low = s.lower()
        if s_low in {"1", "true", "t", "yes", "y", "attack", "anomaly", "abnormal"}:
            return True
        if s_low in {"0", "false", "f", "no", "n", "normal", "nominal"}:
            return False
        return False


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [str(c).strip() for c in header]


def _count_rows_and_attack(
    path: str,
    attack_col: str = "attack",
    *,
    debug: bool = False,
    debug_max_items: int = 12,
) -> Tuple[int, int]:
    """返回 (总行数, 异常点数量)；按行流式读取，避免大文件占内存。"""
    n_rows = 0
    n_anom = 0
    dist: dict[str, int] = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        if attack_col not in reader.fieldnames:
            raise KeyError(f"Missing '{attack_col}' column in {path}. Columns={reader.fieldnames}")

        for row in reader:
            n_rows += 1
            val = row.get(attack_col, "")
            if debug:
                key = str(val).strip()
                dist[key] = dist.get(key, 0) + 1
            if _to_bool_anomaly(val):
                n_anom += 1

    if debug:
        top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[: int(debug_max_items)]
        print(f"[Debug] {os.path.basename(path)} label distribution (top {len(top)}): {top}", file=sys.stderr)

    return n_rows, n_anom


def _resolve_processed_paths(base_dir: str, dataset: str) -> Tuple[str, str]:
    ds = dataset
    ds_dir = os.path.join(base_dir, "dataset", ds)

    if ds.lower() == "swat":
        train_csv = os.path.join(ds_dir, "swat_train.csv")
        test_csv = os.path.join(ds_dir, "swat_test.csv")
    elif ds.lower() == "wadi":
        train_csv = os.path.join(ds_dir, "wadi_train.csv")
        test_csv = os.path.join(ds_dir, "wadi_test.csv")
    elif ds.lower() == "smd":
        train_csv = os.path.join(ds_dir, "smd_train.csv")
        test_csv = os.path.join(ds_dir, "smd_test.csv")
    elif ds.lower() == "smap":
        train_csv = os.path.join(ds_dir, "smap_train.csv")
        test_csv = os.path.join(ds_dir, "smap_test.csv")
    elif ds.lower() == "msl":
        train_csv = os.path.join(ds_dir, "msl_train.csv")
        test_csv = os.path.join(ds_dir, "msl_test.csv")
    else:
        train_csv = os.path.join(ds_dir, f"{ds.lower()}_train.csv")
        test_csv = os.path.join(ds_dir, f"{ds.lower()}_test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing processed train CSV for {dataset}: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing processed test CSV for {dataset}: {test_csv}")

    return train_csv, test_csv


def _paper_dim_string(dataset: str, dim_features: int, base_dir: str) -> str:
    """尽量对齐论文表 4.1 的维度表示（如 28*38 / 55*25 / 27*55）。"""
    ds = dataset.upper()

    list_path = os.path.join(base_dir, "dataset", ds, "list.txt")
    n_nodes: Optional[int] = None
    if os.path.exists(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            n_nodes = len([ln for ln in f.read().splitlines() if ln.strip()])

    if ds == "SMD":
        if dim_features == 38:
            return "28*38"
        return f"{dim_features}"

    if ds == "SMAP":
        if dim_features == 55:
            return "55*25"
        if n_nodes is not None:
            return f"{n_nodes}*25"

    if ds == "MSL":
        if dim_features == 55:
            # 常见为 27 条通道
            return "27*55" if n_nodes is None else f"{n_nodes}*55"

    return f"{dim_features}"


def compute_stats_for_dataset(base_dir: str, dataset: str, *, debug_labels: bool = False) -> DatasetStat:
    train_csv, test_csv = _resolve_processed_paths(base_dir, dataset)

    train_header = _read_csv_header(train_csv)
    test_header = _read_csv_header(test_csv)

    train_feat_cols = [c for c in train_header if c.strip().lower() not in {"attack", "label", "labels"}]
    test_feat_cols = [c for c in test_header if c.strip().lower() not in {"attack", "label", "labels"}]
    dim_features = min(len(train_feat_cols), len(test_feat_cols))

    # train rows count (exclude header)
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        train_size = max(sum(1 for _ in f) - 1, 0)

    test_size, n_anom = _count_rows_and_attack(test_csv, attack_col="attack", debug=debug_labels)
    anomaly_rate_pct = (float(n_anom) / float(test_size) * 100.0) if test_size > 0 else 0.0

    dim_paper = _paper_dim_string(dataset, dim_features=dim_features, base_dir=base_dir)

    return DatasetStat(
        dataset=str(dataset),
        train_size=int(train_size),
        test_size=int(test_size),
        dim_features=int(dim_features),
        anomaly_rate_pct=float(anomaly_rate_pct),
        dim_paper=str(dim_paper),
    )


def _format_md_table(stats: Sequence[DatasetStat], *, use_paper_dim: bool = True) -> str:
    headers = ["数据集", "训练集", "测试集", "维度", "异常率（%）"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for s in stats:
        dim = s.dim_paper if use_paper_dim else str(s.dim_features)
        lines.append(
            "| "
            + " | ".join(
                [
                    s.dataset,
                    str(s.train_size),
                    str(s.test_size),
                    dim,
                    f"{s.anomaly_rate_pct:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_csv(stats: Sequence[DatasetStat], *, use_paper_dim: bool = True) -> str:
    out_lines: List[List[str]] = [["dataset", "train", "test", "dim", "anomaly_rate_pct"]]
    for s in stats:
        dim = s.dim_paper if use_paper_dim else str(s.dim_features)
        out_lines.append([s.dataset, str(s.train_size), str(s.test_size), dim, f"{s.anomaly_rate_pct:.4f}"])

    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerows(out_lines)
    return buf.getvalue()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dataset statistics table (Table 4.1)")
    p.add_argument(
        "--base_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
        help="repo root (default: parent of utils/)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="datasets to include (default: SWaT WADI SMD SMAP MSL)",
    )
    p.add_argument("--format", type=str, choices=["md", "csv"], default="md")
    p.add_argument("--use_raw_dim", action="store_true", help="use raw feature dim in table")
    p.add_argument("--out", type=str, default=None, help="optional output file path")
    p.add_argument("--skip_missing", action="store_true", help="skip missing datasets")
    p.add_argument("--debug_labels", action="store_true", help="print attack label distribution")
    return p.parse_args()


def _missing_hint(base_dir: str, dataset: str) -> str:
    ds = dataset.upper()
    scripts_map = {
        "SWAT": "scripts/process_swat.py",
        "WADI": "scripts/process_wadi.py",
        "SMD": "scripts/process_smd.py",
        "SMAP": "scripts/process_smap.py",
        "MSL": "scripts/process_msl.py",
    }
    rel = scripts_map.get(ds)
    if rel:
        return (
            f"Processed files for {dataset} not found under {os.path.join(base_dir, 'dataset', ds)}. "
            f"You can generate them by running: python {rel} (see script args for raw_root/out_root)."
        )
    return (
        f"Processed files for {dataset} not found. Expected under: {os.path.join(base_dir, 'dataset', ds)}. "
        "Please run the corresponding scripts/process_*.py first."
    )


def main() -> None:
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    stats: List[DatasetStat] = []
    for ds in args.datasets:
        try:
            stats.append(compute_stats_for_dataset(base_dir, ds, debug_labels=args.debug_labels))
        except FileNotFoundError as e:
            if args.skip_missing:
                print(f"[Skip] {e}", file=sys.stderr)
                continue
            print(_missing_hint(base_dir, ds), file=sys.stderr)
            raise

    if args.format == "md":
        text = _format_md_table(stats, use_paper_dim=(not args.use_raw_dim))
    else:
        text = _format_csv(stats, use_paper_dim=(not args.use_raw_dim))

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(out_path)
    else:
        print(text)


if __name__ == "__main__":
    main()
