# -*- coding: utf-8 -*-
"""Collect branch/test.py metrics from log files and export to CSV.

This script scans the experiment log directory (by default: expe_branch/log)
for *branch-only test* logs and extracts P/R/F1 for method=max/mean/sum.

It is designed for the logging style used by branch/test.py:
  ================= Find best f1 from score (method=max) =================
  {'best-f1': ..., 'precision': ..., 'recall': ..., ...}

Output CSV columns:
  1) beijing_time
  2-4) max_precision, max_recall, max_f1
  5-7) mean_precision, mean_recall, mean_f1
  8-10) sum_precision, sum_recall, sum_f1

Incomplete logs (missing any of the three methods) are skipped.

Usage (PowerShell):
  python utils/collect_branch_test_logs.py --data SWaT --branch mamba_recon

Optional:
  --log_dir expe_branch/log   # override log directory
  --out_dir utils/out         # where to write the CSV

"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


# The logger formatter uses "%Y-%m-%d %H:%M" (see lib/logger.py)
_LOG_TS_FMT = "%Y-%m-%d %H:%M"

# Beijing time is UTC+8. We assume log timestamps are in local machine time.
# For reproducibility across machines, we treat the parsed naive time as local
# time and *then* label it as Beijing time by adding +8h offset from UTC.
# If you want exact conversion from local timezone, add --assume_tz.
_BJ_TZ = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class MetricsRow:
    log_path: str
    beijing_dt: datetime
    # NEW: ablation params
    num_mamba_layers: str
    lss_residual: str
    local_conv_variant: str
    kernel_sizes: str
    max_p: float
    max_r: float
    max_f1: float
    mean_p: float
    mean_r: float
    mean_f1: float
    sum_p: float
    sum_r: float
    sum_f1: float


_METHOD_HEADER_RE = re.compile(r"Find best f1 from score \(method=(max|mean|sum)\)")


def _safe_float(v: Any) -> float:
    """Convert values like np.float64(0.123) or '0.123' to float."""
    if v is None:
        raise ValueError("None is not a float")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    # handle 'np.float64(0.123)'
    m = re.match(r"^\s*np\.float\d+\((.*)\)\s*$", s)
    if m:
        return float(m.group(1))
    return float(s)


def _parse_first_timestamp(content: str) -> datetime | None:
    """Parse the first logger timestamp at the start of a line."""
    # logger lines look like: '2025-01-29 12:34: message'
    m = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):", content, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), _LOG_TS_FMT)
    except ValueError:
        return None


def _extract_method_dict(content: str, method: str) -> dict[str, Any] | None:
    """Extract the first dict printed after a given method header."""
    # We scan line-by-line: once we hit the method header, the next line which
    # starts with '{' (or contains '{') is treated as the info dict.
    lines = content.splitlines()
    want = f"method={method}"
    seen_header = False
    for line in lines:
        if not seen_header:
            if want in line and "Find best f1 from score" in line:
                seen_header = True
            continue

        # After header: find a line containing a dict literal.
        if "{" in line and "}" in line:
            # strip leading timestamps / extra text
            # e.g. "2025-01-29 12:34: {'best-f1': ...}"
            idx = line.find("{")
            candidate = line[idx:]
            try:
                d = ast.literal_eval(candidate)
            except Exception:
                continue
            if isinstance(d, dict) and ("precision" in d and "recall" in d and ("best-f1" in d or "f1" in d)):
                return d
    return None


def _metrics_from_info(d: dict[str, Any]) -> tuple[float, float, float]:
    p = _safe_float(d.get("precision"))
    r = _safe_float(d.get("recall"))
    f1 = _safe_float(d.get("best-f1", d.get("f1")))
    return p, r, f1


def _is_relevant_log_filename(filename: str, *, data: str, model: str) -> bool:
    """Check if a log filename matches dataset+model.

    We accept both:
      - *_test.log (branch/test.py)
      - *_train.log (sometimes users only keep train logs, or test output is appended)

    Filename format (lib/logger.py):
      <run_id>_<DATA>_<MODEL>_<TAG>.log
    """

    fn = filename.lower()
    if not (fn.endswith("_test.log") or fn.endswith("_train.log")):
        return False
    return f"_{data.lower()}_" in fn and f"_{model.lower()}_" in fn


def _extract_bestf1_block(content: str, method: str) -> dict[str, Any] | None:
    """Extract structured BestF1/<method> blocks written by lib.logger.log_test_results.

    Example lines in log:
      2026-01-29 12:34: [BestF1/max]
      2026-01-29 12:34: - best-f1: 0.95
      2026-01-29 12:34: - precision: 0.90
      2026-01-29 12:34: - recall: 1.0

    We convert them into a dict compatible with _metrics_from_info.
    """

    header = f"[BestF1/{method}]"
    lines = content.splitlines()
    out: dict[str, Any] = {}
    seen = False
    for line in lines:
        if not seen:
            if header in line:
                seen = True
            continue

        # stop when next block starts
        if "[BestF1/" in line and header not in line:
            break
        if "[" in line and "]" in line and "[BestF1/" not in line and out:
            # another new section (Hyperparameters/TestRun/...) after we started collecting
            break

        m = re.search(r"-\s*([A-Za-z0-9_\-]+)\s*:\s*(.*)\s*$", line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        out[k] = v

    if "precision" in out and "recall" in out and ("best-f1" in out or "f1" in out):
        return out
    return None


def _extract_method_metrics(content: str, method: str) -> dict[str, Any] | None:
    """Try both supported log formats to get metrics dict for a method."""
    d = _extract_method_dict(content, method)
    if d is not None:
        return d
    return _extract_bestf1_block(content, method)


def _extract_ablation_block(content: str) -> dict[str, str] | None:
    """Extract the [Ablation] block written by branch/test.py.

    Expected lines:
      2026-01-29 12:34: [Ablation]
      2026-01-29 12:34: - num_mamba_layers: 1
      2026-01-29 12:34: - lss_residual: True
      2026-01-29 12:34: - local_conv_variant: only_dwconv
      2026-01-29 12:34: - kernel_sizes: 3,5

    Returns a dict with these keys if found, else None.
    """

    lines = content.splitlines()
    seen = False
    out: dict[str, str] = {}

    for line in lines:
        if not seen:
            if "[Ablation]" in line:
                seen = True
            continue

        # stop when next section starts (next bracketed header)
        if "[" in line and "]" in line and "[Ablation]" not in line and out:
            break

        m = re.search(r"-\s*([A-Za-z0-9_\-]+)\s*:\s*(.*)\s*$", line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        out[k] = v

    if not out:
        return None

    want = {"num_mamba_layers", "lss_residual", "local_conv_variant", "kernel_sizes"}
    if not (want & set(out.keys())):
        return None
    return out


def collect_rows(log_dir: str, *, data: str, branch: str) -> list[MetricsRow]:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(
            "Log directory not found: "
            f"{log_dir}\n"
            "Tip: branch/test.py writes logs under <log_dir>/log. "
            "If you used a different --log_dir when running branch/test.py, pass it here via --log_dir."
        )

    model = f"branch_{branch}"

    rows: list[MetricsRow] = []
    for name in os.listdir(log_dir):
        name_str = str(name)
        name_l = name_str.lower()
        if not name_l.endswith(".log"):
            continue
        if not _is_relevant_log_filename(name_str, data=data, model=model):
            continue

        path = os.path.join(log_dir, name_str)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue

        ts = _parse_first_timestamp(content)
        if ts is None:
            # no timestamp => likely not produced by our logger; skip
            continue

        # Interpret the timestamp as Beijing time (UTC+8) for reporting.
        # If the log was written in local Beijing time already, this is correct.
        beijing_dt = ts.replace(tzinfo=_BJ_TZ)

        info_max = _extract_method_metrics(content, "max")
        info_mean = _extract_method_metrics(content, "mean")
        info_sum = _extract_method_metrics(content, "sum")
        if not (info_max and info_mean and info_sum):
            # interrupted or incomplete => skip
            continue

        ab = _extract_ablation_block(content) or {}

        try:
            max_p, max_r, max_f1 = _metrics_from_info(info_max)
            mean_p, mean_r, mean_f1 = _metrics_from_info(info_mean)
            sum_p, sum_r, sum_f1 = _metrics_from_info(info_sum)
        except Exception:
            continue

        rows.append(
            MetricsRow(
                log_path=path,
                beijing_dt=beijing_dt,
                num_mamba_layers=str(ab.get("num_mamba_layers", "")),
                lss_residual=str(ab.get("lss_residual", "")),
                local_conv_variant=str(ab.get("local_conv_variant", "")),
                kernel_sizes=str(ab.get("kernel_sizes", "")),
                max_p=max_p,
                max_r=max_r,
                max_f1=max_f1,
                mean_p=mean_p,
                mean_r=mean_r,
                mean_f1=mean_f1,
                sum_p=sum_p,
                sum_r=sum_r,
                sum_f1=sum_f1,
            )
        )

    rows.sort(key=lambda r: r.beijing_dt)
    return rows


def _default_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_log_dir() -> str:
    return os.path.join(_default_repo_root(), "expe_branch", "log")


def _default_out_dir() -> str:
    return os.path.join(_default_repo_root(), "expe_branch", "csv")


def write_csv(rows: list[MetricsRow], *, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "beijing_time",
                "num_mamba_layers",
                "lss_residual",
                "local_conv_variant",
                "kernel_sizes",
                "max_precision",
                "max_recall",
                "max_f1",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "sum_precision",
                "sum_recall",
                "sum_f1",
                "log_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.beijing_dt.strftime("%Y-%m-%d %H:%M"),
                    r.num_mamba_layers,
                    r.lss_residual,
                    r.local_conv_variant,
                    r.kernel_sizes,
                    f"{r.max_p:.4g}",
                    f"{r.max_r:.4g}",
                    f"{r.max_f1:.4g}",
                    f"{r.mean_p:.4g}",
                    f"{r.mean_r:.4g}",
                    f"{r.mean_f1:.4g}",
                    f"{r.sum_p:.4g}",
                    f"{r.sum_r:.4g}",
                    f"{r.sum_f1:.4g}",
                    r.log_path,
                ]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect branch test logs and export metrics to CSV")
    p.add_argument("--data", type=str, required=True, help="dataset name, e.g. SWaT")
    p.add_argument("--branch", type=str, required=True, choices=["llm_pred", "mamba_pred", "mamba_recon"], help="branch name")
    p.add_argument("--log_dir", type=str, default=_default_log_dir(), help="directory containing *.log (default: expe_branch/log)")
    p.add_argument("--out_dir", type=str, default=_default_out_dir(), help="output directory for CSV")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    rows = collect_rows(args.log_dir, data=args.data, branch=args.branch)

    # output file naming: <data>_<branch>_branch_test_metrics_<YYYYmmdd>.csv
    out_name = f"{args.data}_{args.branch}_branch_test_metrics.csv"
    out_path = os.path.join(args.out_dir, out_name)

    write_csv(rows, out_path=out_path)

    print(f"Found {len(rows)} complete log(s).")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
