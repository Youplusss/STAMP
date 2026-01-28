# -*- coding: utf-8 -*-
"""Grid search for STAMP test weights (alpha/beta/gamma).

This script runs `test.py` repeatedly with different combinations of:
  --test_alpha, --test_beta, --test_gamma

By default it enforces alpha + beta + gamma == 1.

It parses the console output from `test.py` and extracts the best-f1 for
aggregation methods: max / sum / mean.

Usage examples
--------------
# Use a base command (everything except weights) and search weights on a coarse grid.
python scripts/search_test_weights.py \
  --base-cmd "python test.py --data SWaT --pred_model mamba --recon_model mamba --gpu_id 0 --batch_size 128" \
  --step 0.1

# Finer grid
python scripts/search_test_weights.py --base-cmd "python test.py --data SWaT --pred_model mamba --recon_model mamba --gpu_id 0" --step 0.05

# Control search space
python scripts/search_test_weights.py --base-cmd "python test.py --data SWaT --pred_model mamba --recon_model mamba" --step 0.1 --min 0.0 --max 1.0

Notes
-----
- The script suppresses *all* outputs from `test.py` except the final summary.
- It is robust to extra prints in `test.py` as long as the `get_final_result`
  output contains a dict-like line with keys including 'best-f1'.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


_METHODS = ("max", "sum", "mean")


@dataclass
class BestRecord:
    f1: float = float("-inf")
    precision: float = float("nan")
    recall: float = float("nan")
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    raw_line: str = ""


_FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# Match a python dict-ish line that contains best-f1.
# Handles formats like:
#   {'best-f1': np.float64(0.9540), ...}
#   {'best_f1': 0.95, ...}
_BEST_F1_LINE_RE = re.compile(r"\{[^\n\r]*?(?:best-f1|best_f1|f1)[^\n\r]*?}")

# Extract a numeric best-f1 value.
# Accept both:
#   'best-f1': np.float64(0.954)
#   'best-f1': 0.954
_BEST_F1_VAL_RE = re.compile(
    rf"['\"]?(?:best-f1|best_f1|f1)['\"]?\s*[:=]\s*(?:np\.float\d+\()?\s*({_FLOAT_RE})",
    re.IGNORECASE,
)

_PRECISION_VAL_RE = re.compile(
    rf"['\"]?precision['\"]?\s*[:=]\s*(?:np\.float\d+\()?\s*({_FLOAT_RE})",
    re.IGNORECASE,
)

_RECALL_VAL_RE = re.compile(
    rf"['\"]?recall['\"]?\s*[:=]\s*(?:np\.float\d+\()?\s*({_FLOAT_RE})",
    re.IGNORECASE,
)


def _frange_grid(min_v: float, max_v: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--step must be > 0")
    # avoid accumulation error: work in ints
    n = int(round((max_v - min_v) / step))
    vals = [min_v + i * step for i in range(n + 1)]
    # clamp
    out = []
    for v in vals:
        if v < min_v - 1e-12 or v > max_v + 1e-12:
            continue
        out.append(round(v, 10))
    return out


def _build_cmd(base_cmd: str, alpha: float, beta: float, gamma: float) -> List[str]:
    parts = shlex.split(base_cmd)
    parts += [
        "--test_alpha",
        str(alpha),
        "--test_beta",
        str(beta),
        "--test_gamma",
        str(gamma),
    ]
    return parts


def _extract_metrics_from_info_line(info_line: str) -> Tuple[float, float, float]:
    """Extract (f1, precision, recall) from a dict-like info line."""
    f1 = float("nan")
    p = float("nan")
    r = float("nan")
    m = _BEST_F1_VAL_RE.search(info_line)
    if m:
        f1 = float(m.group(1))
    mp = _PRECISION_VAL_RE.search(info_line)
    if mp:
        p = float(mp.group(1))
    mr = _RECALL_VAL_RE.search(info_line)
    if mr:
        r = float(mr.group(1))
    return f1, p, r


def _extract_best_f1_by_method(stdout: str) -> Dict[str, float]:
    """Parse test.py output and return best metrics for each method.

    We assume `test.py` prints a dict-like line after each method section.
    The order is typically max -> sum -> mean, but we also fallback by scanning
    method headers.
    """
    # Strategy:
    # 1) Split output into blocks per method based on header lines.
    # 2) Inside each block, find the last dict-ish line containing best-f1.
    by_method: Dict[str, Tuple[float, float, float]] = {}

    lower = stdout.lower()

    # Locate method blocks by their headers (printed in test.py)
    # e.g. "Find best f1 from score (method=max)"
    indices: List[Tuple[str, int]] = []
    for m in _METHODS:
        # test.py prints headers like: "Find best f1 from score (method=max)"
        # So we search for "method=max" etc.
        key = f"method={m}".lower()
        pos = lower.find(key)
        if pos != -1:
            indices.append((m, pos))

    if indices:
        indices.sort(key=lambda x: x[1])
        for i, (m, start) in enumerate(indices):
            end = indices[i + 1][1] if i + 1 < len(indices) else len(stdout)
            block = stdout[start:end]
            # find best-f1 line
            lines = _BEST_F1_LINE_RE.findall(block)
            if not lines:
                continue
            last = lines[-1]
            f1, p, r = _extract_metrics_from_info_line(last)
            if math.isfinite(f1):
                by_method[m] = (f1, p, r)

    # Fallback: if we couldn't split, just take the first 3 best-f1 lines.
    if len(by_method) < 3:
        lines_all = _BEST_F1_LINE_RE.findall(stdout)
        vals: List[Tuple[float, float, float]] = []
        for ln in lines_all:
            f1, p, r = _extract_metrics_from_info_line(ln)
            if math.isfinite(f1):
                vals.append((f1, p, r))
        if len(vals) >= 3:
            # assume order max,sum,mean
            for m, v in zip(_METHODS, vals[:3]):
                by_method.setdefault(m, v)

    # For backward compatibility with existing call sites, convert to f1-only dict.
    # (Callers inside this file should use _extract_best_metrics_by_method instead.)
    return {k: float(v[0]) for k, v in by_method.items()}


def _extract_best_metrics_by_method(stdout: str) -> Dict[str, Tuple[float, float, float]]:
    """Parse test.py output and return (f1, precision, recall) for each method."""
    lower = stdout.lower()
    indices: List[Tuple[str, int]] = []
    for m in _METHODS:
        key = f"method={m}".lower()
        pos = lower.find(key)
        if pos != -1:
            indices.append((m, pos))

    by_method: Dict[str, Tuple[float, float, float]] = {}
    if indices:
        indices.sort(key=lambda x: x[1])
        for i, (m, start) in enumerate(indices):
            end = indices[i + 1][1] if i + 1 < len(indices) else len(stdout)
            block = stdout[start:end]
            lines = _BEST_F1_LINE_RE.findall(block)
            if not lines:
                continue
            last = lines[-1]
            f1, p, r = _extract_metrics_from_info_line(last)
            if math.isfinite(f1):
                by_method[m] = (f1, p, r)

    if len(by_method) < 3:
        lines_all = _BEST_F1_LINE_RE.findall(stdout)
        vals: List[Tuple[float, float, float]] = []
        for ln in lines_all:
            f1, p, r = _extract_metrics_from_info_line(ln)
            if math.isfinite(f1):
                vals.append((f1, p, r))
        if len(vals) >= 3:
            for m, v in zip(_METHODS, vals[:3]):
                by_method.setdefault(m, v)

    return by_method


def _run_one(base_cmd: str, alpha: float, beta: float, gamma: float, timeout_s: int) -> Dict[str, float]:
    cmd = _build_cmd(base_cmd, alpha, beta, gamma)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    out = proc.stdout or ""
    parsed = _extract_best_metrics_by_method(out)
    # return f1-only dict for legacy callers
    return {k: float(v[0]) for k, v in parsed.items()}


def _run_one_metrics(base_cmd: str, alpha: float, beta: float, gamma: float, timeout_s: int) -> Dict[str, Tuple[float, float, float]]:
    cmd = _build_cmd(base_cmd, alpha, beta, gamma)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    out = proc.stdout or ""
    return _extract_best_metrics_by_method(out)


def _iter_triplets_sum_to_one(values: List[float], tol: float = 1e-9) -> List[Tuple[float, float, float]]:
    # Enumerate alpha,beta then compute gamma=1-a-b and keep if in grid.
    # Use a set for fast membership.
    s = set(values)
    triplets = []
    # Quantize to the grid step inferred from input values to avoid -0.0/1.0000000002 artifacts.
    # Example: step=0.1 -> scale=10
    step = None
    if len(values) >= 2:
        diffs = sorted({round(abs(values[i + 1] - values[i]), 12) for i in range(len(values) - 1) if abs(values[i + 1] - values[i]) > 0})
        if diffs:
            step = diffs[0]
    scale = int(round(1.0 / step)) if step and step > 0 else 10

    for a in values:
        for b in values:
            g = 1.0 - a - b
            # snap to nearest grid point
            g = round(round(g * scale) / scale, 10)
            # eliminate negative zero
            if abs(g) < 1e-12:
                g = 0.0
            # reject tiny negatives / overflows caused by float
            if g < min(values) - 1e-9 or g > max(values) + 1e-9:
                continue
            if abs((a + b + g) - 1.0) > 1e-6:
                continue
            if g in s:
                triplets.append((a, b, g))
    # Deterministic order
    triplets.sort()
    return triplets


def main() -> int:
    p = argparse.ArgumentParser(description="Grid search best test_alpha/test_beta/test_gamma (sum to 1).")
    p.add_argument(
        "--base-cmd",
        type=str,
        required=True,
        help='Base command to run, without weight flags, e.g. "python test.py --data SWaT --pred_model mamba --recon_model mamba --gpu_id 0 --batch_size 128"',
    )
    p.add_argument("--step", type=float, default=0.1, help="Grid step size (default 0.1)")
    p.add_argument("--min", dest="min_v", type=float, default=0.0, help="Min value for each weight (default 0.0)")
    p.add_argument("--max", dest="max_v", type=float, default=1.0, help="Max value for each weight (default 1.0)")
    p.add_argument("--timeout", type=int, default=3600, help="Timeout seconds per test run")
    p.add_argument("--topk", type=int, default=1, help="Optional: print top-k candidates per method (default 1)")

    args = p.parse_args()

    grid = _frange_grid(args.min_v, args.max_v, args.step)
    triplets = _iter_triplets_sum_to_one(grid)
    if not triplets:
        print("No (alpha,beta,gamma) triplets found for the given grid. Try a larger range or different step.")
        return 2

    # Keep best per method
    best: Dict[str, BestRecord] = {m: BestRecord() for m in _METHODS}

    # We'll also optionally keep top-k lists (small; only if requested)
    topk = int(args.topk)
    top_lists: Dict[str, List[BestRecord]] = {m: [] for m in _METHODS}

    total = len(triplets)
    for i, (a, b, g) in enumerate(triplets, 1):
        try:
            results = _run_one_metrics(args.base_cmd, a, b, g, timeout_s=int(args.timeout))
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

        for m in _METHODS:
            if m not in results:
                continue
            f1, prec, rec = results[m]

            # update best
            if f1 > best[m].f1:
                best[m] = BestRecord(f1=f1, precision=prec, recall=rec, alpha=a, beta=b, gamma=g)

            if topk > 1:
                top_lists[m].append(BestRecord(f1=f1, precision=prec, recall=rec, alpha=a, beta=b, gamma=g))

        # Minimal progress to stderr (won't pollute stdout summary too much)
        if i % max(1, total // 20) == 0:
            print(f"[Progress] {i}/{total} tried", file=sys.stderr)

    # Final report: exactly what user asked for
    print("=== Best weights by aggregation method (optimize best-f1) ===")
    for m in _METHODS:
        r = best[m]
        if not math.isfinite(r.f1):
            print(f"{m}: no result parsed")
            continue
        p = r.precision
        rc = r.recall
        print(
            f"{m}: best-f1={r.f1:.6f}  P={p:.6f}  R={rc:.6f}  "
            f"alpha={r.alpha}  beta={r.beta}  gamma={r.gamma}"
        )

    if topk > 1:
        print("\n=== Top candidates (for reference) ===")
        for m in _METHODS:
            cands = sorted(top_lists[m], key=lambda x: x.f1, reverse=True)[:topk]
            if not cands:
                continue
            print(f"[{m}]")
            for r in cands:
                print(
                    f"  best-f1={r.f1:.6f}  P={r.precision:.6f}  R={r.recall:.6f}  "
                    f"alpha={r.alpha}  beta={r.beta}  gamma={r.gamma}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
