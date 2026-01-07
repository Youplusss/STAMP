# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


@dataclass
class SeriesChangeDescription:
    """Lightweight, deterministic description of how a univariate series behaves within a window."""
    anomaly_type: str
    direction: str
    evidence: Dict[str, Any]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(-1)[0])


def robust_mad(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad + eps)


def robust_zscore(x: float, ref: np.ndarray, eps: float = 1e-8) -> float:
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    med = np.median(ref)
    mad = robust_mad(ref, eps=eps)
    return float((x - med) / (1.4826 * mad + eps))


def describe_univariate_window(values: np.ndarray) -> SeriesChangeDescription:
    """Heuristic pattern description inside a sliding window.

    The goal is *not* to be perfect, but to provide stable, explainable
    signals that an LLM can turn into fluent language.

    Types (coarse):
      - spike / dip (point-like)
      - level_shift_up / level_shift_down
      - variance_increase
      - trend_up / trend_down
      - oscillation
      - unknown

    Returns:
      SeriesChangeDescription
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    T = v.shape[0]
    if T < 4:
        return SeriesChangeDescription(
            anomaly_type="unknown",
            direction="unknown",
            evidence={"reason": "window too short", "T": int(T)},
        )

    # split early/late
    mid = T // 2
    early = v[:mid]
    late = v[mid:]

    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    early_std = float(np.std(early) + 1e-8)
    late_std = float(np.std(late) + 1e-8)

    # slope (linear fit)
    x = np.arange(T, dtype=np.float64)
    slope = float(np.polyfit(x, v, deg=1)[0])

    # point deviation at end (often strongest for window-level AD)
    last = float(v[-1])
    z_last = robust_zscore(last, early)

    # max step change
    dv = np.diff(v)
    max_step = float(np.max(np.abs(dv)))
    # scale by early variability
    step_ratio = float(max_step / (early_std + 1e-8))

    # level shift detection
    shift = late_mean - early_mean
    shift_ratio = float(np.abs(shift) / (early_std + 1e-8))

    # oscillation: large alternating sign in diff
    sign_changes = np.sum(np.sign(dv[1:]) * np.sign(dv[:-1]) < 0)
    osc_ratio = float(sign_changes / max(1, (T - 2)))

    # variance increase
    var_ratio = float(late_std / (early_std + 1e-8))

    direction = "up" if shift > 0 else "down"

    # rules (ordered)
    if abs(z_last) >= 3.5 and step_ratio >= 2.0:
        # strong end-point outlier
        return SeriesChangeDescription(
            anomaly_type="spike" if z_last > 0 else "dip",
            direction="up" if z_last > 0 else "down",
            evidence={
                "z_last": z_last,
                "max_step": max_step,
                "step_ratio": step_ratio,
                "early_mean": early_mean,
                "early_std": early_std,
                "last": last,
            },
        )

    if shift_ratio >= 2.5:
        return SeriesChangeDescription(
            anomaly_type="level_shift_up" if shift > 0 else "level_shift_down",
            direction=direction,
            evidence={
                "shift": shift,
                "shift_ratio": shift_ratio,
                "early_mean": early_mean,
                "late_mean": late_mean,
                "early_std": early_std,
            },
        )

    if var_ratio >= 2.0:
        return SeriesChangeDescription(
            anomaly_type="variance_increase",
            direction="unknown",
            evidence={
                "var_ratio": var_ratio,
                "early_std": early_std,
                "late_std": late_std,
            },
        )

    if abs(slope) >= (0.5 * early_std):
        return SeriesChangeDescription(
            anomaly_type="trend_up" if slope > 0 else "trend_down",
            direction="up" if slope > 0 else "down",
            evidence={
                "slope": slope,
                "early_std": early_std,
            },
        )

    if osc_ratio >= 0.6 and step_ratio >= 1.5:
        return SeriesChangeDescription(
            anomaly_type="oscillation",
            direction="unknown",
            evidence={
                "osc_ratio": osc_ratio,
                "sign_changes": int(sign_changes),
                "step_ratio": step_ratio,
            },
        )

    return SeriesChangeDescription(
        anomaly_type="unknown",
        direction=direction,
        evidence={
            "shift_ratio": shift_ratio,
            "var_ratio": var_ratio,
            "slope": slope,
            "osc_ratio": osc_ratio,
        },
    )


def series_patch_prototypes(values: np.ndarray, patch_len: int = 8, stride: int = 4, max_patches: int = 16) -> List[str]:
    """Convert a short univariate series into a list of coarse 'prototype tokens'.

    This is a **text-domain** approximation of Time-LLM's 'text prototypes' idea.
    Instead of learning prototypes in embedding space, we deterministically map each patch
    to a small vocabulary of cues.

    Output tokens are from:
      {up, down, steady, volatile, spike, dip}

    These tokens can be inserted into the LLM prompt to provide shape hints without dumping raw numbers.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    T = v.shape[0]
    if T <= 1:
        return ["steady"]

    tokens: List[str] = []
    start = 0
    while start < T:
        end = min(T, start + patch_len)
        patch = v[start:end]
        if patch.shape[0] < 2:
            break

        # basic stats
        slope = float(np.polyfit(np.arange(patch.shape[0]), patch, deg=1)[0])
        std = float(np.std(patch) + 1e-8)
        dv = np.diff(patch)
        max_step = float(np.max(np.abs(dv))) if dv.size else 0.0
        step_ratio = max_step / std

        # endpoint outlier
        z_end = robust_zscore(float(patch[-1]), patch[:-1] if patch.shape[0] > 2 else patch)

        if abs(z_end) >= 3.0 and step_ratio >= 2.0:
            tokens.append("spike" if z_end > 0 else "dip")
        elif abs(slope) < 0.05 * std:
            tokens.append("volatile" if std > 0.5 else "steady")
        else:
            tokens.append("up" if slope > 0 else "down")

        if len(tokens) >= max_patches:
            break
        start += stride

    if not tokens:
        tokens = ["steady"]
    return tokens


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v) + 1e-8),
        "last": float(v[-1]),
        "first": float(v[0]),
    }
