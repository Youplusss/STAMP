# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable


@dataclass
class AnomalySegment:
    """A contiguous segment of anomaly windows (index space of sliding windows)."""
    start: int
    end: int  # inclusive
    peak: int  # an index within [start, end]


def group_contiguous_indices(idxs: Iterable[int]) -> List[Tuple[int, int]]:
    """Group sorted indices into contiguous [start,end] segments (inclusive)."""
    idxs = list(sorted(set(int(i) for i in idxs)))
    if not idxs:
        return []
    segs: List[Tuple[int, int]] = []
    s = e = idxs[0]
    for i in idxs[1:]:
        if i == e + 1:
            e = i
        else:
            segs.append((s, e))
            s = e = i
    segs.append((s, e))
    return segs


def build_segments_from_predictions(pred: List[int] | Tuple[int, ...], scores: List[float] | Tuple[float, ...]) -> List[AnomalySegment]:
    """Build anomaly segments from binary predictions and per-window scores.

    Args:
        pred: length W list/array, 1 indicates anomaly window.
        scores: length W list/array, anomaly score per window (used to pick peak).

    Returns:
        List[AnomalySegment]
    """
    if len(pred) != len(scores):
        raise ValueError(f"pred and scores must have same length, got {len(pred)} vs {len(scores)}")

    anomaly_idxs = [i for i, p in enumerate(pred) if int(p) == 1]
    seg_bounds = group_contiguous_indices(anomaly_idxs)

    segments: List[AnomalySegment] = []
    for s, e in seg_bounds:
        # pick peak by score
        peak = max(range(s, e + 1), key=lambda i: float(scores[i]))
        segments.append(AnomalySegment(start=s, end=e, peak=peak))
    return segments
