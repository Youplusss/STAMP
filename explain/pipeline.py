# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .segment import build_segments_from_predictions
from .statistics import describe_univariate_window, series_patch_prototypes, summarize_values
from .prompt_builder import build_anomaly_explanation_prompt
from .llm_backend import build_explainer


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def aggregate_topk_scores(test_scores: np.ndarray, topk: int, topk_agg: str = 'sum') -> np.ndarray:
    """Aggregate per-feature scores into a per-window score using top-k features.

    Args:
        test_scores: shape (F, W)
        topk: number of features to aggregate
        topk_agg: sum | mean | max

    Returns:
        total_scores: shape (W,)
    """
    test_scores = np.asarray(test_scores)
    if test_scores.ndim != 2:
        raise ValueError(f"test_scores must be 2D (F,W), got {test_scores.shape}")
    F, W = test_scores.shape
    k = int(max(1, min(topk, F)))
    # indices of top-k along feature axis for each window
    topk_indices = np.argpartition(test_scores, range(F - k, F), axis=0)[-k:]
    topk_vals = np.take_along_axis(test_scores, topk_indices, axis=0)
    agg = (topk_agg or 'sum').lower()
    if agg == 'mean':
        return np.mean(topk_vals, axis=0)
    if agg == 'max':
        return np.max(topk_vals, axis=0)
    return np.sum(topk_vals, axis=0)


def _device_to_str(device: Any) -> str:
    try:
        # torch.device
        return str(device)
    except Exception:
        return 'cpu'


def _feature_name(idx: int, nnodes: Optional[int] = None, out_channels: int = 1) -> str:
    # Most STAMP configs use out_channels=1. If out_channels>1, we still show flat index.
    if nnodes is not None and out_channels == 1:
        # map 0..N-1 to sensor_i
        if 0 <= idx < int(nnodes):
            return f"sensor_{idx}"
    return f"feature_{idx}"


def generate_explanations(
    *,
    args: Any,
    dataset: str,
    test_scores: np.ndarray,           # (F, W)
    predict: np.ndarray,               # (W,)
    test_pred_results: List[np.ndarray],     # [pred, gt] shapes (W,n_pred,F)
    test_ae_results: List[np.ndarray],       # [construct, origin] shapes (W,win,F)
    test_generate_results: List[np.ndarray], # [gen, gen_construct] shapes (W,win,F)
    option: int = 2,
    method: str = 'max',
    threshold: Optional[float] = None,
    out_json_path: Optional[str] = None,
    out_md_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate LLM-based explanations for predicted anomaly segments.

    This function is designed to be called after `test.py` computes `test_scores` and `predict`.

    It will:
      1) compute per-window total scores (top-k aggregation)
      2) group contiguous anomaly windows into segments
      3) extract feature-level evidences (top variables, pattern type, stats)
      4) build a prompt and call an LLM (or template fallback)
      5) save explanations to JSON/Markdown

    Returns:
        dict with metadata and a list of explanations.
    """
    dataset_up = str(dataset).upper()
    test_scores = np.asarray(test_scores)
    predict = np.asarray(predict).astype(int).reshape(-1)

    F, W = test_scores.shape
    if predict.shape[0] != W:
        raise ValueError(f"predict length must match W={W}, got {predict.shape}")

    # Aggregate per-window score
    topk_in_score = int(getattr(args, 'test_topk', 1))
    topk_agg = str(getattr(args, 'test_topk_agg', 'sum'))
    win_scores = aggregate_topk_scores(test_scores, topk=topk_in_score, topk_agg=topk_agg)

    # Build anomaly segments
    segments = build_segments_from_predictions(predict.tolist(), win_scores.tolist())

    # Limit number of segments (for speed)
    max_segments = int(getattr(args, 'explain_max_segments', 20))
    segments = segments[:max_segments]

    # compute branch-wise scores (for contribution breakdown)
    alpha = float(getattr(args, 'test_alpha', 0.4))
    beta = float(getattr(args, 'test_beta', 0.3))
    gamma = float(getattr(args, 'test_gamma', 0.3))

    # We reuse the repo's scoring utilities to stay consistent.
    try:
        from lib.evaluate import get_Test_scores_err_max
    except Exception as e:
        raise ImportError("Cannot import lib.evaluate.get_Test_scores_err_max; ensure you run inside STAMP repo.") from e

    pred_scores = np.zeros((F, W), dtype=np.float64)
    ae_scores = np.zeros((F, W), dtype=np.float64)
    gen_scores = np.zeros((F, W), dtype=np.float64)
    if alpha > 0:
        pred_scores = get_Test_scores_err_max(test_pred_results, option=option, method=method)
    if beta > 0:
        ae_scores = get_Test_scores_err_max(test_ae_results, option=option, method=method)
    if gamma > 0:
        gen_scores = get_Test_scores_err_max(test_generate_results, option=option, method=method)

    # Evidence source arrays
    construct = np.asarray(test_ae_results[0])  # (W, win, F)
    origin = np.asarray(test_ae_results[1])     # (W, win, F)
    pred = np.asarray(test_pred_results[0])     # (W, n_pred, F)
    gt = np.asarray(test_pred_results[1])       # (W, n_pred, F)

    window_size = int(getattr(args, 'window_size', origin.shape[1]))
    n_pred = int(getattr(args, 'n_pred', pred.shape[1]))
    nnodes = getattr(args, 'nnodes', None)
    out_channels = int(getattr(args, 'out_channels', 1))

    # --- global baselines per feature (computed on test origin windows) ---
    # These give a global reference distribution for each feature, computed from all test windows:
    # - baseline_median: robust center
    # - baseline_mad: median absolute deviation (robust scale)
    # - baseline_p05/p95: robust value range
    origin_flat = origin.reshape(-1, origin.shape[-1]).astype(np.float64)  # (W*win, F)
    baseline_median = np.median(origin_flat, axis=0)
    baseline_p05 = np.percentile(origin_flat, 5, axis=0)
    baseline_p95 = np.percentile(origin_flat, 95, axis=0)
    baseline_mad = np.median(np.abs(origin_flat - baseline_median[None, :]), axis=0) + 1e-8

    topk_features = int(getattr(args, 'explain_topk_features', 5))
    language = str(getattr(args, 'explain_language', 'zh'))

    # LLM backend config
    backend = str(getattr(args, 'explain_backend', 'template'))
    llm_model = str(getattr(args, 'explain_llm_model', 'gpt2'))
    max_new_tokens = int(getattr(args, 'explain_max_new_tokens', 256))
    temperature = float(getattr(args, 'explain_temperature', 0.0))
    do_sample = bool(getattr(args, 'explain_do_sample', False))
    top_p = float(getattr(args, 'explain_top_p', 0.95))
    repetition_penalty = float(getattr(args, 'explain_repetition_penalty', 1.05))
    trust_remote_code = bool(getattr(args, 'explain_trust_remote_code', False))
    local_files_only = bool(getattr(args, 'explain_local_files_only', False))
    hf_endpoint = getattr(args, 'explain_hf_endpoint', None)
    hf_cache_dir = getattr(args, 'explain_hf_cache_dir', None)
    explain_force_gpu = bool(getattr(args, 'explain_force_gpu', False))

    explainer = build_explainer(
        backend=backend,
        model_name_or_path=llm_model,
        device=_device_to_str(getattr(args, 'device', 'cpu')),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        hf_endpoint=hf_endpoint,
        cache_dir=hf_cache_dir,
        force_gpu=explain_force_gpu,
    )

    explanations: List[Dict[str, Any]] = []

    for seg in segments:
        # segment-level feature importance
        seg_scores = np.mean(test_scores[:, seg.start:seg.end + 1], axis=1)  # (F,)
        feat_rank = np.argsort(seg_scores)[::-1]
        top_feat_idx = feat_rank[:min(topk_features, F)].tolist()

        # compute contribution ratios among chosen top features
        top_feat_scores = seg_scores[top_feat_idx]
        denom = float(np.sum(np.abs(top_feat_scores)) + 1e-12)

        # global evidence
        seg_win_scores = win_scores[seg.start:seg.end + 1]
        global_evidence = {
            "score_agg": {"topk": topk_in_score, "topk_agg": topk_agg},
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "segment_score": {
                "mean": float(np.mean(seg_win_scores)),
                "max": float(np.max(seg_win_scores)),
                "min": float(np.min(seg_win_scores)),
            },
        }

        feature_evidence: List[Dict[str, Any]] = []
        peak = int(seg.peak)

        for idx in top_feat_idx:
            name = _feature_name(int(idx), nnodes=nnodes, out_channels=out_channels)

            # window series (origin)
            series = origin[peak, :, idx].astype(np.float64).reshape(-1)
            series_hat = construct[peak, :, idx].astype(np.float64).reshape(-1)

            # stats & pattern
            stats = summarize_values(series)
            pattern = describe_univariate_window(series)
            protos = series_patch_prototypes(series, patch_len=min(8, len(series)), stride=max(1, len(series)//4))

            # branch scores at segment (mean over windows)
            ps = float(np.mean(pred_scores[idx, seg.start:seg.end + 1])) if alpha > 0 else 0.0
            rs = float(np.mean(ae_scores[idx, seg.start:seg.end + 1])) if beta > 0 else 0.0
            gs = float(np.mean(gen_scores[idx, seg.start:seg.end + 1])) if gamma > 0 else 0.0

            # branch score at peak (single window)
            ps_peak = float(pred_scores[idx, peak]) if alpha > 0 else 0.0
            rs_peak = float(ae_scores[idx, peak]) if beta > 0 else 0.0
            gs_peak = float(gen_scores[idx, peak]) if gamma > 0 else 0.0

            # direct reconstruction error (peak window)
            recon_mse = float(np.mean((series_hat - series) ** 2))

            # direct prediction error (peak window) - last n_pred steps
            pred_series = pred[peak, :, idx].astype(np.float64).reshape(-1)
            gt_series = gt[peak, :, idx].astype(np.float64).reshape(-1)
            pred_mse = float(np.mean((pred_series - gt_series) ** 2))

            feature_evidence.append({
                "index": int(idx),
                "name": name,
                "score": float(seg_scores[idx]),
                "contribution": float(abs(seg_scores[idx]) / denom),
                "branch_score_mean": {"pred": ps, "recon": rs, "gen": gs},
                "branch_score_peak": {"pred": ps_peak, "recon": rs_peak, "gen": gs_peak},
                "mse_peak": {"pred": pred_mse, "recon": recon_mse},
                "pattern": {
                    "anomaly_type": pattern.anomaly_type,
                    "direction": pattern.direction,
                    "evidence": pattern.evidence,
                },
                "stats": stats,
                "baseline": {
                    "median": float(baseline_median[idx]),
                    "p05": float(baseline_p05[idx]),
                    "p95": float(baseline_p95[idx]),
                    "z_last_global": float((stats['last'] - baseline_median[idx]) / (1.4826 * baseline_mad[idx] + 1e-8)),
                },
                "shape_tokens": protos,
            })

        segment_info = {
            "start": int(seg.start),
            "end": int(seg.end),
            "peak": int(seg.peak),
            "time_range_in_raw": [int(seg.peak), int(seg.peak + window_size - 1)],
            "threshold": float(threshold) if threshold is not None else None,
        }

        # build prompt
        prompt = build_anomaly_explanation_prompt(
            dataset=dataset_up,
            window_size=window_size,
            n_pred=n_pred,
            segment_info=segment_info,
            feature_evidence=feature_evidence,
            global_evidence=global_evidence,
            language=language,
        )

        # call explainer
        evidence_payload = {
            "dataset": dataset_up,
            "window": {"window_size": window_size, "n_pred": n_pred},
            "segment": segment_info,
            "global": global_evidence,
            "features": feature_evidence,
        }
        explanation_text = explainer.explain(prompt, evidence_payload)

        explanations.append({
            "segment": segment_info,
            "global_evidence": global_evidence,
            "feature_evidence": feature_evidence,
            "prompt": prompt,
            "explanation": explanation_text,
        })

    result = {
        "dataset": dataset_up,
        "num_windows": int(W),
        "num_features": int(F),
        "num_segments": int(len(explanations)),
        "method": method,
        "option": int(option),
        "threshold": float(threshold) if threshold is not None else None,
        "explanations": explanations,
    }

    # save
    if out_json_path:
        _ensure_dir(out_json_path)
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if out_md_path:
        _ensure_dir(out_md_path)
        md_lines: List[str] = []
        md_lines.append(f"# Anomaly Explanations ({dataset_up})")
        md_lines.append(f"- windows: {W}, features: {F}")
        md_lines.append(f"- method: {method}, option: {option}, threshold: {threshold}")
        md_lines.append("")

        for k, item in enumerate(explanations, 1):
            seg = item['segment']
            md_lines.append(f"## Case {k}: window[{seg['start']},{seg['end']}], peak={seg['peak']}")
            md_lines.append("")

            md_lines.append("### LLM / Template Explanation")
            md_lines.append(item['explanation'])
            md_lines.append("")

            md_lines.append("### Top Feature Evidence")
            for fe in item['feature_evidence']:
                md_lines.append(f"- **{fe['name']}** (idx={fe['index']}), contrib={fe['contribution']:.3f}, score={fe['score']:.4f}")
                md_lines.append(f"  - pattern: {fe['pattern']['anomaly_type']} ({fe['pattern']['direction']})")
                md_lines.append(f"  - stats: min={fe['stats']['min']:.3f}, max={fe['stats']['max']:.3f}, last={fe['stats']['last']:.3f}")
                md_lines.append(f"  - shape_tokens: {' '.join(fe.get('shape_tokens', []))}")
            md_lines.append("")

        with open(out_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))

    return result
