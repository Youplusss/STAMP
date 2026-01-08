# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List
import json


_DATASET_DESC_ZH = {
    "SMD": "SMD (Server Machine Dataset) 是多台服务器的多变量监控指标序列，异常通常表现为某些传感器/指标在某段时间内出现突变、漂移或振荡。",
    "MSL": "MSL (Mars Science Laboratory) 是航天器遥测多变量时间序列，异常表示系统状态偏离正常运行模式。",
    "SMAP": "SMAP 是 NASA SMAP 卫星的遥测多变量时间序列，异常表示传感器读数或系统状态异常。",
    "SWAT": "SWaT 是工业水处理系统的多变量传感器序列，异常通常对应攻击或设备故障导致的过程变量变化。",
    "WADI": "WADI 是工业配水系统的多变量传感器序列，异常通常对应攻击或设备故障导致的过程变量变化。",
}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_anomaly_explanation_prompt(
    *,
    dataset: str,
    window_size: int,
    n_pred: int,
    segment_info: Dict[str, Any],
    feature_evidence: List[Dict[str, Any]],
    global_evidence: Dict[str, Any],
    language: str = "zh",
) -> str:
    """Build a Prompt-as-Prefix (PaP) style prompt for anomaly explanation.

    We follow the spirit of Time-LLM:
      - provide dataset context
      - provide task instruction
      - provide input statistics / evidences

    But our downstream task is *explanation* (text generation) rather than forecasting.
    """
    dataset_up = str(dataset).upper()

    def _compact_feature_lines(features: List[Dict[str, Any]]) -> str:
        lines = []
        for i, fe in enumerate(features, 1):
            pat = fe.get('pattern', {}) or {}
            stats = fe.get('stats', {}) or {}
            base = fe.get('baseline', {}) or {}
            mse = fe.get('mse_peak', {}) or {}
            lines.append(
                f"{i}. {fe.get('name','')} (idx={fe.get('index')}, contrib={fe.get('contribution',0):.3f}, score={fe.get('score',0):.4f})"
                f" | pattern={pat.get('anomaly_type','unknown')}({pat.get('direction','unknown')})"
                f" | stats[min={stats.get('min',0):.3f}, max={stats.get('max',0):.3f}, last={stats.get('last',0):.3f}]"
                f" | z_last={base.get('z_last_global',0):.2f}"
                f" | mse[pred={mse.get('pred',0):.4g}, recon={mse.get('recon',0):.4g}]"
            )
        return "\n".join(lines)

    if language.lower().startswith("zh"):
        dataset_desc = _DATASET_DESC_ZH.get(dataset_up, f"数据集 {dataset_up} 的多变量时间序列。")

        instruction = (
            "你是时间序列异常诊断与解释助手。\n"
            "请严格基于【证据】进行解释；不要编造传感器物理含义；不要引用外部资料或链接。\n"
            "输出必须只用中文，并严格包含以下 6 个小节：\n"
            "1) 异常摘要（1-2句）\n"
            "2) 关键变量与证据（按贡献度排序，逐条引用证据）\n"
            "3) 异常形态判断（点异常/区间异常/漂移/振荡等）\n"
            "4) 可能根因（2-4条通用假设，明确不确定性）\n"
            "5) 处置/排查建议（可执行清单）\n"
            "6) 需要补充的数据（若要进一步定位，需要哪些日志/指标）\n"
        )

        seg_text = (
            f"窗口设置: window_size={window_size}, n_pred={n_pred}\n"
            f"异常段: start={segment_info.get('start')}, end={segment_info.get('end')}, peak={segment_info.get('peak')}\n"
            f"阈值: {segment_info.get('threshold')}\n"
        )

        global_text = (
            f"分数聚合: topk={global_evidence.get('score_agg',{}).get('topk')}, agg={global_evidence.get('score_agg',{}).get('topk_agg')}\n"
            f"分支权重: alpha={global_evidence.get('weights',{}).get('alpha')}, beta={global_evidence.get('weights',{}).get('beta')}, gamma={global_evidence.get('weights',{}).get('gamma')}\n"
            f"段内总分: mean={global_evidence.get('segment_score',{}).get('mean'):.4f}, max={global_evidence.get('segment_score',{}).get('max'):.4f}\n"
        )

        evidence_lines = _compact_feature_lines(feature_evidence)

        return (
            f"【数据集】{dataset_desc}\n\n"
            f"【任务】\n{instruction}\n"
            f"【段信息】\n{seg_text}\n"
            f"【全局证据】\n{global_text}\n"
            f"【Top-K变量证据】\n{evidence_lines}\n\n"
            "请开始输出（只输出最终答案，不要复述提示词）。\n"
        )

    # English fallback (keep concise)
    dataset_desc = f"Multivariate time series dataset: {dataset_up}."
    instruction = (
        "You are a time-series anomaly diagnosis assistant.\n"
        "Use ONLY the evidence. Do not invent sensor meanings or cite external sources.\n"
        "Output 6 sections: 1) Summary 2) Key variables 3) Pattern 4) Possible root causes 5) Actions 6) Needed extra data.\n"
    )
    seg_text = f"window_size={window_size}, n_pred={n_pred}, start={segment_info.get('start')}, end={segment_info.get('end')}, peak={segment_info.get('peak')}\n"
    evidence_lines = _compact_feature_lines(feature_evidence)
    return (
        f"[Dataset] {dataset_desc}\n\n"
        f"[Task]\n{instruction}\n"
        f"[Segment]\n{seg_text}\n"
        f"[Top-K Evidence]\n{evidence_lines}\n\n"
        "Return only the final answer.\n"
    )
