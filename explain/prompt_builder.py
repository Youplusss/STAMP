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
    if language.lower().startswith("zh"):
        dataset_desc = _DATASET_DESC_ZH.get(dataset_up, f"数据集 {dataset_up} 的多变量时间序列。")
        instruction = (
            "你是一名时间序列异常诊断与解释助手。\n"
            "请基于【证据】对检测到的异常做出解释。\n\n"
            "硬性要求（非常重要）：\n"
            "- 必须【只用中文】输出；不要输出英文、不要引用外部资料、不要给链接、不要出现与证据无关的内容；\n"
            "- 只能使用证据中的信息，不要编造传感器物理含义；\n"
            "- 必须点名 Top-K 关键变量（按贡献度排序），并引用对应的‘异常类型’与‘统计证据’；\n"
            "- ‘可能原因’只能给出通用假设，并明确说明需要结合业务/系统确认；\n"
            "- 输出尽量简洁、结构化，避免长段落。\n\n"
            "输出格式（严格按序号）：\n"
            "1) 异常摘要（1-2 句）\n"
            "2) 关键变量与证据（Top-K 列表，每项一行）\n"
            "3) 异常形态判断（点异常/区间异常/漂移/振荡等）\n"
            "4) 可能原因（2-4 条通用假设，标明不确定性）\n"
            "5) 建议检查项（可操作的检查清单）\n"
        )

        prompt = (
            f"### 数据集背景\n{dataset_desc}\n\n"
            f"### 任务\n{instruction}\n\n"
            f"### 窗口设置\nwindow_size={window_size}, n_pred={n_pred}\n\n"
            f"### 异常段信息\n{_json_dumps(segment_info)}\n\n"
            f"### 全局证据\n{_json_dumps(global_evidence)}\n\n"
            f"### 关键变量证据（Top-K）\n{_json_dumps(feature_evidence)}\n\n"
            "### 请开始生成解释（只用中文）\n"
        )
        return prompt

    # English fallback
    dataset_desc = f"Multivariate time series dataset: {dataset_up}."
    instruction = (
        "You are a time-series anomaly diagnosis assistant.\n"
        "Explain the detected anomaly based ONLY on the provided evidence.\n"
        "Do NOT invent sensor meanings.\n"
        "Output sections: (1) Summary, (2) Key variables & evidence, (3) Pattern type, (4) Possible causes (generic), (5) Checks.\n"
    )
    prompt = (
        f"### Dataset\n{dataset_desc}\n\n"
        f"### Task\n{instruction}\n\n"
        f"### Window\nwindow_size={window_size}, n_pred={n_pred}\n\n"
        f"### Segment\n{_json_dumps(segment_info)}\n\n"
        f"### Global evidence\n{_json_dumps(global_evidence)}\n\n"
        f"### Feature evidence (Top-K)\n{_json_dumps(feature_evidence)}\n\n"
        "### Generate explanation\n"
    )
    return prompt
