# mambatsad/models/forecast.py
# -*- coding: utf-8 -*-
"""
基于 Mamba 的时间序列预测分支（TSAD 预测模型）。

整体思路：
- 参考 S-D-Mamba（Is Mamba Effective for Time Series Forecasting?）中的 "倒置嵌入" 结构，
  将每个变量视作一个 token；
- 对每个变量上一段时间窗口进行线性投影得到 token 表征，再在变量维上用 Mamba
  做序列建模；
- 使用多步预测误差作为异常评分，类似 TranAD、MTAD-GAT 等预测式异常检测模型；
- 本文件对应原仓库中的 mambatsad_pred.py。

本次改动要点：
- 增加 use_last_residual：在归一化空间中以“最后一个时间步”为基线做残差预测，
  Mamba 负责学习残差部分，有利于稳定训练并提升异常敏感度。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "使用预测分支模型需要先安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


# =============================================================================
# 0) Forecast ablation presets（只改一行变量或设置环境变量即可切换）
# =============================================================================
# 用法：
# 1) 直接改这一行：FORECAST_ABLATION_PRESET = "base" / "w_o_vc" / ...
# 2) 或者设置环境变量：MAMBA_FORECAST_ABLATION_PRESET=w_o_vc
# 3) 或者外部调用 build_forecast_model(..., preset="w_o_vc")
FORECAST_ABLATION_PRESET: str = "td_uni_mamba"

AblationPreset = Literal[
    "base",
    "w_o_vc",
    "vc_uni_mamba",
    "vc_attention",
    "w_o_td",
    "td_linear",
    "td_uni_mamba",
    "td_bi_mamba",
    "td_attention",
    "w_o_norm",
    "w_o_last_residual",
    "var_reverse",
    "var_shuffle",
    "uni_mamba"
]


@dataclass(frozen=True)
class ForecastAblationConfig:
    vc_mode: Literal["bi_mamba", "uni_mamba", "attention", "none"] = "bi_mamba"
    td_mode: Literal[
        "ffn",
        "linear",
        "uni_mamba",
        "bi_mamba",
        "attention",
        "none",
    ] = "ffn"
    use_norm: bool = True
    use_last_residual: bool = True
    var_permute: Literal["none", "reverse", "shuffle"] = "none"
    var_permute_seed: int = 0
    attn_heads: int = 4


FORECAST_ABLATION_PRESETS: Dict[str, ForecastAblationConfig] = {
    "base": ForecastAblationConfig(),
    "w_o_vc": ForecastAblationConfig(vc_mode="none"),
    "vc_uni_mamba": ForecastAblationConfig(vc_mode="uni_mamba"),
    "vc_attention": ForecastAblationConfig(vc_mode="attention", attn_heads=4),
    "w_o_td": ForecastAblationConfig(td_mode="none"),
    "td_linear": ForecastAblationConfig(td_mode="linear"),
    "td_uni_mamba": ForecastAblationConfig(td_mode="uni_mamba"),
    "td_bi_mamba": ForecastAblationConfig(td_mode="bi_mamba"),
    "td_attention": ForecastAblationConfig(td_mode="attention", attn_heads=4),
    "w_o_norm": ForecastAblationConfig(use_norm=False),
    "w_o_last_residual": ForecastAblationConfig(use_last_residual=False),
    "var_reverse": ForecastAblationConfig(var_permute="reverse"),
    "var_shuffle": ForecastAblationConfig(var_permute="shuffle", var_permute_seed=0),
    "uni_mamba": ForecastAblationConfig(vc_mode="uni_mamba", td_mode="uni_mamba"),
}


def get_forecast_ablation_config(preset: Optional[str] = None) -> ForecastAblationConfig:
    """解析 preset -> ForecastAblationConfig。

    回退版本约束：不考虑 preset=None 的特殊语义。

    优先级：
      1) 显式传入 preset
      2) 环境变量 MAMBA_FORECAST_ABLATION_PRESET
      3) 本文件 FORECAST_ABLATION_PRESET
    """
    if preset is not None:
        final = str(preset)
    else:
        env = os.getenv("MAMBA_FORECAST_ABLATION_PRESET", "").strip()
        final = env if env else FORECAST_ABLATION_PRESET

    if final not in FORECAST_ABLATION_PRESETS:
        valid = ", ".join(sorted(FORECAST_ABLATION_PRESETS.keys()))
        raise KeyError(f"未知 FORECAST_ABLATION_PRESET={final!r}，可选：{valid}")

    return FORECAST_ABLATION_PRESETS[final]


class InvertedTimeEmbedding(nn.Module):
    """
    简单版“倒置时间嵌入”。

    输入:
        x: [B, L, D]，其中 L 为时间长度，D 为变量数。

    做法:
        - 先把 x 转置为 [B, D, L]，把每个变量的一维时间序列视作一个 token；
        - 再通过线性层将长度 L 的时间序列投影到 d_model 维度；
        - 得到 [B, D, d_model] 的 token 表示。
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"InvertedTimeEmbedding 期望输入形状为 [B, L, D]，实际为 {x.shape}"
            )
        B, L, D = x.shape
        if L != self.seq_len:
            raise ValueError(
                f"InvertedTimeEmbedding 配置的 seq_len={self.seq_len}，"
                f"但当前 batch 的时间长度为 {L}，请确保 win_size - pred_len 与 seq_len 一致。"
            )

        # [B, L, D] -> [B, D, L]
        x_perm = x.permute(0, 2, 1).contiguous()
        x_flat = x_perm.view(B * D, L)  # [B*D, L]

        emb_flat = self.proj(x_flat)  # [B*D, d_model]
        emb = emb_flat.view(B, D, self.d_model)  # [B, D, d_model]
        emb = self.dropout(emb)
        return emb


class _VCEncoder(nn.Module):
    def __init__(
        self,
        mode: Literal["bi_mamba", "uni_mamba", "attention", "none"],
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if mode == "bi_mamba":
            self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.attn = None
        elif mode == "uni_mamba":
            self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = None
            self.attn = None
        elif mode == "attention":
            if d_model % attn_heads != 0:
                raise ValueError(f"d_model={d_model} 必须能被 attn_heads={attn_heads} 整除")
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mamba_fwd = None
            self.mamba_bwd = None
        elif mode == "none":
            self.mamba_fwd = None
            self.mamba_bwd = None
            self.attn = None
        else:  # pragma: no cover
            raise ValueError(f"未知 vc_mode={mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        if self.mode == "bi_mamba":
            y_f = self.mamba_fwd(z)
            y_b = self.mamba_bwd(z.flip(dims=[1])).flip(dims=[1])
            x = x + self.dropout(0.5 * (y_f + y_b))
            return x
        if self.mode == "uni_mamba":
            x = x + self.dropout(self.mamba_fwd(z))
            return x
        if self.mode == "attention":
            y, _ = self.attn(z, z, z, need_weights=False)
            x = x + self.dropout(y)
            return x
        return x


class _TDEncoder(nn.Module):
    def __init__(
        self,
        mode: Literal["ffn", "linear", "uni_mamba", "bi_mamba", "attention", "none"],
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.dropout = nn.Dropout(dropout)

        if mode in ("ffn", "linear"):
            self.norm = nn.LayerNorm(d_model)
            if mode == "ffn":
                self.fc1 = nn.Linear(d_model, d_model * 4)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(d_model * 4, d_model)
            else:
                self.fc1 = nn.Linear(d_model, d_model)
                self.act = None
                self.fc2 = None
            self.mamba_fwd = None
            self.mamba_bwd = None
            self.attn = None
        elif mode in ("uni_mamba", "bi_mamba"):
            self.norm = nn.LayerNorm(d_model)
            self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = None
            if mode == "bi_mamba":
                self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.attn = None
            self.fc1 = None
            self.fc2 = None
            self.act = None
        elif mode == "attention":
            self.norm = nn.LayerNorm(d_model)
            if d_model % attn_heads != 0:
                raise ValueError(f"d_model={d_model} 必须能被 attn_heads={attn_heads} 整除")
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mamba_fwd = None
            self.mamba_bwd = None
            self.fc1 = None
            self.fc2 = None
            self.act = None
        elif mode == "none":
            self.norm = None
            self.fc1 = None
            self.fc2 = None
            self.act = None
            self.mamba_fwd = None
            self.mamba_bwd = None
            self.attn = None
        else:  # pragma: no cover
            raise ValueError(f"未知 td_mode={mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        z = self.norm(x)
        if self.mode == "ffn":
            y = self.fc2(self.dropout(self.act(self.fc1(z))))
            return x + self.dropout(y)
        if self.mode == "linear":
            y = self.fc1(z)
            return x + self.dropout(y)
        if self.mode in ("uni_mamba", "bi_mamba"):
            y_f = self.mamba_fwd(z)
            if self.mamba_bwd is not None:
                y_b = self.mamba_bwd(z.flip(dims=[1])).flip(dims=[1])
                y = 0.5 * (y_f + y_b)
            else:
                y = y_f
            return x + self.dropout(y)
        if self.mode == "attention":
            y, _ = self.attn(z, z, z, need_weights=False)
            return x + self.dropout(y)
        return x


class ForecastBlock1D(nn.Module):
    """一个可消融的 Block：VC + TD。"""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        vc_mode: Literal["bi_mamba", "uni_mamba", "attention", "none"] = "bi_mamba",
        td_mode: Literal["ffn", "linear", "uni_mamba", "bi_mamba", "attention", "none"] = "ffn",
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vc = _VCEncoder(
            mode=vc_mode,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            attn_heads=attn_heads,
        )
        self.td = _TDEncoder(
            mode=td_mode,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            attn_heads=attn_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vc(x)
        x = self.td(x)
        return x


class MambaTSADForecast(nn.Module):
    """
    基于 Mamba 的时间序列预测模型（TSAD 预测分支）。

    输入:
        x: [B, L_c, D]，L_c = context_len = win_size - pred_len

    输出:
        y_pred: [B, pred_len, D]

    预测流程：
    1. 对每个样本在时间维上做样本内归一化（可选），减弱不同变量的尺度差异；
    2. 使用 InvertedTimeEmbedding 将时间维映射为 token 表示，得到 [B, D, d_model]；
    3. 在“变量维”上通过若干层 BiMambaBlock1D 建模变量间依赖；
    4. 通过 Linear 投影得到每个变量在 pred_len 个时间步上的预测（残差）；
    5. （可选）以“最后一个时间步”为基线做残差预测，再反归一化回原始尺度。
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        e_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
        use_last_residual: bool = True,
        # --- ablation knobs (default keeps old behavior) ---
        vc_mode: Literal["bi_mamba", "uni_mamba", "attention", "none"] = "bi_mamba",
        td_mode: Literal["ffn", "linear", "uni_mamba", "bi_mamba", "attention", "none"] = "ffn",
        attn_heads: int = 4,
        var_permute: Literal["none", "reverse", "shuffle"] = "none",
        var_permute_seed: int = 0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.use_norm = use_norm
        # 是否在归一化空间中以“最后一个时间步”为基线做残差预测
        self.use_last_residual = use_last_residual

        self.vc_mode = vc_mode
        self.td_mode = td_mode
        self.attn_heads = int(attn_heads)
        self.var_permute = var_permute
        self.var_permute_seed = int(var_permute_seed)

        # 倒置时间嵌入：时间维 -> token 特征
        self.enc_embedding = InvertedTimeEmbedding(
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
        )

        # 在变量维上堆叠多层 VC+TD blocks（默认相当于 bi-mamba + ffn）
        self.encoder_layers = nn.ModuleList(
            [
                ForecastBlock1D(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    vc_mode=vc_mode,
                    td_mode=td_mode,
                    attn_heads=attn_heads,
                )
                for _ in range(e_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # 预测头：每个变量 token -> pred_len 个时间步的残差
        self.projector = nn.Linear(d_model, pred_len, bias=True)

        # variable permutation indices
        self.register_buffer("_perm_idx", torch.arange(self.input_dim), persistent=False)
        self.register_buffer("_inv_perm_idx", torch.arange(self.input_dim), persistent=False)
        self._init_var_permutation()

    def _init_var_permutation(self) -> None:
        if self.var_permute == "none":
            idx = torch.arange(self.input_dim)
        elif self.var_permute == "reverse":
            idx = torch.arange(self.input_dim - 1, -1, -1)
        elif self.var_permute == "shuffle":
            g = torch.Generator()
            g.manual_seed(self.var_permute_seed)
            idx = torch.randperm(self.input_dim, generator=g)
        else:  # pragma: no cover
            raise ValueError(f"未知 var_permute={self.var_permute}")

        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(self.input_dim)
        self._perm_idx = idx
        self._inv_perm_idx = inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x: [B, L_c, D]，L_c 必须等于初始化时的 seq_len

        返回
        ----
        y_pred: [B, pred_len, D]
        """
        if x.dim() != 3:
            raise ValueError(
                f"MambaTSADForecast 期望输入形状为 [B, L_c, D]，实际为 {x.shape}"
            )
        B, L, D = x.shape
        if L != self.seq_len:
            raise ValueError(
                f"MambaTSADForecast 配置的 seq_len={self.seq_len}，"
                f"但当前输入长度为 {L}。请检查 --win_size 与 --pred_len 的设置。"
            )
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSADForecast 配置的 input_dim={self.input_dim}，"
                f"但当前输入通道数为 {D}。"
            )

        # ------------ （可选）样本内归一化 ------------
        # 这里在时间维上做均值 / 方差归一化，保证每个变量在本窗口内
        # 的数值尺度大致一致，有利于 Mamba 稳定训练。
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()  # [B, 1, D]
            x_centered = x - means
            var = torch.var(x_centered, dim=1, keepdim=True, unbiased=False)
            stdev = torch.sqrt(var + 1e-5)
            x_norm = x_centered / stdev
        else:
            means = None
            stdev = None
            x_norm = x

        # ------------ 倒置嵌入：时间 -> token ------------
        # x_norm: [B, L_c, D] -> tokens: [B, D, d_model]
        tokens = self.enc_embedding(x_norm)

        if self.var_permute != "none":
            tokens = tokens.index_select(dim=1, index=self._perm_idx)

        # ------------ 在“变量序列”上堆叠 Mamba ------------
        h = tokens  # [B, D, d_model]
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.enc_norm(h)  # [B, D, d_model]

        # ------------ 预测头：变量 token -> pred_len 个时间步 ------------
        y = self.projector(h)  # [B, D, pred_len]
        y = y.permute(0, 2, 1).contiguous()  # [B, pred_len, D]

        # ------------ 残差预测：相对于最后一个时间步 ------------
        if self.use_last_residual:
            if self.use_norm:
                # 在归一化空间中使用最后一个时间步作为 baseline
                last_step = x_norm[:, -1:, :]  # [B, 1, D]
            else:
                # 若未归一化，则直接用原始尺度的最后一个时间步
                last_step = x[:, -1:, :]  # [B, 1, D]

            # baseline: [B, pred_len, D]
            baseline = last_step.repeat(1, self.pred_len, 1)
            # 让 Mamba 只学习相对于 baseline 的残差，常规模式更稳定
            y = y + baseline

        # ------------ 反归一化到原始尺度 ------------
        if self.use_norm:
            y = y * stdev[:, 0:1, :] + means[:, 0:1, :]

        if self.var_permute != "none":
            y = y.index_select(dim=2, index=self._inv_perm_idx)

        return y


def build_forecast_model_from_cfg(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    cfg: ForecastAblationConfig,
    **overrides,
) -> MambaTSADForecast:
    base_kwargs = dict(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=256,
        e_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_norm=cfg.use_norm,
        use_last_residual=cfg.use_last_residual,
        vc_mode=cfg.vc_mode,
        td_mode=cfg.td_mode,
        attn_heads=cfg.attn_heads,
        var_permute=cfg.var_permute,
        var_permute_seed=cfg.var_permute_seed,
    )
    base_kwargs.update(overrides)
    return MambaTSADForecast(**base_kwargs)


def build_forecast_model(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    preset: Optional[str] = None,
    **overrides,
) -> MambaTSADForecast:
    """预测分支工厂函数（兼容旧接口 + 支持消融 preset）。"""
    cfg = get_forecast_ablation_config(preset)
    return build_forecast_model_from_cfg(
        input_dim=int(input_dim),
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        cfg=cfg,
        **overrides,
    )
