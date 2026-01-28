# -*- coding: utf-8 -*-
"""STAMP x Mamba adapters.

本文件提供 **包装器（wrapper/adaptor）**，用于把你给的 Mamba 重构模型（MambaTSAD）
和 Mamba 预测模型（MambaTSADForecast）无缝接入 STAMP 的训练/测试管线。

设计目标
---------
- 尽量不改动 STAMP 原始训练/测试逻辑（trainer.py/test.py 只做很小改动）；
- 预测分支接口保持一致：
    - 输入:  x  [B, Lc, N, C]
    - 输出: y  [B, n_pred, N, out_channels]
- 重构分支接口保持一致：
    - 输入:  x_flat [B, T*N*C]
    - 输出: recon_flat [B, T*N*C]

多尺度 MA（MAS）
-----------------
如果 args.is_mas=True，原 STAMP 会额外提供 mas: [B, Lc, N, 4]。
本包装器支持将 raw 与 mas 在 token 维拼接，使 Forecast Mamba 在“变量维”
同时建模 raw 与 multi-scale moving average。

用法
----
在 run.py / test.py 中：
- 当 args.pred_model == 'mamba' 时，用 STAMPForecastMamba 作为预测分支；
- 当 args.recon_model == 'mamba' 时，用 STAMPReconMamba 作为重构分支。

注意
----
这些模型依赖 mamba-ssm：
    pip install mamba-ssm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn

from model.mamba_forecast import build_forecast_model
from model.mamba_recon import ABLATION_PRESET, build_recon_model, get_ablation_config


def _parse_int_tuple(s: str, expected_len: int = 3) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"期望 {expected_len} 个整数，用逗号分隔，例如 '2,2,2'，但得到: {s}")
    return tuple(int(p) for p in parts)


def set_requires_grad(model: nn.Module, flag: bool) -> None:
    """在对抗/耦合训练中用于冻结另一分支的参数，减少无用梯度与显存占用。"""
    for p in model.parameters():
        p.requires_grad_(flag)


@dataclass
class ForecastMambaConfig:
    d_model: int = 256
    e_layers: int = 3
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    use_norm: bool = True
    use_last_residual: bool = True


@dataclass
class ReconMambaConfig:
    d_model: int = 256
    num_layers: Tuple[int, int, int] = (2, 2, 2)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    output_activation: str = "auto"  # 'auto' | 'none' | 'sigmoid' | 'tanh'


class STAMPForecastMamba(nn.Module):
    """把 MambaTSADForecast 适配到 STAMP 预测分支接口。

    输入
    ----
    x:   [B, Lc, N, C] (通常 C=1)
    mas: [B, Lc, N, 4] (当 is_mas=True)

    输出
    ----
    y: [B, n_pred, N, out_channels]

    说明
    ----
    - MambaTSADForecast 期望输入 [B, Lc, D]；这里把 (N, C) 展平为 D。
    - 若启用 use_mas，则把 raw 与 mas 在最后一维拼接：每个 node 变成 (C+4) 个 token。
    - Forecast 输出 [B, n_pred, D_total]；我们只取 raw token（每个 node 的前 out_channels 个）
      做监督与 anomaly score。
    """

    def __init__(
        self,
        nnodes: int,
        in_channels: int,
        out_channels: int,
        context_len: int,
        pred_len: int,
        *,
        use_mas: bool = True,
        mas_channels: int = 4,
        cfg: Optional[ForecastMambaConfig] = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg = ForecastMambaConfig()

        self.nnodes = int(nnodes)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)

        if self.out_channels > self.in_channels:
            # STAMP 默认 in_channels==out_channels==1；如果你要扩展到多通道，建议保持 out<=in。
            raise ValueError(
                f"out_channels={out_channels} > in_channels={in_channels}，"
                "raw token 选择逻辑不明确，请保证 out_channels <= in_channels。"
            )

        self.use_mas = bool(use_mas)
        self.mas_channels = int(mas_channels)

        # 每个 node 对应多少个 token
        self.tokens_per_node = self.in_channels + (self.mas_channels if self.use_mas else 0)

        # Forecast 模型的 input_dim = 变量数 D_total
        self.input_dim = self.nnodes * self.tokens_per_node

        # 使用带 preset 的工厂函数（默认 preset/base 不改变原行为）
        # 允许用户通过 model/mamba_forecast.py 的 FORECAST_ABLATION_PRESET 或环境变量切换消融
        self.core = build_forecast_model(
            input_dim=self.input_dim,
            seq_len=self.context_len,
            pred_len=self.pred_len,
            # overrides (keep old args-driven hyperparams)
            d_model=cfg.d_model,
            e_layers=cfg.e_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            use_norm=cfg.use_norm,
            use_last_residual=cfg.use_last_residual,
        )

        # raw token 索引：每个 node 的前 out_channels 个
        raw_indices = []
        for n in range(self.nnodes):
            base = n * self.tokens_per_node
            for c in range(self.out_channels):
                raw_indices.append(base + c)
        raw_idx = torch.tensor(raw_indices, dtype=torch.long)
        self.register_buffer("raw_indices", raw_idx, persistent=False)

    def forward(self, x: torch.Tensor, idx=None, mas: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, Lc, N, C]
        if x.dim() != 4:
            raise ValueError(f"STAMPForecastMamba 期望 x 形状 [B,L,N,C]，但得到 {x.shape}")
        B, L, N, C = x.shape
        if N != self.nnodes:
            raise ValueError(f"STAMPForecastMamba 配置 nnodes={self.nnodes}，但输入 N={N}")
        if L != self.context_len:
            raise ValueError(f"STAMPForecastMamba 配置 context_len={self.context_len}，但输入 L={L}")
        if C != self.in_channels:
            raise ValueError(f"STAMPForecastMamba 配置 in_channels={self.in_channels}，但输入 C={C}")

        if self.use_mas:
            if mas is None:
                raise ValueError("启用了 use_mas=True，但 forward 未传入 mas")
            if mas.dim() != 4:
                raise ValueError(f"mas 期望形状 [B,L,N,4]，但得到 {mas.shape}")
            Bm, Lm, Nm, Cm = mas.shape
            if (Bm != B) or (Lm != L) or (Nm != N) or (Cm != self.mas_channels):
                raise ValueError(
                    f"mas 形状不匹配，期望 [{B},{L},{N},{self.mas_channels}]，但得到 {mas.shape}"
                )
            x_all = torch.cat([x, mas], dim=-1)  # [B,L,N,tokens_per_node]
            x_in = x_all.reshape(B, L, N * self.tokens_per_node)
        else:
            x_in = x.reshape(B, L, N * C)

        y_pred_all = self.core(x_in)  # [B, pred_len, D_total]

        raw_idx = self.raw_indices.to(y_pred_all.device)
        y_raw = y_pred_all.index_select(dim=2, index=raw_idx)  # [B, pred_len, N*out_channels]
        y_raw = y_raw.view(B, self.pred_len, self.nnodes, self.out_channels)
        return y_raw


class STAMPReconMamba(nn.Module):
    """把 MambaTSAD 适配到 STAMP 重构分支接口。

    STAMP 原 AE 以展平向量作为输入：
        x_flat: [B, T*N*C]

    MambaTSAD 期望：
        x_seq:  [B, T, D]，D=N*C

    该 wrapper 在 forward 内部完成 reshape，并返回展平后的重构向量。
    """

    def __init__(
        self,
        window_size: int,
        nnodes: int,
        in_channels: int,
        *,
        cfg: Optional[ReconMambaConfig] = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = ReconMambaConfig()

        self.window_size = int(window_size)
        self.nnodes = int(nnodes)
        self.in_channels = int(in_channels)

        self.input_dim = self.nnodes * self.in_channels  # D
        self.ae_channels = self.window_size * self.input_dim

        # 关键：让重构分支实际使用 mamba_recon.py 的 preset/cfg（通过注释切换）
        self.core = build_recon_model(input_dim=self.input_dim)

        # 可选：打印当前消融配置，方便确认“注释切换”确实生效
        import os
        if str(os.getenv("STAMP_PRINT_RECON_CFG", "0")).lower() in {"1", "true", "yes"}:
            print("[STAMPReconMamba] ABLATION_PRESET=", ABLATION_PRESET)
            print("[STAMPReconMamba] recon_cfg=", get_ablation_config(ABLATION_PRESET))

        act = cfg.output_activation.lower()
        if act not in {"auto", "none", "sigmoid", "tanh"}:
            raise ValueError("output_activation 仅支持 auto/none/sigmoid/tanh")
        # auto: 默认根据数据缩放选择；若未在 build 函数里解析，这里退化为 sigmoid（适用于 MinMaxScaler 到 [0,1] 的场景）
        self.output_activation = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容两种输入：
        # 1) x_flat: [B, T*N*C]
        # 2) x_win:  [B, T, N, C]
        if x.dim() == 4:
            B, T, N, C = x.shape
            if (T != self.window_size) or (N != self.nnodes) or (C != self.in_channels):
                raise ValueError(
                    f"STAMPReconMamba 输入 [B,T,N,C] 与配置不一致："
                    f"期望 T={self.window_size}, N={self.nnodes}, C={self.in_channels}，但得到 {x.shape}"
                )
            x_flat = x.reshape(B, -1)
        elif x.dim() == 2:
            x_flat = x
            B, D = x_flat.shape
            if D != self.ae_channels:
                raise ValueError(f"STAMPReconMamba 期望展平维度 {self.ae_channels}，但得到 {D}")
        else:
            raise ValueError(f"STAMPReconMamba 期望输入 [B,T,N,C] 或 [B,T*N*C]，但得到 {x.shape}")

        x_seq = x_flat.view(-1, self.window_size, self.input_dim)  # [B,T,D]

        out = self.core(x_seq)
        recon = out["recon"]  # [B,T,D]

        if self.output_activation in {"sigmoid", "auto"}:
            recon = torch.sigmoid(recon)
        elif self.output_activation == "tanh":
            recon = torch.tanh(recon)

        recon_flat = recon.reshape(-1, self.ae_channels)
        return recon_flat


def build_stamp_mamba_models(args) -> tuple[nn.Module, nn.Module]:
    """根据 args 构建 (pred_model, ae_model) 的 Mamba 版本。"""

    f_cfg = ForecastMambaConfig(
        d_model=getattr(args, "mamba_d_model", 256),
        e_layers=getattr(args, "mamba_e_layers", 3),
        d_state=getattr(args, "mamba_d_state", 16),
        d_conv=getattr(args, "mamba_d_conv", 4),
        expand=getattr(args, "mamba_expand", 2),
        dropout=getattr(args, "mamba_dropout", 0.1),
        use_norm=getattr(args, "mamba_use_norm", True),
        use_last_residual=getattr(args, "mamba_use_last_residual", True),
    )
    use_mas = bool(getattr(args, "mamba_use_mas", True)) and bool(getattr(args, "is_mas", False))

    pred_model = STAMPForecastMamba(
        nnodes=args.nnodes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        context_len=args.window_size - args.n_pred,
        pred_len=args.n_pred,
        use_mas=use_mas,
        mas_channels=4,
        cfg=f_cfg,
    )

    recon_layers_str = getattr(args, "recon_num_layers", "2,2,2")
    # --- 关键修复：输出激活/范围约束 ---
    # 由于 STAMP 的各数据集 dataloader 默认使用 MinMaxScaler，把数据缩放到 [0,1]。
    # 若重构分支输出不做约束，在对抗项 ( -adv_loss ) 作用下非常容易把输出推到极大，导致 MSE 爆炸。
    out_act = getattr(args, "recon_output_activation", "auto")
    if out_act is None:
        out_act = "auto"
    out_act = str(out_act).lower()
    if out_act == "auto":
        # real_value=True 表示在原始数值空间训练/评估，此时不应强行 sigmoid
        out_act = "none" if bool(getattr(args, "real_value", False)) else "sigmoid"

    r_cfg = ReconMambaConfig(
        d_model=getattr(args, "recon_d_model", 256),
        num_layers=cast(Tuple[int, int, int], _parse_int_tuple(recon_layers_str, expected_len=3)),
        d_state=getattr(args, "recon_d_state", 16),
        d_conv=getattr(args, "recon_d_conv", 4),
        expand=getattr(args, "recon_expand", 2),
        dropout=getattr(args, "recon_dropout", 0.1),
        output_activation=out_act,
    )

    ae_model = STAMPReconMamba(
        window_size=args.window_size,
        nnodes=args.nnodes,
        in_channels=args.in_channels,
        cfg=r_cfg,
    )

    return pred_model, ae_model


__all__ = [
    "ForecastMambaConfig",
    "ReconMambaConfig",
    "STAMPForecastMamba",
    "STAMPReconMamba",
    "build_stamp_mamba_models",
    "set_requires_grad",
]
