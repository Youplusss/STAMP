# mambatsad/models/recon.py
# -*- coding: utf-8 -*-
"""
基于 Mamba 的重构分支模型（时间序列版 MambaAD）。

整体思路：
1. 使用 Mamba（State Space Model）在时间维度上做全局建模，捕获长时间依赖；
2. 结合多核深度可分离卷积做局部模式建模（对应 MambaAD 中的 LSS 模块）；
3. 构建多尺度时间金字塔，在不同时间尺度上进行重构，
   将多尺度重构误差作为异常评分。

该文件对应原仓库中的 mambatsad_ts.py，只是进行了轻微整理与重命名。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 官方 mamba-ssm 库
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "使用 Mamba 重构分支需要先安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


class BiMambaBlock(nn.Module):
    """双向 Mamba Block。

    - 沿时间维做前向 Mamba 扫描；
    - 再对时间反转的序列做一次 Mamba（相当于“反向扫描”）；
    - 将两者平均后加残差；
    - 再接一层前馈网络（FFN）+ 残差。

    这与图像版 MambaAD 中的 HSS 思路一致，只是将二维特征图
    换成了一维时间序列。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # 正向 / 反向两个 Mamba 分支
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # FFN 前馈网络
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x:
            [B, L, C]，B 为 batch，L 为时间长度，C 为通道数。

        返回
        ----
        out:
            与 x 形状相同的输出。
        """
        # ---- Mamba 全局建模 ----
        x_norm = self.norm(x)

        # 正向时间 Mamba
        y_fwd = self.mamba_fwd(x_norm)  # [B, L, C]

        # 反向时间 Mamba：在时间维翻转
        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # 融合双向信息
        y = (y_fwd + y_bwd) / 2.0

        # 残差 1
        x = x + self.dropout(y)

        # ---- FFN 层 ----
        y_ffn = self.ffn(x)
        out = x + self.dropout(y_ffn)
        return out


class LocalConvBlock1D(nn.Module):
    """一维局部卷积模块。

    结构：Conv1d(1x1) -> 深度可分离卷积 -> Conv1d(1x1)

    - groups=channels 的 Conv1d 就是深度卷积，只在时间维度上卷。
    - 对时间序列而言，相当于在局部时间窗口上建模形状模式。
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            # 1x1 卷积做通道间线性变换
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            # 深度可分离卷积（仅在时间维卷积）
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
            ),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] -> 经过 1D 卷积后仍为 [B, L, C]
        """
        x_perm = x.transpose(1, 2)  # [B, C, L]
        y = self.block(x_perm)
        return y.transpose(1, 2)  # [B, L, C]


class LSSBlockTS(nn.Module):
    """时间序列版 LSS 模块（Locality-Enhanced State Space）。

    - Global 分支：若干层 BiMambaBlock，用于长序列建模；
    - Local 分支：两个不同 kernel_size 的 LocalConvBlock1D，用于局部形状建模；
    - 输出：拼接 Global + Local 分支后，通过线性层降回 d_model 并加残差。
    """

    def __init__(
        self,
        d_model: int,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        kernel_sizes=(5, 7),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Global 分支：多层 BiMambaBlock
        self.global_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_mamba_layers)
            ]
        )

        # Local 分支：多种卷积核
        self.local_branches = nn.ModuleList(
            [LocalConvBlock1D(d_model, k) for k in kernel_sizes]
        )

        out_dim = d_model * (1 + len(kernel_sizes))
        self.proj = nn.Linear(out_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        """
        # Global 分支
        g = x
        for layer in self.global_layers:
            g = layer(g)

        # Local 分支
        locals_out = [branch(x) for branch in self.local_branches]

        # 拼接 Global + Local
        concat = torch.cat([g] + locals_out, dim=-1)
        out = self.proj(concat)

        # 残差
        return out + x


class MambaTSAD(nn.Module):
    """整体重构分支模型。

    输入：
        x: [B, L, D_in] 多维时间序列窗口。

    结构：
        1. 线性编码到 d_model 维度；
        2. 构建三层时间金字塔：
           - Level1: 原分辨率 L
           - Level2: 下采样 2 倍 (L/2)
           - Level3: 下采样 4 倍 (L/4)
           各层均堆叠若干 LSSBlockTS；
        3. 每一层解码回原始通道数 D_in，并上采样回 L；
        4. 将多尺度重构结果简单平均，得到最终重构。

    输出：
        {
            "recon_multi": [rec1, rec2, rec3],  # 各尺度重构
            "recon": recon,                     # 多尺度平均重构
        }
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers=(2, 2, 2),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 编码器：输入维度 -> d_model
        self.encoder = nn.Linear(input_dim, d_model)

        # 三个尺度上的 LSS 堆叠
        self.level1 = nn.ModuleList(
            [
                LSSBlockTS(
                    d_model=d_model,
                    num_mamba_layers=2,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers[0])
            ]
        )
        self.level2 = nn.ModuleList(
            [
                LSSBlockTS(
                    d_model=d_model,
                    num_mamba_layers=2,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers[1])
            ]
        )
        self.level3 = nn.ModuleList(
            [
                LSSBlockTS(
                    d_model=d_model,
                    num_mamba_layers=2,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers[2])
            ]
        )

        # 解码器：d_model -> 原始通道数
        self.decoder = nn.Linear(d_model, input_dim)

    def forward_single_scale(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        """在单个尺度上串行通过若干 LSSBlockTS。"""
        out = x
        for blk in blocks:
            out = blk(out)
        return out

    def forward(self, x: torch.Tensor):
        """
        参数
        ----
        x:
            [B, L, D_in]

        返回
        ----
        out_dict:
            包含多尺度重构结果的字典。
        """
        B, L, D = x.shape
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSAD 期望输入维度为 {self.input_dim}，但得到 {D}"
            )

        # 编码到 d_model
        x_embed = self.encoder(x)  # [B, L, d_model]

        # ----- Level 1: 原始分辨率 -----
        l1 = self.forward_single_scale(x_embed, self.level1)  # [B, L, d_model]

        # ----- Level 2: 下采样 2 倍 -----
        x2 = F.avg_pool1d(x_embed.transpose(1, 2), kernel_size=2, stride=2)
        l2 = self.forward_single_scale(x2.transpose(1, 2), self.level2)  # [B, L/2, d_model]

        # ----- Level 3: 下采样 4 倍 -----
        x3 = F.avg_pool1d(x_embed.transpose(1, 2), kernel_size=4, stride=4)
        l3 = self.forward_single_scale(x3.transpose(1, 2), self.level3)  # [B, L/4, d_model]

        # ----- 解码并上采样回原分辨率 -----
        rec1 = self.decoder(l1)  # [B, L, D_in]

        l2_up = F.interpolate(
            l2.transpose(1, 2), size=L, mode="linear", align_corners=False
        ).transpose(1, 2)
        rec2 = self.decoder(l2_up)

        l3_up = F.interpolate(
            l3.transpose(1, 2), size=L, mode="linear", align_corners=False
        ).transpose(1, 2)
        rec3 = self.decoder(l3_up)

        # 多尺度重构的简单平均
        recon = (rec1 + rec2 + rec3) / 3.0

        return {
            "recon_multi": [rec1, rec2, rec3],
            "recon": recon,
        }


def build_recon_model(input_dim: int) -> MambaTSAD:
    """重构分支的默认工厂函数，方便在外部统一调用。"""
    model = MambaTSAD(
        input_dim=input_dim,
        d_model=256,
        num_layers=(2, 2, 2),
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    )
    return model
