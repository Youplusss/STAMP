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

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, cast

try:
    # 官方 mamba-ssm 库
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "使用 Mamba 重构分支需要先安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


# ---------------------------
# 1) 消融配置（保持默认行为=原 mamba_recon.py）
# ---------------------------
PermuteType = Literal["identity", "evenodd", "chunk_flip", "chunk_zigzag"]
DownsampleMode = Literal["avg", "max"]
FusionMode = Literal["mean", "fixed", "learnable"]
BiDirFusion = Literal["mean", "sum", "fwd", "bwd"]


@dataclass(frozen=True)
class ReconAblationConfig:
    # 模型维度
    d_model: int = 256

    # 多尺度：downsample factor 列表（1 表示不下采样）
    scales: Tuple[int, ...] = (1, 2, 4)
    # 每个尺度堆叠多少个 LSSBlockTS
    num_layers_per_scale: Tuple[int, ...] = (2, 2, 2)

    # LSSBlockTS 内部（全局/局部分支）
    use_global: bool = True
    use_local: bool = True
    kernel_sizes: Tuple[int, ...] = (5, 7)
    lss_fusion: Literal["concat_proj", "sum"] = "concat_proj"
    lss_residual: bool = True

    # 全局分支（Mamba）参数
    num_mamba_layers: int = 2
    bidirectional: bool = True
    bidir_fusion: BiDirFusion = "mean"
    use_ffn: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # 扫描路径近似：对 1D 序列做可逆重排
    permute_type: PermuteType = "identity"
    permute_chunk_size: int = 16

    # 训练常用
    dropout: float = 0.1

    # 下/上采样
    downsample_mode: DownsampleMode = "avg"
    upsample_mode: Literal["linear", "nearest"] = "linear"

    # 是否 pad 到 factor 的倍数（避免尾部丢点导致维度不一致）
    pad_to_multiple: bool = False

    # 多尺度融合
    fusion: FusionMode = "mean"
    fusion_weights: Optional[Tuple[float, ...]] = None


# ---------------------------
# 2) 工具函数：可逆重排 / padding / 下采样 / 上采样
# ---------------------------

def _make_permutation_indices(
    L: int,
    permute_type: PermuteType,
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """构建长度为 L 的可逆重排 idx 与其逆 idx_inv。"""
    if permute_type == "identity":
        idx = torch.arange(L, device=device)
    elif permute_type == "evenodd":
        even = torch.arange(0, L, 2, device=device)
        odd = torch.arange(1, L, 2, device=device)
        idx = torch.cat([even, odd], dim=0)
    elif permute_type in ("chunk_flip", "chunk_zigzag"):
        if chunk_size <= 0:
            raise ValueError("permute_chunk_size 必须 > 0")
        idx_list: List[torch.Tensor] = []
        n_chunks = (L + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, L)
            block = torch.arange(s, e, device=device)
            if permute_type == "chunk_flip":
                block = torch.flip(block, dims=[0])
            else:
                if ci % 2 == 1:
                    block = torch.flip(block, dims=[0])
            idx_list.append(block)
        idx = torch.cat(idx_list, dim=0)
    else:  # pragma: no cover
        raise ValueError(f"未知 permute_type: {permute_type}")

    idx_inv = torch.argsort(idx)
    idx_l = cast(torch.LongTensor, idx.to(dtype=torch.long))
    idx_inv_l = cast(torch.LongTensor, idx_inv.to(dtype=torch.long))
    return idx_l, idx_inv_l


def _pad_to_multiple_1d(
    x: torch.Tensor,
    multiple: int,
    pad_mode: str = "replicate",
) -> Tuple[torch.Tensor, int]:
    """将序列长度 padding 到 multiple 的倍数（只在尾部 padding）。返回 (padded_x, pad_len)。

    x: [B, L, C]
    """
    if multiple <= 1:
        return x, 0
    _B, L, _C = x.shape
    r = L % multiple
    if r == 0:
        return x, 0
    pad_len = multiple - r
    x_bcL = x.transpose(1, 2)
    x_bcL = F.pad(x_bcL, (0, pad_len), mode=pad_mode)
    return x_bcL.transpose(1, 2), pad_len


def _downsample_1d(
    x: torch.Tensor,
    factor: int,
    mode: DownsampleMode = "avg",
    *,
    pad_to_multiple: bool = False,
) -> Tuple[torch.Tensor, int]:
    """1D 下采样（pool），返回 (x_ds, pad_len)。

    x: [B, L, C]
    """
    if factor == 1:
        return x, 0
    if factor <= 0:
        raise ValueError("downsample factor 必须为正整数")
    if pad_to_multiple:
        x, pad_len = _pad_to_multiple_1d(x, factor, pad_mode="replicate")
    else:
        pad_len = 0

    x_bcL = x.transpose(1, 2)
    if mode == "avg":
        y = F.avg_pool1d(x_bcL, kernel_size=factor, stride=factor)
    elif mode == "max":
        y = F.max_pool1d(x_bcL, kernel_size=factor, stride=factor)
    else:  # pragma: no cover
        raise ValueError(f"未知 downsample_mode: {mode}")
    return y.transpose(1, 2), pad_len


def _upsample_1d(
    x: torch.Tensor,
    target_L: int,
    mode: Literal["linear", "nearest"] = "linear",
) -> torch.Tensor:
    """1D 上采样回 target_L。

    x: [B, L', C]
    """
    x_bcL = x.transpose(1, 2)
    y = F.interpolate(
        x_bcL,
        size=target_L,
        mode=mode,
        align_corners=False if mode == "linear" else None,
    )
    return y.transpose(1, 2)


# ---------------------------
# 3) 基础模块：BiMambaBlock / LocalConvBlock1D / LSSBlockTS
# ---------------------------
class BiMambaBlock(nn.Module):
    """（可选双向）Mamba Block（时间序列）。

    默认行为等价于原 mamba_recon.py：bidirectional=True + bidir_fusion="mean" + use_ffn=True + permute_type="identity"。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        *,
        bidirectional: bool = True,
        bidir_fusion: BiDirFusion = "mean",
        use_ffn: bool = True,
        permute_type: PermuteType = "identity",
        permute_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.bidirectional = bidirectional
        self.bidir_fusion = bidir_fusion
        self.use_ffn = use_ffn

        self.permute_type = permute_type
        self.permute_chunk_size = permute_chunk_size
        self._perm_cache: Dict[
            Tuple[int, str, int, torch.device],
            Tuple[torch.LongTensor, torch.LongTensor],
        ] = {}

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = None
        if bidirectional:
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        else:
            self.ffn = None

        self.dropout = nn.Dropout(dropout)

    def _get_perm(self, L: int, device: torch.device) -> Tuple[torch.LongTensor, torch.LongTensor]:
        key = (L, self.permute_type, self.permute_chunk_size, device)
        if key not in self._perm_cache:
            idx, idx_inv = _make_permutation_indices(
                L=L,
                permute_type=self.permute_type,
                chunk_size=self.permute_chunk_size,
                device=device,
            )
            self._perm_cache[key] = (idx, idx_inv)
        return self._perm_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _C = x.shape
        x_norm = self.norm(x)

        # 扫描路径近似：可逆重排
        if self.permute_type != "identity":
            idx, idx_inv = self._get_perm(L, x.device)
            x_scan = x_norm.index_select(dim=1, index=idx)
        else:
            x_scan = x_norm
            idx_inv = None

        y_fwd = self.mamba_fwd(x_scan)

        if self.bidirectional:
            assert self.mamba_bwd is not None
            x_rev = torch.flip(x_scan, dims=[1])
            y_bwd = self.mamba_bwd(x_rev)
            y_bwd = torch.flip(y_bwd, dims=[1])

            if self.bidir_fusion == "mean":
                y = (y_fwd + y_bwd) / 2.0
            elif self.bidir_fusion == "sum":
                y = y_fwd + y_bwd
            elif self.bidir_fusion == "fwd":
                y = y_fwd
            elif self.bidir_fusion == "bwd":
                y = y_bwd
            else:  # pragma: no cover
                raise ValueError(f"未知 bidir_fusion: {self.bidir_fusion}")
        else:
            y = y_fwd

        if idx_inv is not None:
            y = y.index_select(dim=1, index=idx_inv)

        x = x + self.dropout(y)

        if self.ffn is not None:
            y_ffn = self.ffn(x)
            x = x + self.dropout(y_ffn)

        return x


class LocalConvBlock1D(nn.Module):
    """一维局部卷积模块（LSS 的 Local 分支）。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class IdentityBlockTS(nn.Module):
    """Ablation helper: identity block for fully removing LSSBlockTS computation."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LSSBlockTS(nn.Module):
    """LSS（Local & State Space）模块：全局 (Bi)Mamba + 多核局部卷积。"""

    def __init__(
        self,
        d_model: int,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        kernel_sizes: Sequence[int] = (5, 7),
        dropout: float = 0.0,
        *,
        use_global: bool = True,
        use_local: bool = True,
        lss_fusion: Literal["concat_proj", "sum"] = "concat_proj",
        lss_residual: bool = True,
        bidirectional: bool = True,
        bidir_fusion: BiDirFusion = "mean",
        use_ffn: bool = True,
        permute_type: PermuteType = "identity",
        permute_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if not (use_global or use_local):
            # This module requires at least one branch.
            # For full LSS ablation, use IdentityBlockTS at the caller side.
            raise ValueError("use_global 与 use_local 不能同时为 False")
        if use_local and len(kernel_sizes) == 0:
            raise ValueError("use_local=True 时 kernel_sizes 不能为空")

        self.use_global = use_global
        self.use_local = use_local
        self.lss_fusion = lss_fusion
        self.lss_residual = lss_residual

        if use_global:
            self.global_layers = nn.ModuleList(
                [
                    BiMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        bidir_fusion=bidir_fusion,
                        use_ffn=use_ffn,
                        permute_type=permute_type,
                        permute_chunk_size=permute_chunk_size,
                    )
                    for _ in range(int(num_mamba_layers))
                ]
            )
        else:
            self.global_layers = None

        if use_local:
            self.local_branches = nn.ModuleList(
                [LocalConvBlock1D(d_model, int(k)) for k in kernel_sizes]
            )
        else:
            self.local_branches = None

        if lss_fusion == "concat_proj":
            out_dim = 0
            if use_global:
                out_dim += d_model
            if use_local:
                out_dim += d_model * len(kernel_sizes)
            self.proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        elif lss_fusion == "sum":
            self.proj = nn.Identity()
        else:  # pragma: no cover
            raise ValueError(f"未知 lss_fusion: {lss_fusion}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        if self.use_global:
            assert self.global_layers is not None
            g = x
            for layer in self.global_layers:
                g = layer(g)
            feats.append(g)

        if self.use_local:
            assert self.local_branches is not None
            feats.extend([branch(x) for branch in self.local_branches])

        if self.lss_fusion == "concat_proj":
            out = self.proj(torch.cat(feats, dim=-1))
        else:
            out = self.proj(torch.stack(feats, dim=0).mean(dim=0))

        return out + x if self.lss_residual else out


# ---------------------------
# 4) 主模型：MambaTSAD（多尺度重构 + 消融）
# ---------------------------
class MambaTSAD(nn.Module):
    """整体重构分支模型（兼容消融配置）。

    输出：
      - recon_multi: List[Tensor]，每个尺度的重构（都已上采样回 L）
      - recon: 融合后的最终重构
      - fusion_weights: (可选) learnable/fixed 融合权重
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
        *,
        cfg: Optional[ReconAblationConfig] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        # 兼容旧构造方式：若未提供 cfg，则用旧参数构建一个等价 cfg
        if cfg is None:
            cfg = ReconAblationConfig(
                d_model=int(d_model),
                scales=(1, 2, 4),
                num_layers_per_scale=(int(num_layers[0]), int(num_layers[1]), int(num_layers[2])),
                use_global=True,
                use_local=True,
                kernel_sizes=(5, 7),
                lss_fusion="concat_proj",
                lss_residual=True,
                num_mamba_layers=2,
                bidirectional=True,
                bidir_fusion="mean",
                use_ffn=True,
                mamba_d_state=int(d_state),
                mamba_d_conv=int(d_conv),
                mamba_expand=int(expand),
                dropout=float(dropout),
                downsample_mode="avg",
                upsample_mode="linear",
                pad_to_multiple=False,
                fusion="mean",
            )
        self.cfg = cfg

        if len(cfg.scales) != len(cfg.num_layers_per_scale):
            raise ValueError("scales 与 num_layers_per_scale 长度必须一致")
        if any(int(s) <= 0 for s in cfg.scales):
            raise ValueError("scales 中的下采样因子必须为正整数")
        if cfg.fusion == "fixed":
            if cfg.fusion_weights is None:
                raise ValueError('fusion="fixed" 时必须提供 fusion_weights')
            if len(cfg.fusion_weights) != len(cfg.scales):
                raise ValueError("fusion_weights 长度必须与 scales 一致")

        self.encoder = nn.Linear(self.input_dim, cfg.d_model)

        # ---- build blocks (LSSBlockTS or Identity for w_o_lss) ----
        ablate_lss = (not cfg.use_global) and (not cfg.use_local)

        levels: List[nn.ModuleList] = []
        for n_blocks in cfg.num_layers_per_scale:
            if ablate_lss:
                levels.append(nn.ModuleList([IdentityBlockTS() for _ in range(int(n_blocks))]))
            else:
                levels.append(
                    nn.ModuleList(
                        [
                            LSSBlockTS(
                                d_model=cfg.d_model,
                                num_mamba_layers=cfg.num_mamba_layers,
                                d_state=cfg.mamba_d_state,
                                d_conv=cfg.mamba_d_conv,
                                expand=cfg.mamba_expand,
                                kernel_sizes=cfg.kernel_sizes,
                                dropout=cfg.dropout,
                                use_global=cfg.use_global,
                                use_local=cfg.use_local,
                                lss_fusion=cfg.lss_fusion,
                                lss_residual=cfg.lss_residual,
                                bidirectional=cfg.bidirectional,
                                bidir_fusion=cfg.bidir_fusion,
                                use_ffn=cfg.use_ffn,
                                permute_type=cfg.permute_type,
                                permute_chunk_size=cfg.permute_chunk_size,
                            )
                            for _ in range(int(n_blocks))
                        ]
                    )
                )
        self.levels: nn.ModuleList = nn.ModuleList(levels)

        self.decoder = nn.Linear(cfg.d_model, self.input_dim)

        if cfg.fusion == "learnable":
            self.fusion_logits = nn.Parameter(torch.zeros(len(cfg.scales), dtype=torch.float32))
        else:
            self.fusion_logits = None

        if cfg.fusion == "fixed" and cfg.fusion_weights is not None:
            w = torch.tensor(cfg.fusion_weights, dtype=torch.float32)
            w = w / (w.sum() + 1e-8)
            self.register_buffer("fusion_w_fixed", w, persistent=False)
        else:
            self.fusion_w_fixed = None

    @staticmethod
    def forward_single_scale(x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        out = x
        for blk in blocks:
            out = blk(out)
        return out

    def _get_fusion_weights(self) -> Optional[torch.Tensor]:
        if self.cfg.fusion == "mean":
            return None
        if self.cfg.fusion == "learnable":
            assert self.fusion_logits is not None
            return torch.softmax(self.fusion_logits, dim=0)
        if self.cfg.fusion == "fixed":
            assert self.fusion_w_fixed is not None
            return self.fusion_w_fixed
        raise ValueError(f"未知 fusion: {self.cfg.fusion}")  # pragma: no cover

    def forward(self, x: torch.Tensor):
        _B, L, D = x.shape
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSAD 期望输入维度为 {self.input_dim}，但得到 {D}"
            )

        x_embed = self.encoder(x)

        recon_multi: List[torch.Tensor] = []

        for scale, blocks in zip(self.cfg.scales, self.levels):
            scale = int(scale)
            blocks_ml = cast(nn.ModuleList, blocks)
            if scale == 1:
                xs = x_embed
            else:
                xs, _pad_len = _downsample_1d(
                    x_embed,
                    factor=scale,
                    mode=self.cfg.downsample_mode,
                    pad_to_multiple=self.cfg.pad_to_multiple,
                )
            hs = self.forward_single_scale(xs, blocks_ml)  # type: ignore[arg-type]
            if scale != 1:
                hs = _upsample_1d(hs, target_L=L, mode=self.cfg.upsample_mode)
            rec = self.decoder(hs)
            recon_multi.append(rec)

        w = self._get_fusion_weights()
        if w is None:
            recon = torch.stack(recon_multi, dim=0).mean(dim=0)
        else:
            S = len(recon_multi)
            wv = w.view(S, 1, 1, 1).to(recon_multi[0].dtype)
            recon = (torch.stack(recon_multi, dim=0) * wv).sum(dim=0)

        out = {"recon_multi": recon_multi, "recon": recon}
        if w is not None:
            out["fusion_weights"] = w.detach()
        return out


# ---------------------------
# 5) 预置消融方案（只改一行字符串即可切换）
# ---------------------------
# 说明：
# - 这是把 mamba_recon_ablation.py 的“注释/取消注释切换 preset”方式迁到这里。
# - 默认 baseline 的行为应与历史 mamba_recon.py 一致。
# 你可以通过“注释/取消注释”快速切换：
ABLATION_PRESET: str = "baseline"
# ABLATION_PRESET = "basic_mamba_single_scale"
# ABLATION_PRESET = "ablate_no_local"
# ABLATION_PRESET = "ablate_no_global"
# ABLATION_PRESET = "ablate_no_pyramid"
# ABLATION_PRESET = "ablate_unidirectional"
# ABLATION_PRESET = "kernel_5_only"
# ABLATION_PRESET = "kernel_7_only"
# ABLATION_PRESET = "kernel_3_5_7"
# ABLATION_PRESET = "mamba_layers_1"
# ABLATION_PRESET = "mamba_layers_3"
# ABLATION_PRESET = "fusion_learnable"
# ABLATION_PRESET = "scan_evenodd"
# ABLATION_PRESET = "scan_chunk_zigzag"
# ABLATION_PRESET = "w_o_lss"


_PRESETS: Dict[str, ReconAblationConfig] = {
    # 与历史 mamba_recon.py 保持一致的默认配置
    "baseline": ReconAblationConfig(),

    # —— 论文“Basic Mamba”风格：单尺度 + 仅全局（保留 BiMamba 以对齐“2-direction sweep”）
    "basic_mamba_single_scale": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_local=False,
        kernel_sizes=(),
    ),

    # —— 去局部分支（对应 LSS ablation: w/o local）
    "ablate_no_local": ReconAblationConfig(
        use_local=False,
        kernel_sizes=(),
    ),

    # —— 去全局分支（对应 LSS ablation: w/o global）
    "ablate_no_global": ReconAblationConfig(
        use_global=False,
        bidirectional=True,  # 不生效，但保持语义清晰
        kernel_sizes=(5, 7),
    ),

    # —— 去多尺度金字塔（只保留 Level1）
    "ablate_no_pyramid": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
    ),

    # —— 单向扫描（Bidirectional -> Unidirectional）
    "ablate_unidirectional": ReconAblationConfig(
        bidirectional=False,
        bidir_fusion="fwd",
    ),

    # —— 局部核大小 ablation（单核/多核）
    "kernel_5_only": ReconAblationConfig(kernel_sizes=(5,)),
    "kernel_7_only": ReconAblationConfig(kernel_sizes=(7,)),
    "kernel_3_5_7": ReconAblationConfig(kernel_sizes=(3, 5, 7)),

    # —— HSS 堆叠深度 ablation（这里对应每个 LSSBlockTS 内的 Mamba 层数）
    "mamba_layers_1": ReconAblationConfig(num_mamba_layers=1),
    "mamba_layers_3": ReconAblationConfig(num_mamba_layers=3),

    # —— 多尺度融合策略 ablation：learnable vs mean
    "fusion_learnable": ReconAblationConfig(fusion="learnable"),

    # —— “扫描路径”近似：不同可逆重排（类比 scan_type）
    "scan_evenodd": ReconAblationConfig(permute_type="evenodd"),
    "scan_chunk_zigzag": ReconAblationConfig(permute_type="chunk_zigzag", permute_chunk_size=16),
    "w_o_lss": ReconAblationConfig(
        use_local=False,
        use_global=False,
        fusion="learnable",
        num_mamba_layers=1

    ),
}


def get_ablation_config(preset: str) -> ReconAblationConfig:
    if preset not in _PRESETS:
        raise KeyError(
            f"未知 ABLATION_PRESET='{preset}'。可选：{list(_PRESETS.keys())}"
        )
    return _PRESETS[preset]


# ---------------------------
# 6) 对外工厂函数（保持原接口 + 提供增强接口）
# ---------------------------

def build_recon_model_from_cfg(input_dim: int, cfg: ReconAblationConfig) -> MambaTSAD:
    """显式传入 cfg 的构造方式（推荐在训练脚本里使用）。"""
    return MambaTSAD(input_dim=int(input_dim), cfg=cfg)


def build_recon_model(input_dim: int) -> MambaTSAD:
    """重构分支的默认工厂函数（保持原接口）。

    - 你可以通过修改/注释 ABLATION_PRESET 来切换消融方案。
    - 回退版本不考虑 ABLATION_PRESET=None。
    """
    if ABLATION_PRESET is None:  # pragma: no cover
        raise ValueError("ABLATION_PRESET 不能为 None")
    cfg = get_ablation_config(ABLATION_PRESET)
    return build_recon_model_from_cfg(input_dim=input_dim, cfg=cfg)
