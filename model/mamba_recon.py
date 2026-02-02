# mambatsad/models/recon.py
# -*- coding: utf-8 -*-
"""
基于 Mamba 的重构分支模型（时间序列版 MambaAD）。

你当前的“消融切换方式”是通过修改 ABLATION_PRESET 的字符串来选择预置配置；
本文件确保：只要把 ABLATION_PRESET 设置成某个 key，就一定能在 _PRESETS 中找到对应配置并生效。

对外接口保持不变：
- build_recon_model(input_dim: int) -> MambaTSAD
- build_recon_model_from_cfg(input_dim: int, cfg: ReconAblationConfig) -> MambaTSAD
- 模型 forward 输出 dict: {"recon_multi": List[Tensor], "recon": Tensor, (optional) "fusion_weights": Tensor}

新增/补全的消融能力（均为可选，不影响 baseline 行为）：
- Table 5 / HSS-like：1D 多视角 permutation（permute_views）× 双向扫描，模拟 4/8 directions
- Table 7：local_conv_variant 支持 "dwconv_1x1" 与 "only_dwconv"
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
# 1) 消融配置
# ---------------------------
PermuteType = Literal["identity", "evenodd", "chunk_flip", "chunk_zigzag"]
DownsampleMode = Literal["avg", "max"]
FusionMode = Literal["mean", "fixed", "learnable"]
BiDirFusion = Literal["mean", "sum", "fwd", "bwd"]
LocalConvVariant = Literal["dwconv_1x1", "only_dwconv"]


@dataclass(frozen=True)
class ReconAblationConfig:
    # 模型维度
    d_model: int = 256

    # 多尺度：downsample factor 列表（1 表示不下采样）
    scales: Tuple[int, ...] = (1, 2, 4)
    # 每个尺度堆叠多少个 LSSBlockTS / IdentityBlockTS
    num_layers_per_scale: Tuple[int, ...] = (2, 2, 2)

    # LSSBlockTS 内部（全局/局部分支）
    use_global: bool = True
    use_local: bool = True
    kernel_sizes: Tuple[int, ...] = (5, 7)
    local_conv_variant: LocalConvVariant = "dwconv_1x1"
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
    # 多视角：若不为 None，则覆盖 permute_type；用于模拟多方向扫描（Table 5 / HSS-like）
    permute_views: Optional[Tuple[PermuteType, ...]] = None
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
                # chunk_zigzag：奇数块翻转，偶数块不翻
                if ci % 2 == 1:
                    block = torch.flip(block, dims=[0])
            idx_list.append(block)
        idx = torch.cat(idx_list, dim=0)
    else:  # pragma: no cover
        raise ValueError(f"未知 permute_type: {permute_type}")

    idx_inv = torch.argsort(idx)
    return cast(torch.LongTensor, idx.to(dtype=torch.long)), cast(torch.LongTensor, idx_inv.to(dtype=torch.long))


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

    baseline 行为：
      - bidirectional=True, bidir_fusion="mean"
      - permute_views=None（仅使用 permute_type="identity"）
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
        permute_views: Optional[Tuple[PermuteType, ...]] = None,
        permute_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.bidirectional = bool(bidirectional)
        self.bidir_fusion = bidir_fusion
        self.use_ffn = bool(use_ffn)

        self.permute_type = permute_type
        self.permute_views = permute_views
        self.permute_chunk_size = int(permute_chunk_size)

        self._perm_cache: Dict[
            Tuple[int, PermuteType, int, torch.device],
            Tuple[torch.LongTensor, torch.LongTensor],
        ] = {}

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = None
        if self.bidirectional:
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if self.use_ffn:
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

    def _get_perm(self, L: int, device: torch.device, permute_type: PermuteType) -> Tuple[torch.LongTensor, torch.LongTensor]:
        key = (L, permute_type, self.permute_chunk_size, device)
        if key not in self._perm_cache:
            self._perm_cache[key] = _make_permutation_indices(
                L=L, permute_type=permute_type, chunk_size=self.permute_chunk_size, device=device
            )
        return self._perm_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, _C = x.shape
        x_norm = self.norm(x)

        view_types: Tuple[PermuteType, ...]
        if self.permute_views is None or len(self.permute_views) == 0:
            view_types = (self.permute_type,)
        else:
            view_types = self.permute_views

        y_views: List[torch.Tensor] = []
        for vt in view_types:
            if vt != "identity":
                idx, idx_inv = self._get_perm(L, x.device, vt)
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

            y_views.append(y)

        # 多视角融合：默认取均值（不影响 baseline）
        y = torch.stack(y_views, dim=0).mean(dim=0)

        x = x + self.dropout(y)

        if self.ffn is not None:
            x = x + self.dropout(self.ffn(x))

        return x


class LocalConvBlock1D(nn.Module):
    """一维局部卷积模块（LSS 的 Local 分支）：1x1 → DWConv(k) → 1x1。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            # InstanceNorm1d requires more than 1 spatial element in training; GroupNorm doesn't.
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class LocalDWConvOnly1D(nn.Module):
    """消融：仅保留 depthwise conv（去掉 1x1 投影），用于对齐 Table 7 的 Only DWConv。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class IdentityBlockTS(nn.Module):
    """消融：完全跳过该 block。"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LSSBlockTS(nn.Module):
    """LSS（Local & State Space）模块：全局 (Bi)Mamba + 多核局部卷积。"""

    def __init__(
        self,
        d_model: int,
        *,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        kernel_sizes: Sequence[int] = (5, 7),
        local_conv_variant: LocalConvVariant = "dwconv_1x1",
        dropout: float = 0.0,
        use_global: bool = True,
        use_local: bool = True,
        lss_fusion: Literal["concat_proj", "sum"] = "concat_proj",
        lss_residual: bool = True,
        bidirectional: bool = True,
        bidir_fusion: BiDirFusion = "mean",
        use_ffn: bool = True,
        permute_type: PermuteType = "identity",
        permute_views: Optional[Tuple[PermuteType, ...]] = None,
        permute_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if not (use_global or use_local):
            # 完整去掉 LSSBlockTS 请在外层用 IdentityBlockTS
            raise ValueError("use_global 与 use_local 不能同时为 False（请改用 IdentityBlockTS）")
        if use_local and len(kernel_sizes) == 0:
            raise ValueError("use_local=True 时 kernel_sizes 不能为空")

        self.use_global = bool(use_global)
        self.use_local = bool(use_local)
        self.lss_fusion = lss_fusion
        self.lss_residual = bool(lss_residual)

        if self.use_global:
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
                        permute_views=permute_views,
                        permute_chunk_size=permute_chunk_size,
                    )
                    for _ in range(int(num_mamba_layers))
                ]
            )
        else:
            self.global_layers = None

        if self.use_local:
            if local_conv_variant == "dwconv_1x1":
                branch_cls = LocalConvBlock1D
            elif local_conv_variant == "only_dwconv":
                branch_cls = LocalDWConvOnly1D
            else:  # pragma: no cover
                raise ValueError(f"未知 local_conv_variant: {local_conv_variant}")
            self.local_branches = nn.ModuleList([branch_cls(d_model, int(k)) for k in kernel_sizes])
        else:
            self.local_branches = None

        if lss_fusion == "concat_proj":
            out_dim = 0
            if self.use_global:
                out_dim += d_model
            if self.use_local:
                out_dim += d_model * len(kernel_sizes)
            self.proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        elif lss_fusion == "sum":
            # sum: mean over branches
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

    forward 输出：
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
                local_conv_variant="dwconv_1x1",
                lss_fusion="concat_proj",
                lss_residual=True,
                num_mamba_layers=2,
                bidirectional=True,
                bidir_fusion="mean",
                use_ffn=True,
                mamba_d_state=int(d_state),
                mamba_d_conv=int(d_conv),
                mamba_expand=int(expand),
                permute_type="identity",
                permute_views=None,
                permute_chunk_size=16,
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

        # ---- build blocks ----
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
                                local_conv_variant=cfg.local_conv_variant,
                                dropout=cfg.dropout,
                                use_global=cfg.use_global,
                                use_local=cfg.use_local,
                                lss_fusion=cfg.lss_fusion,
                                lss_residual=cfg.lss_residual,
                                bidirectional=cfg.bidirectional,
                                bidir_fusion=cfg.bidir_fusion,
                                use_ffn=cfg.use_ffn,
                                permute_type=cfg.permute_type,
                                permute_views=cfg.permute_views,
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
        # x: [B, L, input_dim]
        _B, L, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"MambaTSAD 期望输入维度为 {self.input_dim}，但得到 {D}")

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
            hs = self.forward_single_scale(xs, blocks_ml)
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
ABLATION_PRESET: str = "baseline"
# 你只需要把上面这行改成 _PRESETS 的某个 key 即可生效。
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
# ABLATION_PRESET = "t2_basic_mamba"
# ABLATION_PRESET = "t2_plus_lss"
# ABLATION_PRESET = "scan_multiview_4"
# ABLATION_PRESET = "scan_multiview_8"
# ABLATION_PRESET = "local_only_dwconv"
# ABLATION_PRESET = "mi2_noskip"
# ABLATION_PRESET = "kernel_1_3"
# ABLATION_PRESET = "kernel_1_3_5_7"
# ABLATION_PRESET = "t2_plus_lss_multiview_4"
# ABLATION_PRESET = "t2_plus_lss_multiview_8"
# ABLATION_PRESET = "only_dwconv_mi3"
# ABLATION_PRESET = "mi1_noskip"
# ABLATION_PRESET = "mi3_noskip"

# ABLATION_PRESET = "d_model_128"
# ABLATION_PRESET = "d_model_384"
# ABLATION_PRESET = "depth_shallow"
# ABLATION_PRESET = "depth_deep"

# ABLATION_PRESET = "common"

# Multi-view recommended combinations (simulate 4/8 directions)
# eff_dirs ~= (#views) * (2 if bidirectional else 1)
_MULTIVIEW_4: Tuple[PermuteType, ...] = ("identity", "chunk_flip")
_MULTIVIEW_8: Tuple[PermuteType, ...] = ("identity", "evenodd", "chunk_flip", "chunk_zigzag")


def _apply_env_overrides(cfg: ReconAblationConfig) -> ReconAblationConfig:
    """Optionally override some ablation knobs via environment variables.

    IMPORTANT (pollution guard):
      - Overrides are ONLY applied when STAMP_RECON_APPLY_ENV_OVERRIDES is truthy.
      - This prevents accidentally changing presets like `baseline` when your shell
        still has STAMP_RECON_* variables set.

    Supported env vars (all optional, but only effective when the guard is enabled):
      - STAMP_RECON_APPLY_ENV_OVERRIDES: 0/1/true/false
      - STAMP_RECON_NUM_MAMBA_LAYERS: int
      - STAMP_RECON_LSS_RESIDUAL: 0/1/true/false
      - STAMP_RECON_LOCAL_CONV_VARIANT: dwconv_1x1 | only_dwconv
      - STAMP_RECON_KERNEL_SIZES: comma-separated ints, e.g. "3,5" or "5". Empty disables local kernels.

    Notes:
      - Invalid values are ignored (so a bad env won't crash training).
      - We return a NEW dataclass instance (ReconAblationConfig is frozen).
    """

    import os

    def _parse_bool(v: str) -> bool | None:
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return None

    # Guard: do nothing unless explicitly enabled
    guard_v = os.getenv("STAMP_RECON_APPLY_ENV_OVERRIDES", "0")
    guard = _parse_bool(guard_v)
    if guard is not True:
        return cfg

    updates: dict[str, object] = {}

    v = os.getenv("STAMP_RECON_NUM_MAMBA_LAYERS")
    if v:
        try:
            updates["num_mamba_layers"] = int(v)
        except Exception:
            pass

    v = os.getenv("STAMP_RECON_LSS_RESIDUAL")
    if v is not None and v != "":
        b = _parse_bool(v)
        if b is not None:
            updates["lss_residual"] = b

    v = os.getenv("STAMP_RECON_LOCAL_CONV_VARIANT")
    if v:
        vv = str(v).strip().lower()
        if vv in {"dwconv_1x1", "only_dwconv"}:
            updates["local_conv_variant"] = vv

    v = os.getenv("STAMP_RECON_KERNEL_SIZES")
    if v is not None and v != "":
        s = str(v).strip()
        if s == "":
            pass
        elif s.lower() in {"none", "null"}:
            updates["kernel_sizes"] = tuple()
        else:
            try:
                ks = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
                updates["kernel_sizes"] = ks
            except Exception:
                pass

    if not updates:
        return cfg

    # Ensure kernel_sizes is a tuple[int,...]
    if "kernel_sizes" in updates and not isinstance(updates["kernel_sizes"], tuple):
        updates["kernel_sizes"] = tuple(updates["kernel_sizes"])  # type: ignore[arg-type]

    return ReconAblationConfig(**{**cfg.__dict__, **updates})


_PRESETS: Dict[str, ReconAblationConfig] = {
    # ---------- baseline ----------
    "baseline": ReconAblationConfig(),

    # ---------- 原有（保留兼容） ----------
    "basic_mamba_single_scale": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_local=False,
        kernel_sizes=(),
    ),
    "ablate_no_local": ReconAblationConfig(
        use_local=False,
        kernel_sizes=(),
    ),
    "ablate_no_global": ReconAblationConfig(
        use_global=False,
        kernel_sizes=(5, 7),
    ),
    "ablate_no_pyramid": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
    ),
    "ablate_unidirectional": ReconAblationConfig(
        bidirectional=False,
        bidir_fusion="fwd",
    ),
    "kernel_5_only": ReconAblationConfig(kernel_sizes=(5,)),
    "kernel_7_only": ReconAblationConfig(kernel_sizes=(7,)),
    "kernel_3_5_7": ReconAblationConfig(kernel_sizes=(3, 5, 7)),
    "mamba_layers_1": ReconAblationConfig(num_mamba_layers=1),
    "mamba_layers_3": ReconAblationConfig(num_mamba_layers=3),
    "fusion_learnable": ReconAblationConfig(fusion="learnable"),
    "scan_evenodd": ReconAblationConfig(permute_type="evenodd"),
    "scan_chunk_zigzag": ReconAblationConfig(permute_type="chunk_zigzag", permute_chunk_size=16),

    # 完整去掉 LSS（全局/局部都关）：用 IdentityBlockTS 走“纯多尺度重构器”路径
    "w_o_lss": ReconAblationConfig(
        use_local=False,
        use_global=False,
        fusion="learnable",
        num_mamba_layers=1,
    ),

    # ---------- 你要对齐 Table 2：Basic → +LSS → +HSS-like ----------
    # Table 2 Row1：Basic Mamba（单尺度、仅全局）
    "t2_basic_mamba": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_global=True,
        use_local=False,
        kernel_sizes=(),
        permute_type="identity",
        permute_views=cast(Optional[Tuple[PermuteType, ...]], None),
    ),
    # Table 2 Row2：+LSS（单尺度、全局+局部）
    "t2_plus_lss": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_global=True,
        use_local=True,
        kernel_sizes=(5, 7),
        permute_type="identity",
        permute_views=cast(Optional[Tuple[PermuteType, ...]], None),
    ),
    # Table 2 Row3：+HSS-like（单尺度、全局+局部、多视角=8 directions 近似）
    "t2_plus_lss_multiview_4": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_global=True,
        use_local=True,
        kernel_sizes=(5, 7),
        permute_views=_MULTIVIEW_4,
    ),
    "t2_plus_lss_multiview_8": ReconAblationConfig(
        scales=(1,),
        num_layers_per_scale=(2,),
        use_global=True,
        use_local=True,
        kernel_sizes=(5, 7),
        permute_views=_MULTIVIEW_8,
    ),

    # ---------- 你要对齐 Table 5：scan / multiview ----------
    "scan_chunk_flip": ReconAblationConfig(permute_type="chunk_flip", permute_chunk_size=16),
    "scan_multiview_4": ReconAblationConfig(permute_views=_MULTIVIEW_4),
    "scan_multiview_8": ReconAblationConfig(permute_views=_MULTIVIEW_8),

    # ---------- 你要对齐 Table 7：Only DWConv / noskip / kernels / Mi ----------
    "local_only_dwconv": ReconAblationConfig(local_conv_variant="only_dwconv"),
    "only_dwconv_mi3": ReconAblationConfig(local_conv_variant="only_dwconv", num_mamba_layers=3),

    "mi1_noskip": ReconAblationConfig(num_mamba_layers=1, lss_residual=False),
    "mi2_noskip": ReconAblationConfig(num_mamba_layers=2, lss_residual=False),
    "mi3_noskip": ReconAblationConfig(num_mamba_layers=3, lss_residual=False),

    "kernel_1_3": ReconAblationConfig(kernel_sizes=(1, 3)),
    "kernel_1_3_5_7": ReconAblationConfig(kernel_sizes=(1, 3, 5, 7)),

    # ---------- 你要对齐 Table 4：容量/深度 ----------
    "d_model_128": ReconAblationConfig(d_model=128),
    "d_model_384": ReconAblationConfig(d_model=384),

    "depth_shallow": ReconAblationConfig(num_layers_per_scale=(1, 1, 1)),
    "depth_deep": ReconAblationConfig(num_layers_per_scale=(3, 3, 3)),
    "common": ReconAblationConfig(
        num_mamba_layers=3,
        lss_residual=True,
        local_conv_variant="dwconv_1x1",
        kernel_sizes=(3, 5)
    ),
}


def get_ablation_config(preset: str) -> ReconAblationConfig:
    if preset not in _PRESETS:
        raise KeyError(f"未知 ABLATION_PRESET='{preset}'。可选：{sorted(_PRESETS.keys())}")
    return _apply_env_overrides(_PRESETS[preset])


# ---------------------------
# 6) 对外工厂函数（保持原接口）
# ---------------------------
def build_recon_model_from_cfg(input_dim: int, cfg: ReconAblationConfig) -> MambaTSAD:
    """显式传入 cfg 的构造方式（推荐在训练脚本里使用）。"""
    return MambaTSAD(input_dim=int(input_dim), cfg=cfg)


def build_recon_model(input_dim: int) -> MambaTSAD:
    """重构分支的默认工厂函数（保持原接口）。

    通过修改 ABLATION_PRESET 选择消融。
    """
    if ABLATION_PRESET is None:  # pragma: no cover
        raise ValueError("ABLATION_PRESET 不能为 None")
    cfg = get_ablation_config(ABLATION_PRESET)
    return build_recon_model_from_cfg(input_dim=input_dim, cfg=cfg)
