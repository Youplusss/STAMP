"""Wrappers to plug TimeLLMForecast into the STAMP pipeline.

STAMP's Trainer/Tester expect:
  - pred_model(x, mas=None) -> y_hat with shape [B, n_pred, N, C]
  - (optionally) ae_model(...) handled elsewhere

This file provides:
  - STAMPTimeLLMPredictor: reshape/flatten logic + optional MAS fusion
  - build_stamp_llm_predictor(args): convenience builder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.llm_time_llm import TimeLLMForecast, TimeLLMTSADConfig


class STAMPTimeLLMPredictor(nn.Module):
    """A STAMP-compatible predictor using a frozen LLM (Time-LLM style)."""

    def __init__(
        self,
        *,
        nnodes: int,
        in_channels: int,
        out_channels: int,
        window_size: int,
        n_pred: int,
        dataset_name: str,
        is_mas: bool,
        llm_use_mas: bool,
        llm_cfg: TimeLLMTSADConfig,
        feature_mixer: str = 'none',
        mixer_rank: int = 64,
    ):
        super().__init__()

        self.nnodes = int(nnodes)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.window_size = int(window_size)
        self.n_pred = int(n_pred)
        self.context_len = int(window_size - n_pred)
        self.is_mas = bool(is_mas)
        self.llm_use_mas = bool(llm_use_mas)

        if self.in_channels != self.out_channels:
            raise ValueError(
                f"STAMPTimeLLMPredictor requires in_channels==out_channels for concat window generation, "
                f"got {self.in_channels} vs {self.out_channels}."
            )

        # MAS has 4 channels in this repo's dataloaders
        self.mas_channels = 4

        self.raw_dim = self.nnodes * self.out_channels
        self.total_dim = self.raw_dim
        if self.is_mas and self.llm_use_mas:
            self.total_dim += self.nnodes * self.mas_channels

        # patch/seq config must match context_len
        llm_cfg.seq_len = self.context_len
        llm_cfg.pred_len = self.n_pred
        llm_cfg.enc_in = self.total_dim
        llm_cfg.dataset_name = dataset_name

        self.llm_core = TimeLLMForecast(llm_cfg)

        # Optional cross-feature mixing before patching.
        # Motivation: Time-LLM's core is channel-independent; a lightweight mixer
        # can inject cross-sensor correlations without changing downstream code.
        fm = (feature_mixer or 'none').lower()
        if fm in ('none', 'identity'):
            self.feature_mixer = nn.Identity()
            self.use_mixer = False
        elif fm in ('mlp', 'lowrank', 'mix'):
            r = int(mixer_rank)
            r = max(1, min(r, self.total_dim))
            self.feature_mixer = nn.Sequential(
                nn.Linear(self.total_dim, r),
                nn.GELU(),
                nn.Linear(r, self.total_dim),
            )
            self.use_mixer = True
        else:
            raise ValueError(f"Unknown feature_mixer: {feature_mixer}")

    def forward(self, x: torch.Tensor, mas: Optional[torch.Tensor] = None):
        """Forward.

        Args:
            x:   [B, context_len, N, C]
            mas: [B, context_len, N, 4] or None

        Returns:
            y_hat: [B, n_pred, N, C]
        """

        if x.dim() != 4:
            raise ValueError(f"Expect x [B,T,N,C], got {tuple(x.shape)}")
        B, T, N, C = x.shape
        if T != self.context_len:
            raise ValueError(f"context_len mismatch: expect {self.context_len}, got {T}")
        if N != self.nnodes or C != self.out_channels:
            raise ValueError(
                f"shape mismatch: model expects N={self.nnodes}, C={self.out_channels}, got N={N}, C={C}"
            )

        x_flat = x.reshape(B, T, self.raw_dim)

        if self.is_mas and self.llm_use_mas:
            if mas is None:
                raise ValueError("llm_use_mas=True but mas is None. Set --is_mas False or provide MAS loader.")
            if mas.dim() != 4:
                raise ValueError(f"Expect mas [B,T,N,4], got {tuple(mas.shape)}")
            mas_flat = mas.reshape(B, T, self.nnodes * self.mas_channels)
            x_in = torch.cat([x_flat, mas_flat], dim=-1)
        else:
            x_in = x_flat

        if self.use_mixer:
            x_in = x_in + self.feature_mixer(x_in)

        y_hat = self.llm_core(x_in)  # [B, n_pred, total_dim]
        y_hat = y_hat[:, :, : self.raw_dim]
        y_hat = y_hat.reshape(B, self.n_pred, self.nnodes, self.out_channels)
        return y_hat


def build_stamp_llm_predictor(args) -> STAMPTimeLLMPredictor:
    """Factory used by run.py / test.py."""

    # Map CLI args -> TimeLLMTSADConfig
    llm_cfg = TimeLLMTSADConfig(
        # will be overwritten by wrapper to match context_len
        seq_len=max(1, int(args.window_size - args.n_pred)),
        pred_len=int(args.n_pred),
        enc_in=int(args.nnodes * args.out_channels),
        patch_len=int(getattr(args, "llm_patch_len", 4)),
        stride=int(getattr(args, "llm_stride", 2)),
        d_model=int(getattr(args, "llm_d_model", 32)),
        d_ff=int(getattr(args, "llm_d_ff", 32)),
        n_heads=int(getattr(args, "llm_n_heads", 4)),
        dropout=float(getattr(args, "llm_dropout", 0.1)),
        llm_model=str(getattr(args, "llm_model", "gpt2")),
        llm_backend=str(getattr(args, "llm_backend", "gpt2")),
        llm_layers=int(getattr(args, "llm_layers", 6)),
        llm_pretrained=bool(getattr(args, "llm_pretrained", True)),
        llm_grad_ckpt=bool(getattr(args, "llm_grad_ckpt", False)),
        llm_load_in_4bit=bool(getattr(args, "llm_load_in_4bit", False)),
        llm_load_in_8bit=bool(getattr(args, "llm_load_in_8bit", False)),
        llm_dtype=str(getattr(args, "llm_dtype", "auto")),
        # HF cache controls
        hf_cache_dir=getattr(args, 'hf_cache_dir', None),
        hf_local_files_only=bool(getattr(args, 'hf_local_files_only', False)),
        llm_use_cache=bool(getattr(args, 'llm_use_cache', False)),
        prompt_mode=str(getattr(args, "llm_prompt_mode", "stats_short")),
        prompt_root=str(getattr(args, "llm_prompt_root", "expe/prompt_bank")),
        dataset_name=str(getattr(args, "data", "SWaT")),
        prompt_domain=str(getattr(args, "llm_prompt_domain", "anomaly_detection")),
        top_k_lags=int(getattr(args, "llm_top_k_lags", 5)),
        head_dropout=float(getattr(args, "llm_head_dropout", 0.0)),
        pred_activation=str(getattr(args, "llm_pred_activation", "none")),
    )

    return STAMPTimeLLMPredictor(
        nnodes=args.nnodes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        window_size=args.window_size,
        n_pred=args.n_pred,
        dataset_name=str(args.data),
        is_mas=bool(getattr(args, "is_mas", False)),
        llm_use_mas=bool(getattr(args, "llm_use_mas", False)),
        llm_cfg=llm_cfg,
        feature_mixer=str(getattr(args, 'llm_feature_mixer', 'none')),
        mixer_rank=int(getattr(args, 'llm_mixer_rank', 64)),
    )
