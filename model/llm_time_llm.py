"""TimeLLM-style core for multivariate time-series forecasting (used for TSAD).

This module implements the core idea from:
  TIME-LLM: Time Series Forecasting by Reprogramming Large Language Models

Adaptation goals for this repository (LLM-TSAD):
  - Provide a lightweight, self-contained implementation that can be plugged into
    the STAMP training/evaluation pipeline.
  - Keep the LLM frozen; train only small reprogramming + heads.
  - Support multivariate inputs (many sensors/features).
  - Provide prompt ablations (none / dataset / stats / stats_short).

NOTE
----
The original Time-LLM repo organizes code across multiple files. For ease of
integration, we inline the essential parts here (PatchEmbedding, Normalize,
ReprogrammingLayer, etc.) and keep the rest of the STAMP code unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# -------------------------- small utilities --------------------------


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _infer_prompt_path(dataset_name: str, prompt_root: str) -> str:
    """Return best-effort prompt file path."""
    # allow both "SWaT" and "SWAT" casing
    cand = [
        os.path.join(prompt_root, f"{dataset_name}.txt"),
        os.path.join(prompt_root, f"{dataset_name.lower()}.txt"),
        os.path.join(prompt_root, f"{dataset_name.upper()}.txt"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    # fallback
    generic = os.path.join(prompt_root, "generic.txt")
    if os.path.exists(generic):
        return generic
    return cand[0]


def _debug_enabled() -> bool:
    return str(os.environ.get("LLM_TSAD_DEBUG_PROMPT", "0")).strip().lower() in {"1", "true", "yes", "y"}


# -------------------------- normalization (RevIN-like) --------------------------


class Normalize(nn.Module):
    """Standardization over time dimension.

    This is the same idea used in Time-LLM (layers/StandardNorm.py).
    It stores per-sample mean/std so that outputs can be denormalized.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        # cached stats
        self._mean: Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in {"norm", "denorm"}:
            raise ValueError(f"Normalize mode must be 'norm' or 'denorm', got: {mode}")
        if mode == "norm":
            # x: [B, T, C]
            mean = x.mean(dim=1, keepdim=True).detach()
            var = torch.var(x, dim=1, keepdim=True, unbiased=False)
            stdev = torch.sqrt(var + self.eps).detach()
            self._mean = mean
            self._stdev = stdev
            x = (x - mean) / stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        # denorm
        if self._mean is None or self._stdev is None:
            raise RuntimeError("Normalize.denorm called before norm")
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self._stdev + self._mean


# -------------------------- patch embedding --------------------------


class TokenEmbedding(nn.Module):
    """Conv1d token embedding for a patch."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, patch_nums, patch_len]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding: Tuple[int, int]):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return torch.nn.functional.pad(x, self.padding, mode="replicate")


class PatchEmbedding(nn.Module):
    """Patchify each variable along time and embed patches."""

    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding_patch_layer = ReplicationPad1d((0, self.stride))
        self.value_embedding = TokenEmbedding(self.patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Args:
        x: [B, N, T]
        Returns:
          enc_out: [(B*N), patch_nums, d_model]
          n_vars: N
        """
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, N, patch_nums, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


# -------------------------- reprogramming layer --------------------------


class ReprogrammingLayer(nn.Module):
    """Align time-series token space to LLM embedding space.

    This is essentially a cross-attention where:
      Query  <- time-series tokens
      Key/Val<- (compressed) LLM token embeddings
    """

    def __init__(self, d_model: int, n_heads: int, d_keys: Optional[int] = None, d_llm: Optional[int] = None, attn_dropout: float = 0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_llm = d_llm or d_model
        self.n_heads = int(n_heads)
        self.d_keys = int(d_keys)
        self.d_llm = int(d_llm)

        self.query_projection = nn.Linear(d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(self.d_llm, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(self.d_llm, self.d_keys * self.n_heads)
        self.out_projection = nn.Linear(self.d_keys * self.n_heads, self.d_llm)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, target_embedding: torch.Tensor, source_embedding: torch.Tensor) -> torch.Tensor:
        """Args:
        target_embedding: [(B*N), L, d_model]
        source_embedding: [S, d_llm]
        Returns:
        out: [(B*N), L, d_llm]
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape

        H = self.n_heads

        q = self.query_projection(target_embedding).view(B, L, H, -1)  # [B, L, H, D]
        k = self.key_projection(source_embedding).view(S, H, -1)  # [S, H, D]
        v = self.value_projection(source_embedding).view(S, H, -1)  # [S, H, D]

        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("blhd,shd->bhls", q, k) * scale  # [B, H, L, S]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,shd->blhd", attn, v)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


# -------------------------- output head --------------------------


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = int(n_vars)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        # Ensure dtype matches Linear params (important when upstream runs in fp16/bf16).
        if x.dtype != self.linear.weight.dtype:
            x = x.to(dtype=self.linear.weight.dtype)
        x = self.linear(x)
        return self.dropout(x)


# -------------------------- lag util (optional, used for prompt) --------------------------


def calcute_lags_fft(x: torch.Tensor, top_k: int) -> torch.Tensor:
    """Compute top-k lag indices using FFT correlation.

    Args:
        x: [B, T, C]
    Returns:
        lags: [B, top_k]
    """
    # This mirrors Time-LLM: calcute_lags()
    if top_k <= 0:
        return torch.zeros((x.shape[0], 0), device=x.device, dtype=torch.long)

    q_fft = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = corr.mean(dim=1)  # [B, T]
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags


# -------------------------- config --------------------------


@dataclass
class TimeLLMTSADConfig:
    """A minimal config surface.

    We keep names close to Time-LLM for easy mapping.
    """

    # data
    seq_len: int
    pred_len: int
    enc_in: int

    # patching
    patch_len: int = 16
    stride: int = 8

    # model dims
    d_model: int = 32
    d_ff: int = 32
    n_heads: int = 4
    dropout: float = 0.1

    # llm
    llm_model: str = "gpt2"  # huggingface model name
    llm_backend: str = "gpt2"  # gpt2 | bert | llama
    llm_layers: int = 6
    llm_pretrained: bool = True
    llm_grad_ckpt: bool = False
    llm_load_in_4bit: bool = False
    llm_load_in_8bit: bool = False
    llm_dtype: str = "auto"

    # HuggingFace loading controls
    hf_cache_dir: Optional[str] = None
    hf_local_files_only: bool = False

    # Forward controls
    llm_use_cache: bool = False

    # prompt
    prompt_mode: str = "stats_short"  # none | dataset | stats | stats_short
    prompt_root: str = "expe/prompt_bank"
    dataset_name: str = "SWaT"
    prompt_domain: str = "anomaly_detection"
    top_k_lags: int = 5

    # output
    head_dropout: float = 0.0
    pred_activation: str = "none"  # none | sigmoid | tanh


# -------------------------- core model --------------------------


class TimeLLMForecast(nn.Module):
    """A Time-LLM style forecaster: x -> y_hat (next pred_len).

    Input:  x [B, seq_len, enc_in]
    Output: y [B, pred_len, enc_in]
    """

    def __init__(self, cfg: TimeLLMTSADConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.patch_len > cfg.seq_len:
            raise ValueError(
                f"llm_patch_len (patch_len={cfg.patch_len}) must be <= context length (seq_len={cfg.seq_len})."
            )

        # lazy import transformers (so that non-LLM baselines do not require it)
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            BertConfig,
            BertModel,
            BertTokenizer,
            GPT2Config,
            GPT2Model,
        )

        # ---- tokenizer & llm ----
        # We support 3 backends (same spirit as Time-LLM): gpt2 / bert / llama.
        backend = (cfg.llm_backend or "gpt2").lower()
        self.llm_backend = backend

        torch_dtype = None
        if cfg.llm_dtype == "float16":
            torch_dtype = torch.float16
        elif cfg.llm_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif cfg.llm_dtype == "float32":
            torch_dtype = torch.float32

        # bitsandbytes quantization is optional
        quantization_config = None
        if cfg.llm_load_in_4bit or cfg.llm_load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=bool(cfg.llm_load_in_4bit),
                    load_in_8bit=bool(cfg.llm_load_in_8bit),
                )
            except Exception:
                quantization_config = None

        # common kwargs for HF from_pretrained
        hf_kwargs = {
            'cache_dir': getattr(cfg, 'hf_cache_dir', None),
            'local_files_only': bool(getattr(cfg, 'hf_local_files_only', False)),
        }

        if backend == "gpt2":
            # NOTE: In some offline/cache_dir layouts, GPT2Tokenizer may resolve vocab_file=None.
            # AutoTokenizer is more robust with HF cache (models--*/snapshots/*) and supports local_files_only.
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True, **hf_kwargs)
            if getattr(self.tokenizer, 'pad_token', None) is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, 'eos_token', None)
            if cfg.llm_pretrained:
                self.llm_model = GPT2Model.from_pretrained(
                    cfg.llm_model,
                    output_attentions=False,
                    output_hidden_states=False,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    **hf_kwargs,
                )
            else:
                # random init GPT2 (ablation)
                base_cfg = GPT2Config.from_pretrained(cfg.llm_model, **hf_kwargs)
                self.llm_model = GPT2Model(base_cfg)
            # keep only first llm_layers
            self.llm_model.h = self.llm_model.h[: cfg.llm_layers]
            self.llm_hidden_size = self.llm_model.config.n_embd
        elif backend == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(cfg.llm_model, **hf_kwargs)
            if cfg.llm_pretrained:
                self.llm_model = BertModel.from_pretrained(
                    cfg.llm_model,
                    output_attentions=False,
                    output_hidden_states=False,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    **hf_kwargs,
                )
            else:
                base_cfg = BertConfig.from_pretrained(cfg.llm_model, **hf_kwargs)
                self.llm_model = BertModel(base_cfg)
            self.llm_model.encoder.layer = self.llm_model.encoder.layer[: cfg.llm_layers]
            self.llm_hidden_size = self.llm_model.config.hidden_size
        else:
            # "llama" or any other AutoModel backend
            # We use AutoTokenizer/AutoModel so user can pass e.g. "meta-llama/Llama-2-7b-hf".
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, **hf_kwargs)
            if self.tokenizer.pad_token is None:
                # many decoder-only LLMs do not have pad_token
                self.tokenizer.pad_token = getattr(self.tokenizer, 'eos_token', None) or self.tokenizer.pad_token
            if cfg.llm_pretrained:
                self.llm_model = AutoModel.from_pretrained(
                    cfg.llm_model,
                    output_attentions=False,
                    output_hidden_states=False,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    **hf_kwargs,
                )
            else:
                base_cfg = AutoConfig.from_pretrained(cfg.llm_model, **hf_kwargs)
                self.llm_model = AutoModel.from_config(base_cfg)

            # try to truncate layers in a best-effort way
            if hasattr(self.llm_model, "layers"):
                self.llm_model.layers = self.llm_model.layers[: cfg.llm_layers]
            elif hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "layers"):
                self.llm_model.model.layers = self.llm_model.model.layers[: cfg.llm_layers]
            self.llm_hidden_size = getattr(self.llm_model.config, "hidden_size", None) or getattr(
                self.llm_model.config, "n_embd", None
            )

        if cfg.llm_grad_ckpt and hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self.llm_model.gradient_checkpointing_enable()

        # freeze llm params
        for p in self.llm_model.parameters():
            p.requires_grad = False

        # ---- prompt ----
        prompt_root = cfg.prompt_root
        prompt_file = _infer_prompt_path(cfg.dataset_name, prompt_root)
        self.prompt_file = prompt_file
        self.dataset_description = _safe_read_text(prompt_file)
        if _debug_enabled():
            desc_len = len((self.dataset_description or "").strip())
            print(f"[LLM Prompt] mode={cfg.prompt_mode} root={prompt_root} file={prompt_file} desc_chars={desc_len}")

        # ---- time-series modules ----
        self.patch_embedding = PatchEmbedding(
            d_model=cfg.d_model,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            dropout=cfg.dropout,
        )
        self.normalize_layer = Normalize(cfg.enc_in, affine=False)

        # map vocab embeddings -> token prototypes
        # (matches Time-LLM idea: mapping_layer: vocab_size -> num_tokens)
        self.vocab_size = int(getattr(self.llm_model.config, "vocab_size", 50257))
        # keep a moderate number of token prototypes (tradeoff speed vs capacity)
        self.num_tokens = int(min(1000, max(128, cfg.d_ff)))
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_keys=cfg.d_model // cfg.n_heads,
            d_llm=self.llm_hidden_size,
            attn_dropout=cfg.dropout,
        )

        # head_nf = d_ff * patch_nums (patch_nums depends on seq_len, patch_len, stride)
        patch_nums = int((cfg.seq_len - cfg.patch_len) / cfg.stride + 2)
        self.patch_nums = patch_nums
        self.head_nf = cfg.d_ff * patch_nums

        self.output_projection = FlattenHead(
            n_vars=cfg.enc_in,
            nf=self.head_nf,
            target_window=cfg.pred_len,
            head_dropout=cfg.head_dropout,
        )

        act = (cfg.pred_activation or "none").lower()
        if act == "sigmoid":
            self.pred_act: Optional[nn.Module] = nn.Sigmoid()
        elif act == "tanh":
            self.pred_act = nn.Tanh()
        else:
            self.pred_act = None

    # -------------------------- prompt --------------------------
    def _build_prompt_list(self, x_enc: torch.Tensor) -> List[str]:
        """Build a prompt per variable (channel independence), like Time-LLM.

        x_enc: [B, T, C] (already normalized)
        Returns: list of length (B*C).
        """

        mode = (self.cfg.prompt_mode or "none").lower()
        if mode == "none":
            # still return empty prompts so shapes match
            B, _, C = x_enc.shape
            return ["" for _ in range(B * C)]

        B, T, C = x_enc.shape

        # compute basic stats per sample (per variable)
        # x_enc: [B, T, C]
        min_values = x_enc.min(dim=1).values  # [B, C]
        max_values = x_enc.max(dim=1).values
        med_values = x_enc.median(dim=1).values
        # simple trend: sum of first differences
        trend = x_enc.diff(dim=1).sum(dim=1)

        if mode in {"stats", "stats_short"} and self.cfg.top_k_lags > 0:
            lags = calcute_lags_fft(x_enc, top_k=self.cfg.top_k_lags)  # [B, k]
        else:
            lags = None

        desc = ""
        if mode in {"dataset", "stats", "stats_short"}:
            desc = (self.dataset_description or "").strip()

        prompts: List[str] = []
        for b in range(B):
            for c in range(C):
                pieces: List[str] = []
                if desc:
                    pieces.append(desc)
                # task instruction
                if self.cfg.prompt_domain == "anomaly_detection":
                    pieces.append(
                        "Task: Forecast the next values under normal behavior to help anomaly detection."
                    )
                else:
                    pieces.append("Task: Forecast the next values.")

                if mode in {"stats", "stats_short"}:
                    pieces.append(
                        f"Input stats: min={min_values[b, c].item():.3f}, max={max_values[b, c].item():.3f}, "
                        f"median={med_values[b, c].item():.3f}."
                    )
                    pieces.append(f"Trend={trend[b, c].item():.3f}.")
                    if mode == "stats" and lags is not None and lags.numel() > 0:
                        pieces.append(f"Top lags={lags[b].tolist()}.")

                prompts.append("\n".join(pieces))
        return prompts

    # -------------------------- forward --------------------------
    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forecast next pred_len.

        Args:
            x_enc: [B, T, C]
        Returns:
            y: [B, pred_len, C]
        """

        if x_enc.dim() != 3:
            raise ValueError(f"TimeLLMForecast expects [B,T,C], got {tuple(x_enc.shape)}")
        B, T, C = x_enc.shape
        if T != self.cfg.seq_len:
            raise ValueError(f"seq_len mismatch: cfg.seq_len={self.cfg.seq_len}, got T={T}")
        if C != self.cfg.enc_in:
            raise ValueError(f"enc_in mismatch: cfg.enc_in={self.cfg.enc_in}, got C={C}")

        # normalize
        x_enc = self.normalize_layer(x_enc, "norm")

        # prompt embeddings
        prompts = self._build_prompt_list(x_enc)
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        if _debug_enabled():
            # tokenized['input_ids']: [B*C, Lp]
            try:
                lp = int(tokenized["input_ids"].shape[1])
            except Exception:
                lp = -1
            # print only once per process to avoid log spam
            if not hasattr(self, "_did_print_prompt_len"):
                self._did_print_prompt_len = True
                sample = prompts[0].replace("\n", " ")[:180] if prompts else ""
                print(f"[LLM Prompt] token_len={lp} sample='{sample}...'")
        tokenized = {k: v.to(x_enc.device) for k, v in tokenized.items()}
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized["input_ids"])  # [B*C, Lp, d_llm]

        # patch embed (channel-independence)
        x_var = x_enc.permute(0, 2, 1).contiguous()  # [B, C, T]
        enc_out, n_vars = self.patch_embedding(x_var)  # [(B*C), patch_nums, d_model]

        # LLM token prototypes
        word_embeddings = self.llm_model.get_input_embeddings().weight  # [vocab, d_llm]
        # compress vocab -> num_tokens
        # Important: when LLM weights run in fp16/bf16 but mapping_layer stays fp32,
        # F.linear will fail (Half vs Float). Align dtypes explicitly.
        map_dtype = self.mapping_layer.weight.dtype
        if word_embeddings.dtype != map_dtype:
            word_embeddings = word_embeddings.to(dtype=map_dtype)
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)  # [num_tokens, d_llm]

        # Reprogram to LLM space.
        # Keep reprogramming in fp32 (matches projection weights) for numerical stability.
        if enc_out.dtype != map_dtype:
            enc_out = enc_out.to(dtype=map_dtype)
        if source_embeddings.dtype != map_dtype:
            source_embeddings = source_embeddings.to(dtype=map_dtype)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings)  # [(B*C), patch_nums, d_llm]

        # Now align to LLM embedding dtype for the frozen LLM forward.
        llm_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if prompt_embeddings.dtype != llm_dtype:
            prompt_embeddings = prompt_embeddings.to(dtype=llm_dtype)
        if enc_out.dtype != llm_dtype:
            enc_out = enc_out.to(dtype=llm_dtype)

        # concat prompt + ts tokens
        llm_in = torch.cat([prompt_embeddings, enc_out], dim=1)
        llm_out = self.llm_model(inputs_embeds=llm_in, use_cache=bool(getattr(self.cfg, 'llm_use_cache', False))).last_hidden_state  # [(B*C), Lp+patch, d_llm]

        # keep first d_ff dims as in Time-LLM (acts like a bottleneck)
        d_ff = self.cfg.d_ff
        llm_out = llm_out[:, :, :d_ff]

        # reshape back
        llm_out = torch.reshape(llm_out, (-1, n_vars, llm_out.shape[-2], llm_out.shape[-1]))  # [B, C, L, d_ff]
        llm_out = llm_out.permute(0, 1, 3, 2).contiguous()  # [B, C, d_ff, L]
        # take last patch tokens (skip prompt)
        ts_out = llm_out[:, :, :, -self.patch_nums :]
        y = self.output_projection(ts_out)  # [B, C, pred_len]
        y = y.permute(0, 2, 1).contiguous()  # [B, pred_len, C]

        # denorm to original scale
        y = self.normalize_layer(y, "denorm")

        if self.pred_act is not None:
            y = self.pred_act(y)

        return y
