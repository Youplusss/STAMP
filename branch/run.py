# -*- coding: utf-8 -*-
"""Branch-only training entrypoint.

This script is meant for *decoupled* experiments where you only train ONE branch:
- llm predictor
- mamba predictor
- mamba recon

It reuses the same dataloaders and model builders as the main `run.py`.

Note
----
For convenience, we keep a set of sane *fixed defaults* inside the script for
common hyper-parameters (window_size/n_pred/downsample/MAS/LRs, etc.).
This makes CLI commands shorter. You can still override them via CLI if needed.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path when running as a script:
#   python branch/run.py ...
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from lib.metrics import masked_mse_loss
from lib.paths import resolve_experiment_dirs
from lib.utils import plot_history_pred
from model.utils import init_seed, print_model_parameters


# ---- fixed defaults (keep commands short) ----
_FIXED_DEFAULTS = {
    # sequence
    "window_size": 15,
    "n_pred": 3,
    "in_channels": 1,
    "out_channels": 1,
    # data
    "val_ratio": 0.2,
    "is_down_sample": True,
    "down_len": 100,
    "is_mas": True,
    "real_value": False,
    # training
    "seed": 666,
    "epochs": 30,
    "batch_size": 4,
    "pred_lr_init": 1e-3,
    "ae_lr_init": 1e-3,
    "pred_weight_decay": 1e-4,
    "ae_weight_decay": 1e-4,
    "early_stop": True,
    "early_stop_patience": 10,
    "grad_clip": True,
    "max_grad_norm": 1.0,
    # mamba
    "mamba_use_mas": True,
    "mamba_d_model": 256,
    "mamba_e_layers": 3,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "mamba_dropout": 0.1,
    "mamba_use_norm": True,
    "mamba_use_last_residual": True,
    # recon
    "recon_d_model": 256,
    "recon_num_layers": "2,2,2",
    "recon_d_state": 16,
    "recon_d_conv": 4,
    "recon_expand": 2,
    "recon_dropout": 0.1,
    "recon_output_activation": "auto",
    # llm
    "llm_backend": "gpt2",
    "llm_pretrained": True,
    "llm_layers": 6,
    "llm_dtype": "auto",
    "llm_grad_ckpt": False,
    "llm_load_in_4bit": False,
    "llm_load_in_8bit": False,
    "llm_use_cache": False,
    "llm_use_mas": False,
    "llm_patch_len": 4,
    "llm_stride": 2,
    "llm_d_model": 32,
    "llm_d_ff": 32,
    "llm_n_heads": 4,
    "llm_dropout": 0.1,
    "llm_head_dropout": 0.0,
    "llm_pred_activation": "none",
    "llm_prompt_mode": "stats_short",
    "llm_prompt_root": "expe/prompt_bank",
    "llm_prompt_domain": "anomaly_detection",
    "llm_top_k_lags": 5,
    "llm_feature_mixer": "none",
    "llm_mixer_rank": 64,
}


def _apply_fixed_defaults(args):
    """Apply internal defaults if a field is missing.

    We intentionally don't override user-provided CLI values.
    """

    for k, v in _FIXED_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-TSAD branch-only training")

    # ---- core ----
    p.add_argument("--data", type=str, default="SWaT")
    p.add_argument("--group", default=None, type=str)
    p.add_argument("--gpu_id", default="0", type=str)
    p.add_argument("--log_dir", default="expe_branch", type=str)
    p.add_argument("--debug", default=False, type=eval)

    # select branch
    p.add_argument(
        "--branch",
        type=str,
        required=True,
        choices=["llm_pred", "mamba_pred", "mamba_recon"],
        help="which single branch to train",
    )

    # dataset optional overrides
    p.add_argument("--train_file", default=None, type=str)
    p.add_argument("--test_file", default=None, type=str)

    # Data/downsample knobs (kept optional; defaults match internal fixed defaults)
    p.add_argument("--val_ratio", type=float, default=_FIXED_DEFAULTS["val_ratio"])
    p.add_argument("--is_down_sample", default=_FIXED_DEFAULTS["is_down_sample"], type=eval)
    p.add_argument("--down_len", type=int, default=_FIXED_DEFAULTS["down_len"], help="downsample length (default: 100)")

    # shapes (mostly inferred from data; keep nnodes override)
    p.add_argument("--nnodes", type=int, default=45)

    # training
    p.add_argument("--epochs", type=int, default=_FIXED_DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=_FIXED_DEFAULTS["batch_size"])

    # lighter-weight overrides when you need them
    p.add_argument("--seed", type=int, default=_FIXED_DEFAULTS["seed"])
    p.add_argument("--pred_lr_init", type=float, default=_FIXED_DEFAULTS["pred_lr_init"])
    p.add_argument("--ae_lr_init", type=float, default=_FIXED_DEFAULTS["ae_lr_init"])

    # ---- LLM predictor (only used when branch=llm_pred) ----
    p.add_argument("--llm_model", type=str, default="gpt2", help="HF repo id or local snapshot path")
    p.add_argument("--llm_backend", type=str, default=_FIXED_DEFAULTS["llm_backend"], help="gpt2 | bert | llama (AutoModel)")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    # IMPORTANT:
    # - False (default): if not found in local cache, allow downloading from Hub (will respect HF_ENDPOINT/HF_TOKEN env).
    # - True: offline-only, never download.
    p.add_argument("--hf_local_files_only", default=False, type=eval)
    p.add_argument("--llm_layers", type=int, default=_FIXED_DEFAULTS["llm_layers"])
    p.add_argument(
        "--llm_dtype",
        type=str,
        default=_FIXED_DEFAULTS["llm_dtype"],
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--llm_grad_ckpt", default=_FIXED_DEFAULTS["llm_grad_ckpt"], type=eval)
    p.add_argument("--llm_use_cache", default=_FIXED_DEFAULTS["llm_use_cache"], type=eval)

    return p


def load_dataloaders(args, device):
    # reuse the same implementation from main test.py (also used in main run.py)
    import importlib

    main_test = importlib.import_module("test")
    return main_test.load_dataloaders(args, device)


def infer_and_override_data_shape(args, train_loader):
    import importlib

    main_run = importlib.import_module("run")
    return main_run.infer_and_override_data_shape(args, train_loader)


def _resolve_hf_snapshot(model_ref: str, cache_root: str | None) -> str | None:
    """Try to resolve a short model name or repo id to a local HF snapshot path.

    Supports common HF cache layouts:
      - <cache_root>/models--ORG--NAME/snapshots/<hash>
      - <cache_root>/hub/models--ORG--NAME/snapshots/<hash>

    Returns a snapshot directory path if found, else None.
    """

    if not model_ref:
        return None

    # if user already passed a local directory, keep it
    if os.path.isdir(model_ref):
        return model_ref

    # candidate cache roots (user-provided > env > common defaults)
    roots: list[str] = []
    if cache_root:
        roots.append(cache_root)
    for env_k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        v = os.environ.get(env_k)
        if v:
            roots.append(v)

    # common linux locations
    roots.extend([
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/code/huggingface"),
    ])

    # build model dir name patterns
    # - if user said "gpt2", try models--gpt2 and models--openai-community--gpt2
    # - if user said "openai-community/gpt2", map to models--openai-community--gpt2
    model_ref = model_ref.strip()
    if "/" in model_ref:
        org, name = model_ref.split("/", 1)
        candidates = [f"models--{org}--{name}"]
    else:
        candidates = [
            f"models--{model_ref}",
            f"models--openai-community--{model_ref}",
        ]

    for root in roots:
        for base in (root, os.path.join(root, "hub")):
            for cand in candidates:
                snap_root = os.path.join(base, cand, "snapshots")
                if not os.path.isdir(snap_root):
                    continue
                # pick the newest snapshot dir
                try:
                    snaps = [
                        os.path.join(snap_root, d)
                        for d in os.listdir(snap_root)
                        if os.path.isdir(os.path.join(snap_root, d))
                    ]
                except OSError:
                    continue
                if not snaps:
                    continue
                snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return snaps[0]

    return None


def _canonicalize_llm_model_args(args):
    """Make --llm_model accept short names while preferring local snapshots.

    Behavior:
      - If `--llm_model` points to an existing local directory, we use it and force offline (local_files_only=True).
      - Otherwise, we try to resolve a local HF snapshot under common cache roots.
        If we find one, we use that snapshot path and force offline.
      - If no local snapshot is found, we keep the original repo id and DO NOT force offline,
        so Transformers can download from Hub/mirror when needed.

    Also: auto-pick llm_backend for common model families.
      - GPT2-like: llm_backend='gpt2'
      - Qwen/Llama-like decoder-only: llm_backend='llama' (AutoModel)
    """

    if str(getattr(args, "branch", "")).lower() != "llm_pred":
        return

    cache_root = getattr(args, "hf_cache_dir", None)
    raw_ref = str(getattr(args, "llm_model", "")).strip()

    # If user already passed a local directory, keep it and force offline.
    if raw_ref and os.path.isdir(raw_ref):
        args.llm_model = raw_ref
        args.hf_local_files_only = True
    else:
        resolved = _resolve_hf_snapshot(raw_ref, cache_root)
        if resolved:
            args.llm_model = resolved
            args.hf_local_files_only = True

    # backend auto-detect (can still be overridden by CLI)
    if not hasattr(args, "llm_backend") or args.llm_backend is None:
        args.llm_backend = _FIXED_DEFAULTS.get("llm_backend", "gpt2")

    ref = (raw_ref or "").lower()
    if "qwen" in ref:
        args.llm_backend = "llama"
    # if a local snapshot path contains qwen in folder name
    if isinstance(getattr(args, "llm_model", None), str) and "qwen" in str(args.llm_model).lower():
        args.llm_backend = "llama"


def main():
    args = build_arg_parser().parse_args()
    _apply_fixed_defaults(args)
    _canonicalize_llm_model_args(args)

    # make a readable model name
    args.model = f"branch_{args.branch}"

    # CUDA device selection:
    # - If user already set CUDA_VISIBLE_DEVICES in the shell (recommended for multi-process/nohup),
    #   DO NOT override it here.
    # - Otherwise, allow --gpu_id to set CUDA_VISIBLE_DEVICES.
    #   Note: when CUDA_VISIBLE_DEVICES is set to a single GPU (e.g. "3"), inside the process
    #   it becomes cuda:0. This is why OOM messages may mention "GPU 0" while actually using the
    #   first (and only) visible device.
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[Device] Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (from --gpu_id)")
    else:
        print(f"[Device] Using existing CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir

    device = get_default_device()
    args.device = device

    init_seed(args.seed)

    train_loader, val_loader, test_loader, _y_test_labels, scaler = load_dataloaders(args, device)
    infer_and_override_data_shape(args, train_loader)

    os.makedirs(exp.log_dir, exist_ok=True)
    os.makedirs(exp.pth_dir, exist_ok=True)
    os.makedirs(exp.pdf_dir, exist_ok=True)

    branch = args.branch.lower()

    if branch in ("llm_pred", "mamba_pred"):
        # build predictor
        if branch == "llm_pred":
            from model.llm_wrappers import build_stamp_llm_predictor

            pred_model = build_stamp_llm_predictor(args)
        else:
            from model.mamba_wrappers import build_stamp_mamba_models

            mamba_pred, _mamba_ae = build_stamp_mamba_models(args)
            pred_model = mamba_pred

        pred_model = to_device(pred_model, device)

        pred_params = list(pred_model.parameters())
        if len(pred_params) == 0:
            raise ValueError("Predictor has no trainable parameters")

        pred_opt = torch.optim.Adam(pred_params, lr=args.pred_lr_init, eps=1e-8, weight_decay=float(args.pred_weight_decay))
        loss_fn = masked_mse_loss(mask_value=-0.01)

        print_model_parameters(pred_model)

        from branch.trainer import PredOnlyTrainer
        from lib.logger import get_logger

        logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag="train", model=args.model, run_id=exp.run_id, console=True)
        trainer = PredOnlyTrainer(pred_model, loss_fn, pred_opt, train_loader, val_loader, test_loader, args, scaler, logger=logger)
        history = trainer.train()
        plot_history_pred(history, model=args.model, data=args.data, out_dir=exp.pdf_dir, show=False)

    elif branch == "mamba_recon":
        from model.mamba_wrappers import build_stamp_mamba_models

        _mamba_pred, ae_model = build_stamp_mamba_models(args)
        ae_model = to_device(ae_model, device)

        ae_params = list(ae_model.parameters())
        if len(ae_params) == 0:
            raise ValueError("Recon model has no trainable parameters")

        ae_opt = torch.optim.Adam(ae_params, lr=args.ae_lr_init, eps=1e-8, weight_decay=float(args.ae_weight_decay))
        loss_fn = masked_mse_loss(mask_value=-0.01)

        print_model_parameters(ae_model)

        from branch.trainer import ReconOnlyTrainer
        from lib.logger import get_logger

        logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag="train", model=args.model, run_id=exp.run_id, console=True)
        trainer = ReconOnlyTrainer(ae_model, loss_fn, ae_opt, train_loader, val_loader, test_loader, args, scaler, logger=logger)
        history = trainer.train()
        plot_history_pred(history, model=args.model, data=args.data, out_dir=exp.pdf_dir, show=False)

    else:
        raise ValueError(f"Unknown branch: {branch}")


if __name__ == "__main__":
    main()
