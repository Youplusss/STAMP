# -*- coding: utf-8 -*-
"""Branch-only testing entrypoint.

Computes anomaly scores using ONLY one branch:
- predictor-only (llm_pred / mamba_pred): score = MSE(pred, target)
- recon-only (mamba_recon): score = MSE(recon(x), x)

Then calls the existing `lib.evaluate.get_final_result` to compute metrics.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path when running as a script:
#   python branch/test.py ...
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch


from lib.evaluate import get_final_result
from lib.paths import resolve_experiment_dirs
from model.utils import init_seed
# NEW: structured logging like main test.py
from lib.logger import get_logger, log_hparams, log_test_results


# Keep the same fixed defaults as branch/run.py, to make CLI commands short.
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
    # evaluation
    "search_steps": 50,
    "test_topk": 1,
    "test_topk_agg": "sum",
}


def _apply_fixed_defaults(args):
    for k, v in _FIXED_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def _apply_branch_default_weights(args):
    """Auto set alpha/beta/gamma for branch-only scoring.

    - pred-only (llm_pred / mamba_pred): alpha=1
    - recon-only (mamba_recon): beta=1
    """

    branch = str(getattr(args, "branch", "")).lower()
    if branch in ("llm_pred", "mamba_pred"):
        args.test_alpha = float(getattr(args, "test_alpha", 1.0))
        args.test_beta = 0.0
        args.test_gamma = 0.0
    elif branch == "mamba_recon":
        args.test_alpha = 0.0
        args.test_beta = float(getattr(args, "test_beta", 1.0))
        args.test_gamma = 0.0


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-TSAD branch-only testing")

    # core
    p.add_argument("--data", type=str, default="SWaT")
    p.add_argument("--group", default=None, type=str)
    p.add_argument("--gpu_id", default="0", type=str)
    p.add_argument("--log_dir", default="expe_branch", type=str)
    p.add_argument("--debug", default=False, type=eval)

    p.add_argument(
        "--branch",
        type=str,
        required=True,
        choices=["llm_pred", "mamba_pred", "mamba_recon"],
    )

    # checkpoint override (optional)
    p.add_argument("--ckpt", type=str, default=None, help="explicit checkpoint path")

    # dataset optional overrides
    p.add_argument("--train_file", default=None, type=str)
    p.add_argument("--test_file", default=None, type=str)

    # Data/downsample knobs (optional; defaults match internal fixed defaults)
    p.add_argument("--val_ratio", type=float, default=_FIXED_DEFAULTS["val_ratio"])
    p.add_argument("--is_down_sample", default=_FIXED_DEFAULTS["is_down_sample"], type=eval)
    p.add_argument("--down_len", type=int, default=_FIXED_DEFAULTS["down_len"], help="downsample length (default: 100)")

    # keep nnodes override (mostly inferred)
    p.add_argument("--nnodes", type=int, default=45)

    # runtime
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--batch_size", type=int, default=4)

    # LLM offline options (only used when branch=llm_pred)
    p.add_argument("--llm_model", type=str, default="gpt2")
    p.add_argument("--llm_backend", type=str, default=_FIXED_DEFAULTS["llm_backend"], help="gpt2 | bert | llama (AutoModel)")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--hf_local_files_only", default=False, type=eval)
    p.add_argument("--llm_layers", type=int, default=_FIXED_DEFAULTS["llm_layers"])
    p.add_argument("--llm_dtype", type=str, default=_FIXED_DEFAULTS["llm_dtype"], choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--llm_grad_ckpt", default=_FIXED_DEFAULTS["llm_grad_ckpt"], type=eval)
    p.add_argument("--llm_use_cache", default=_FIXED_DEFAULTS["llm_use_cache"], type=eval)

    # evaluation params (keep but optional)
    p.add_argument("--search_steps", type=int, default=_FIXED_DEFAULTS["search_steps"])
    p.add_argument("--test_alpha", type=float, default=1.0)
    p.add_argument("--test_beta", type=float, default=1.0)
    p.add_argument("--test_gamma", type=float, default=0.0)
    p.add_argument("--test_topk", type=int, default=_FIXED_DEFAULTS["test_topk"])
    p.add_argument("--test_topk_agg", type=str, default=_FIXED_DEFAULTS["test_topk_agg"], choices=["sum", "mean", "max"])

    return p


def load_dataloaders(args, device):
    import importlib

    main_test = importlib.import_module("test")
    return main_test.load_dataloaders(args, device)


def infer_and_override_data_shape(args, train_loader):
    import importlib

    main_run = importlib.import_module("run")
    return main_run.infer_and_override_data_shape(args, train_loader)


def _score_pred_only(args, pred_model, data_loader, scaler) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    pred_model.eval()

    # Per-window, per-node scores are expected by lib.evaluate.get_score_PredAndAE
    # Shapes should align with (samples, nnodes) after internal reshape.
    scores = []
    preds = []
    gts = []

    pred_channels = int(args.n_pred * args.nnodes * args.out_channels)

    with torch.no_grad():
        for batch_m in data_loader:
            if bool(getattr(args, "is_mas", False)):
                batch, mas = batch_m
                batch = batch.to(args.device, non_blocking=True)
                mas = mas.to(args.device, non_blocking=True)
                mas = mas[:, : args.window_size - args.n_pred, ...]
            else:
                batch, mas = batch_m[0], None
                batch = batch.to(args.device, non_blocking=True)

            x = batch[:, : args.window_size - args.n_pred, ...]
            target = batch[:, -args.n_pred :, ...]
            output = pred_model(x, mas=mas)

            if bool(getattr(args, "real_value", False)):
                target_np = scaler.inverse_transform(
                    target.reshape(-1, args.n_pred, args.nnodes * args.out_channels).detach().cpu().numpy()
                )
                target = (
                    torch.from_numpy(target_np)
                    .float()
                    .view(-1, args.n_pred, args.nnodes, args.out_channels)
                    .to(batch.device)
                )

            # score per node: mean over pred_len and channels
            se = (output - target) ** 2  # [B, n_pred, N, C]
            node_score = se.mean(dim=(1, 3))  # [B, N]
            scores.append(node_score.detach().cpu().numpy())

            preds.append(output.reshape(-1, args.n_pred, args.nnodes * args.out_channels).detach().cpu().numpy())
            gts.append(target.reshape(-1, args.n_pred, args.nnodes * args.out_channels).detach().cpu().numpy())

    node_scores = np.concatenate(scores, axis=0) if scores else np.zeros((0, args.nnodes), dtype=np.float32)
    preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, args.n_pred, args.nnodes * args.out_channels), dtype=np.float32)
    gts = np.concatenate(gts, axis=0) if gts else np.zeros((0, args.n_pred, args.nnodes * args.out_channels), dtype=np.float32)

    return node_scores, (preds, gts)


def _score_recon_only(args, ae_model, data_loader, scaler) -> np.ndarray:
    ae_model.eval()

    scores = []
    ae_channels = int(args.window_size * args.nnodes * args.out_channels)

    with torch.no_grad():
        for batch_m in data_loader:
            if bool(getattr(args, "is_mas", False)):
                batch, _mas = batch_m
            else:
                batch = batch_m[0]
            batch = batch.to(args.device, non_blocking=True)

            target1 = batch.reshape(-1, ae_channels)
            output1 = ae_model(target1)

            if bool(getattr(args, "real_value", False)):
                target1_np = scaler.inverse_transform(
                    target1.reshape(-1, args.window_size, args.nnodes * args.out_channels).detach().cpu().numpy()
                )
                target1 = torch.from_numpy(target1_np).float().view(-1, ae_channels).to(batch.device)

            # per-node score: mean over time and channels
            se = (output1 - target1).view(-1, args.window_size, args.nnodes, args.out_channels) ** 2
            node_score = se.mean(dim=(1, 3))  # [B, N]
            scores.append(node_score.detach().cpu().numpy())

    node_scores = np.concatenate(scores, axis=0) if scores else np.zeros((0, args.nnodes), dtype=np.float32)
    return node_scores


def _score_recon_only_window_first(args, ae_model, data_loader, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Return recon results in evaluate.py expected layout.

    IMPORTANT: Despite the name, evaluate.py's get_Test_scores_err_max expects the 3D arrays to be:
        [samples, window_len, num_features]
    because it does:
        _, w, total_features = arr.shape
        for i in range(w):
            test_result = [data[:, i, :] for data in test_ae_result]
    where `data[:, i, :]` must yield [samples, num_features].

    Therefore, we return:
      recon: [B, L, N]
      orig : [B, L, N]
    """

    ae_model.eval()

    Lw = int(args.window_size)
    N = int(args.nnodes)
    C = int(args.out_channels)
    if C != 1:
        raise ValueError(f"branch-only recon eval currently expects out_channels=1, got {C}")

    recon_all = []
    orig_all = []

    with torch.no_grad():
        for batch_m in data_loader:
            if bool(getattr(args, "is_mas", False)):
                batch, _mas = batch_m
            else:
                batch = batch_m[0]
            batch = batch.to(args.device, non_blocking=True)  # [B, L, N, 1]

            target1 = batch.reshape(-1, Lw * N * C)  # [B, L*N]
            output1 = ae_model(target1)  # [B, L*N]

            if bool(getattr(args, "real_value", False)):
                tgt_np = scaler.inverse_transform(
                    target1.reshape(-1, Lw, N * C).detach().cpu().numpy()
                )
                out_np = scaler.inverse_transform(
                    output1.reshape(-1, Lw, N * C).detach().cpu().numpy()
                )
                target1 = torch.from_numpy(tgt_np).float().view(-1, Lw * N * C).to(batch.device)
                output1 = torch.from_numpy(out_np).float().view(-1, Lw * N * C).to(batch.device)

            # reshape to [B, L, N]
            tgt_w = target1.view(-1, Lw, N)
            out_w = output1.view(-1, Lw, N)

            orig_all.append(tgt_w.detach().cpu().numpy())
            recon_all.append(out_w.detach().cpu().numpy())

    if not orig_all:
        empty = np.zeros((0, Lw, N), dtype=np.float32)
        return empty, empty

    orig = np.concatenate(orig_all, axis=0).astype(np.float32)   # [B_total, L, N]
    recon = np.concatenate(recon_all, axis=0).astype(np.float32) # [B_total, L, N]
    return recon, orig


def _resolve_hf_snapshot(model_ref: str, cache_root: str | None) -> str | None:
    """Try to resolve a short model name or repo id to a local HF snapshot path.

    Supports common HF cache layouts:
      - <cache_root>/models--ORG--NAME/snapshots/<hash>
      - <cache_root>/hub/models--ORG--NAME/snapshots/<hash>

    Returns a snapshot directory path if found, else None.
    """

    if not model_ref:
        return None

    if os.path.isdir(model_ref):
        return model_ref

    roots: list[str] = []
    if cache_root:
        roots.append(cache_root)
    for env_k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        v = os.environ.get(env_k)
        if v:
            roots.append(v)

    roots.extend([
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/code/huggingface"),
    ])

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
    if str(getattr(args, "branch", "")).lower() != "llm_pred":
        return

    cache_root = getattr(args, "hf_cache_dir", None)
    raw_ref = str(getattr(args, "llm_model", ""))
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
    if isinstance(getattr(args, "llm_model", None), str) and "qwen" in str(args.llm_model).lower():
        args.llm_backend = "llama"


def _to_eval_3d_layout(x_windows: np.ndarray) -> np.ndarray:
    """Ensure arrays follow evaluate.py expected layout: [samples, window_len, features]."""
    if x_windows.ndim != 3:
        raise ValueError(f"Expected 3D array [samples,window_len,features], got {x_windows.shape}")
    return x_windows


def main():
    args = build_arg_parser().parse_args()
    _apply_fixed_defaults(args)
    _apply_branch_default_weights(args)
    _canonicalize_llm_model_args(args)

    args.model = f"branch_{args.branch}"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir

    # NEW: create a dedicated *_test.log under exp.log_dir
    logger = get_logger(
        exp.log_dir,
        name=args.model,
        debug=args.debug,
        data=args.data,
        tag="test",
        model=args.model,
        run_id=exp.run_id,
        console=True,
    )
    log_hparams(logger, args)

    device = get_default_device()
    args.device = device
    init_seed(args.seed)

    train_loader, val_loader, test_loader, y_test_labels, scaler = load_dataloaders(args, device)
    infer_and_override_data_shape(args, train_loader)

    branch = args.branch.lower()

    # prepare results expected by lib.evaluate.get_final_result:
    # test_pred_results: [test_predicted_list, test_ground_list]
    # test_ae_results  : [test_constructed_list, test_original_list]
    # test_generate_results: [test_constructed_generate_list, test_generate_list]

    test_pred_results = None
    test_ae_results = None
    test_generate_results = None

    if branch in ("llm_pred", "mamba_pred"):
        if branch == "llm_pred":
            from model.llm_wrappers import build_stamp_llm_predictor

            pred_model = build_stamp_llm_predictor(args)
        else:
            from model.mamba_wrappers import build_stamp_mamba_models

            pred_model, _ae = build_stamp_mamba_models(args)

        pred_model = to_device(pred_model, device)

        ckpt = args.ckpt or os.path.join(args.log_dir_pth, f"best_pred_only_{args.data}_{args.model}.pth")
        if os.path.isfile(ckpt):
            print(f"Load predictor ckpt: {ckpt}")
            cp = torch.load(ckpt, map_location=device, weights_only=False)
            pred_model.load_state_dict(cp["pred_state_dict"])

        node_scores, (preds, gts) = _score_pred_only(args, pred_model, test_loader, scaler)

        # build pseudo "results" arrays; only pred is used by alpha (set beta/gamma=0)
        test_pred_results = [preds, gts]
        zeros = np.zeros_like(node_scores)
        test_ae_results = [zeros, zeros]
        test_generate_results = [zeros, zeros]

    elif branch == "mamba_recon":
        from model.mamba_wrappers import build_stamp_mamba_models

        _pred, ae_model = build_stamp_mamba_models(args)
        ae_model = to_device(ae_model, device)

        ckpt = args.ckpt or os.path.join(args.log_dir_pth, f"best_recon_only_{args.data}_{args.model}.pth")
        if os.path.isfile(ckpt):
            print(f"Load recon ckpt: {ckpt}")
            cp = torch.load(ckpt, map_location=device, weights_only=False)
            ae_model.load_state_dict(cp["ae_state_dict"])

        # IMPORTANT: evaluate.py option=2 expects [window_len, samples, features]
        recon_w, orig_w = _score_recon_only_window_first(args, ae_model, test_loader, scaler)
        recon_w = _to_eval_3d_layout(recon_w)
        orig_w = _to_eval_3d_layout(orig_w)

        # recon-only: drive score using beta (set alpha/gamma=0)
        # pred/generate are dummies but must match expected 3D shape too.
        # NOTE: Do NOT use zeros_like(orig_w) directly later as a 2D node-score placeholder.
        zeros_w = np.zeros_like(orig_w, dtype=np.float32)

        test_pred_results = [zeros_w, zeros_w]
        test_ae_results = [recon_w, orig_w]
        test_generate_results = [zeros_w, zeros_w]

    else:
        raise ValueError(f"Unknown branch: {branch}")

    # Consistent with the original test.py: evaluate multiple score reduction methods.
    # Sanity check: the final score sequence length must match y_test_labels.
    # We do a cheap dry-run to catch shape mismatches early.
    def _check_len(method_name: str):
        _info, _test_scores, _predict = get_final_result(
            test_pred_results,
            test_ae_results,
            test_generate_results,
            y_test_labels,
            topk=int(getattr(args, "test_topk", 1)),
            topk_agg=str(getattr(args, "test_topk_agg", "sum")),
            option=2,
            method=method_name,
            alpha=float(getattr(args, "test_alpha", 1.0)),
            beta=float(getattr(args, "test_beta", 0.0)),
            gamma=float(getattr(args, "test_gamma", 0.0)),
            search_steps=1,
        )
        return _info

    # If this triggers, evaluate.py will later throw "score and label must have the same length".
    # This makes the error earlier & more local.
    _ = _check_len("max")

    print(f"[Branch Test] branch={branch} topk={int(getattr(args, 'test_topk', 1))} topk_agg={str(getattr(args, 'test_topk_agg', 'sum'))}")
    print(f"[Test Weights] alpha={float(getattr(args, 'test_alpha', 1.0))} beta={float(getattr(args, 'test_beta', 0.0))} gamma={float(getattr(args, 'test_gamma', 0.0))}")

    # NEW: record ablation params into the test log (from env)
    def _env_or_empty(k: str) -> str:
        v = os.getenv(k)
        return "" if v is None else str(v)

    ablation_items = [
        ("num_mamba_layers", _env_or_empty("STAMP_RECON_NUM_MAMBA_LAYERS")),
        ("lss_residual", _env_or_empty("STAMP_RECON_LSS_RESIDUAL")),
        ("local_conv_variant", _env_or_empty("STAMP_RECON_LOCAL_CONV_VARIANT")),
        ("kernel_sizes", _env_or_empty("STAMP_RECON_KERNEL_SIZES")),
    ]
    logger.info("[Ablation]\n" + "\n".join([f"- {k}: {v}" for k, v in ablation_items]))

    best_by_method = {}

    print("================= Find best f1 from score (method=max) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        topk=int(getattr(args, "test_topk", 1)),
        topk_agg=str(getattr(args, "test_topk_agg", "sum")),
        option=2,
        method="max",
        alpha=float(getattr(args, "test_alpha", 1.0)),
        beta=float(getattr(args, "test_beta", 0.0)),
        gamma=float(getattr(args, "test_gamma", 0.0)),
        search_steps=int(getattr(args, "search_steps", 50)),
    )
    print(info)
    best_by_method["max"] = info

    print("\n================= Find best f1 from score (method=sum) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        topk=int(getattr(args, "test_topk", 1)),
        topk_agg=str(getattr(args, "test_topk_agg", "sum")),
        option=2,
        method="sum",
        alpha=float(getattr(args, "test_alpha", 1.0)),
        beta=float(getattr(args, "test_beta", 0.0)),
        gamma=float(getattr(args, "test_gamma", 0.0)),
        search_steps=int(getattr(args, "search_steps", 50)),
    )
    print(info)
    best_by_method["sum"] = info

    print("\n================= Find best f1 from score (method=mean) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        topk=int(getattr(args, "test_topk", 1)),
        topk_agg=str(getattr(args, "test_topk_agg", "sum")),
        option=2,
        method="mean",
        alpha=float(getattr(args, "test_alpha", 1.0)),
        beta=float(getattr(args, "test_beta", 0.0)),
        gamma=float(getattr(args, "test_gamma", 0.0)),
        search_steps=int(getattr(args, "search_steps", 50)),
    )
    print(info)
    best_by_method["mean"] = info

    # NEW: write a structured summary into *_test.log
    ckpt_path = args.ckpt or os.path.join(
        args.log_dir_pth,
        (f"best_recon_only_{args.data}_{args.model}.pth" if branch == "mamba_recon" else f"best_pred_only_{args.data}_{args.model}.pth"),
    )
    log_test_results(
        logger,
        dataset=args.data,
        model=args.model,
        checkpoint_path=ckpt_path,
        best_by_method=best_by_method,
    )


if __name__ == "__main__":
    main()
