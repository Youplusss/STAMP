#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAMP ablation runner.

This helper script generates (and optionally executes) a set of ablation
experiments, then calls `test.py` to dump metrics into JSON files.

It is designed to match the ablation plan in `docs/ablation_plan.md`.

Examples (from repo root)
-------------------------
Dry-run (only print commands):

    python scripts/run_ablation_grid.py --suite framework --data SMD --dataset AIOps --dry_run

Run experiments:

    python scripts/run_ablation_grid.py --suite framework --data SMD --dataset AIOps --out_root exp/ablations

Notes
-----
- This script assumes `run.py` writes best checkpoint to:
      {log_dir}/best_model_{data}_{model}.pth
- You can pass extra arguments to run/test via `--extra_run_args` / `--extra_test_args`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Exp:
    exp_id: str
    desc: str
    run_overrides: Dict[str, object]
    test_overrides: Dict[str, object]


def dict_to_cli(d: Dict[str, object]) -> List[str]:
    """Convert a dict into `--k v` arguments."""
    args: List[str] = []
    for k, v in d.items():
        if v is None:
            continue
        args.append(f"--{k}")
        args.append(str(v))
    return args


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def best_ckpt_path(log_dir: str, data: str, model: str = "stamp") -> str:
    return os.path.join(log_dir, f"best_model_{data}_{model}.pth")


def suite_experiments(suite: str) -> List[Exp]:
    """Define experiment suites.

    All suites are defined relative to a *Full Mamba* baseline unless stated.
    """

    # ---------------- framework-level ablations ----------------
    if suite == "framework":
        return [
            Exp(
                exp_id="E0_STAMP",
                desc="Original STAMP (pred=gat, recon=ae)",
                run_overrides={"pred_model": "gat", "recon_model": "ae"},
                test_overrides={},
            ),
            Exp(
                exp_id="E1_MambaPred",
                desc="Only replace prediction branch with Mamba (pred=mamba, recon=ae)",
                run_overrides={"pred_model": "mamba", "recon_model": "ae"},
                test_overrides={},
            ),
            Exp(
                exp_id="E2_MambaRecon",
                desc="Only replace recon branch with Mamba (pred=gat, recon=mamba)",
                run_overrides={"pred_model": "gat", "recon_model": "mamba"},
                test_overrides={},
            ),
            Exp(
                exp_id="E3_FullMamba",
                desc="Full model (pred=mamba, recon=mamba)",
                run_overrides={"pred_model": "mamba", "recon_model": "mamba"},
                test_overrides={},
            ),
        ]

    # ---------------- forecast branch ablations ----------------
    if suite == "forecast":
        base = {"pred_model": "mamba", "recon_model": "mamba"}
        return [
            Exp(
                exp_id="F0_base",
                desc="Forecast baseline (bi-Mamba + FFN)",
                run_overrides={**base},
                test_overrides={},
            ),
            Exp(
                exp_id="F1_uniMamba",
                desc="w/o bi-Mamba (unidirectional Mamba)",
                run_overrides={**base, "mamba_bidirectional": False},
                test_overrides={},
            ),
            Exp(
                exp_id="F2_noFFN",
                desc="w/o FFN inside Mamba blocks",
                run_overrides={**base, "mamba_use_ffn": False},
                test_overrides={},
            ),
            Exp(
                exp_id="F3_noNorm",
                desc="w/o input normalization in forecast branch",
                run_overrides={**base, "mamba_use_norm": False},
                test_overrides={},
            ),
            Exp(
                exp_id="F4_noLastResidual",
                desc="w/o last-value residual shortcut",
                run_overrides={**base, "mamba_use_last_residual": False},
                test_overrides={},
            ),
        ]

    # ---------------- recon branch ablations ----------------
    if suite == "recon":
        base = {"pred_model": "mamba", "recon_model": "mamba"}
        return [
            Exp(
                exp_id="R0_base",
                desc="Recon baseline (multi-scale + global+local + bi-Mamba)",
                run_overrides={**base},
                test_overrides={},
            ),
            Exp(
                exp_id="R1_1scale",
                desc="w/o multi-scale (num_scales=1)",
                run_overrides={**base, "recon_num_scales": 1},
                test_overrides={},
            ),
            Exp(
                exp_id="R2_globalOnly",
                desc="w/o local branch (global only)",
                run_overrides={**base, "recon_use_local": False},
                test_overrides={},
            ),
            Exp(
                exp_id="R3_localOnly",
                desc="w/o global branch (local only)",
                run_overrides={**base, "recon_use_global": False},
                test_overrides={},
            ),
            Exp(
                exp_id="R4_uniMamba",
                desc="w/o bi-Mamba in recon (unidirectional)",
                run_overrides={**base, "recon_bidirectional": False},
                test_overrides={},
            ),
            Exp(
                exp_id="R5_k3",
                desc="local kernels = (3)",
                run_overrides={**base, "recon_local_kernels": "3"},
                test_overrides={},
            ),
        ]

    # ---------------- training/coupling ablations ----------------
    if suite == "training":
        base = {"pred_model": "mamba", "recon_model": "mamba"}
        return [
            Exp(
                exp_id="T0_stampMinMax",
                desc="Original STAMP min-max training (use_adv_train=True, adv_loss_mode=stamp)",
                run_overrides={**base, "use_adv_train": True, "adv_loss_mode": "stamp"},
                test_overrides={},
            ),
            Exp(
                exp_id="T1_noAdvTrain",
                desc="w/o adversarial optimization module (use_adv_train=False)",
                run_overrides={**base, "use_adv_train": False},
                test_overrides={"test_gamma": 0.0},
            ),
            Exp(
                exp_id="T2_constantLoss",
                desc="constant loss weights (adv_loss_mode=constant)",
                run_overrides={**base, "use_adv_train": True, "adv_loss_mode": "constant", "lambda_pred": 5.0, "lambda_ae": 3.0, "lambda_adv": 1.0},
                test_overrides={},
            ),
            Exp(
                exp_id="T3_freezeOtherFalse",
                desc="do not freeze the other branch during coupled updates (adv_freeze_other=False)",
                run_overrides={**base, "adv_freeze_other": False},
                test_overrides={},
            ),
        ]

    raise ValueError(f"Unknown suite: {suite}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, choices=["framework", "forecast", "recon", "training"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="exp/ablations")
    parser.add_argument("--seed", type=int, default=0)

    # common training args (override as needed)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--n_pred", type=int, default=3)

    # common testing args
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--test_alpha", type=float, default=0.5)
    parser.add_argument("--test_beta", type=float, default=0.5)
    parser.add_argument("--test_gamma", type=float, default=0.1)
    parser.add_argument("--score_method", type=str, default="max", choices=["max", "sum", "mean", "all"])

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument("--extra_run_args", type=str, default="")
    parser.add_argument("--extra_test_args", type=str, default="")

    args = parser.parse_args()

    exps = suite_experiments(args.suite)
    ensure_dir(args.out_root)

    summary_rows = []

    for exp in exps:
        log_dir = os.path.join(args.out_root, exp.exp_id)
        ensure_dir(log_dir)

        # -------- run.py --------
        run_args = {
            "data": args.data,
            "dataset": args.dataset,
            "log_dir": log_dir,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "window_size": args.window_size,
            "n_pred": args.n_pred,
        }
        run_args.update(exp.run_overrides)

        run_cmd_list = [sys.executable, "run.py"] + dict_to_cli(run_args) + shlex.split(args.extra_run_args)

        if not args.skip_train:
            run_cmd(run_cmd_list, dry_run=args.dry_run)

        # -------- test.py --------
        ckpt = best_ckpt_path(log_dir, args.data, model="stamp")
        result_json = os.path.join(log_dir, "result.json")

        test_args = {
            "data": args.data,
            "dataset": args.dataset,
            "log_dir": log_dir,
            "seed": args.seed,
            "window_size": args.window_size,
            "n_pred": args.n_pred,
            "topk": args.topk,
            "test_alpha": args.test_alpha,
            "test_beta": args.test_beta,
            "test_gamma": args.test_gamma,
            "score_method": args.score_method,
            "load_checkpoint": ckpt,
            "save_result_json": result_json,
            "exp_id": exp.exp_id,
        }
        test_args.update(exp.run_overrides)   # keep model selection consistent
        test_args.update(exp.test_overrides)  # method-specific overrides

        test_cmd_list = [sys.executable, "test.py"] + dict_to_cli(test_args) + shlex.split(args.extra_test_args)

        if not args.skip_test:
            run_cmd(test_cmd_list, dry_run=args.dry_run)

        # -------- collect --------
        if (not args.dry_run) and os.path.exists(result_json):
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                # pick 'max' if exists else first
                res = payload.get("results", {})
                if "max" in res:
                    pick = res["max"]
                else:
                    pick = next(iter(res.values())) if len(res) else {}

                summary_rows.append(
                    {
                        "exp_id": exp.exp_id,
                        "desc": exp.desc,
                        "f1": pick.get("best_f1"),
                        "precision": pick.get("precision"),
                        "recall": pick.get("recall"),
                    }
                )
            except Exception as e:
                print(f"[WARN] failed to parse {result_json}: {e}")

    # write summary
    if (not args.dry_run) and len(summary_rows) > 0:
        summary_path = os.path.join(args.out_root, f"summary_{args.suite}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved summary] {summary_path}")


if __name__ == "__main__":
    main()
