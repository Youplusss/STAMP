# -*- coding: utf-8 -*-
"""Automate recon ablation experiments driven by utils/params.txt.

Contract
--------
Inputs:
- A params file where each non-empty line contains 4 fields:
    num_mamba_layers  lss_residual  local_conv_variant  kernel_sizes
  Example:
    3 True dwconv_1x1 3,5

Behavior:
- For each line (in order), run 3 training commands (SWaT/WADI/MSL), waiting for
  each to finish, then run 3 test commands, waiting for each to finish.
- Each run sets env vars to override ReconAblationConfig at runtime
  (implemented in model/mamba_recon.py).
- Ensure branch/test.py produces *_test.log (we already patched branch/test.py
  to do so).

Outputs:
- Standard logs under expe_branch/log
- You may optionally redirect stdout/stderr into a dedicated sweep folder.

Why not nohup?
-------------
Using nohup + background makes it hard to "reliably determine completion".
This script instead runs commands synchronously (blocking) and captures exit
codes, which is the most robust way to ensure completion.

If you truly need background execution on a remote server, run this python
script under nohup/screen/tmux.

Usage:
  python utils/run_ablation_from_params.py --params utils/params.txt --gpu_id 0

Run:
nohup python utils/run_ablation_from_params.py \
  --params utils/params.txt \
  --gpu_id 0 \
  --epochs 100 \
  --batch_size 128 \
  --log_dir expe_branch \
  --msl_down_len 10 \
  > log/ablation_controller.log 2>&1 &

"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParamRow:
    num_mamba_layers: int
    lss_residual: bool
    local_conv_variant: str
    kernel_sizes: str  # comma-separated

    def to_env(self) -> dict[str, str]:
        return {
            # IMPORTANT: enable env overrides explicitly (pollution guard in model/mamba_recon.py)
            "STAMP_RECON_APPLY_ENV_OVERRIDES": "1",
            "STAMP_RECON_NUM_MAMBA_LAYERS": str(self.num_mamba_layers),
            "STAMP_RECON_LSS_RESIDUAL": "1" if self.lss_residual else "0",
            "STAMP_RECON_LOCAL_CONV_VARIANT": self.local_conv_variant,
            "STAMP_RECON_KERNEL_SIZES": self.kernel_sizes,
        }

    def tag(self) -> str:
        ks = self.kernel_sizes.replace(",", "-")
        return f"mi{self.num_mamba_layers}_skip{int(self.lss_residual)}_{self.local_conv_variant}_k{ks}"


def _parse_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean: {s!r}")


def read_params(path: str) -> list[ParamRow]:
    rows: list[ParamRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"params file format error at line {ln}: expected 4 fields, got {len(parts)}: {line}")
            num_mi = int(parts[0])
            lss_res = _parse_bool(parts[1])
            variant = parts[2]
            ks = parts[3]
            rows.append(ParamRow(num_mi, lss_res, variant, ks))
    return rows


def run_one(cmd: list[str], *, env: dict[str, str], cwd: str | None = None, log_file: str | None = None) -> None:
    """Run a command synchronously and raise if it fails."""

    pretty = " ".join(cmd)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN: {pretty}")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n# CMD: {pretty}\n")
            f.flush()
            subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, env=env, cwd=cwd, check=True)


def _wait_all(procs: list[tuple[str, subprocess.Popen]], *, poll_s: float = 10.0) -> dict[str, int]:
    """Wait for all processes to finish. Return mapping name->returncode."""
    alive = {name: p for name, p in procs}
    rc: dict[str, int] = {}
    while alive:
        done = []
        for name, p in alive.items():
            r = p.poll()
            if r is not None:
                rc[name] = int(r)
                done.append(name)
        for name in done:
            alive.pop(name, None)
        if alive:
            time.sleep(poll_s)
    return rc


def _launch(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    log_file: Optional[str],
) -> tuple[str, subprocess.Popen]:
    """Launch a subprocess and redirect stdout/stderr to log_file if provided."""
    pretty = " ".join(cmd)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] LAUNCH[{name}]: {pretty}")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f = open(log_file, "a", encoding="utf-8")
        f.write(f"\n\n# CMD: {pretty}\n")
        f.flush()
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        # attach for later close
        p._stamp_log_handle = f  # type: ignore[attr-defined]
    else:
        p = subprocess.Popen(cmd, env=env, cwd=cwd)
    return name, p


def _close_logs(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for _name, p in procs:
        f = getattr(p, "_stamp_log_handle", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run recon ablation sweep from utils/params.txt")
    p.add_argument("--params", type=str, default=os.path.join("utils", "params.txt"))
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--log_dir", type=str, default="expe_branch")
    p.add_argument("--msl_down_len", type=int, default=10)

    p.add_argument(
        "--stdout_dir",
        type=str,
        default=os.path.join("log", "ablation_sweep"),
        help="Optional: directory to append per-run stdout/stderr logs (independent of expe_branch/log)",
    )
    p.add_argument("--no_stdout_logs", action="store_true", help="Disable redirecting stdout/stderr into --stdout_dir")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # IMPORTANT: this script does NOT (and cannot) export variables to your parent shell.
    # Instead, we pass env vars to child processes only. This achieves the same effect
    # as `export STAMP_RECON_APPLY_ENV_OVERRIDES=1` for the duration of the run.

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    branch_run = os.path.join(repo_root, "branch", "run.py")
    branch_test = os.path.join(repo_root, "branch", "test.py")

    rows = read_params(os.path.join(repo_root, args.params) if not os.path.isabs(args.params) else args.params)
    if not rows:
        print("No params found. Nothing to do.")
        return

    # 3 datasets as requested
    train_jobs = [
        ("SWaT", []),
        ("WADI", []),
        ("MSL", ["--down_len", str(args.msl_down_len)]),
    ]

    test_jobs = [
        ("SWaT", []),
        ("WADI", []),
        ("MSL", ["--down_len", str(args.msl_down_len)]),
    ]

    expe_log_dir = os.path.join(repo_root, str(args.log_dir), "log")

    for i, row in enumerate(rows, start=1):
        tag = row.tag()
        print("\n" + "=" * 80)
        print(f"[Ablation] {i}/{len(rows)}: {tag}")
        print("=" * 80)

        env = os.environ.copy()
        env.update(row.to_env())

        if not args.no_stdout_logs:
            os.makedirs(os.path.join(repo_root, args.stdout_dir), exist_ok=True)

        # -----------------
        # 1) TRAIN (3 datasets) in parallel
        # -----------------
        train_procs: list[tuple[str, subprocess.Popen]] = []
        for ds, extra in train_jobs:
            out_log = None
            if not args.no_stdout_logs:
                out_log = os.path.join(repo_root, args.stdout_dir, f"train_{i:03d}_{ds}_{tag}.log")

            cmd = [
                "python",
                branch_run,
                "--data",
                ds,
                "--branch",
                "mamba_recon",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--gpu_id",
                str(args.gpu_id),
                "--log_dir",
                str(args.log_dir),
            ] + extra

            train_procs.append(_launch(f"train/{ds}", cmd, env=env, cwd=repo_root, log_file=out_log))

        train_rc = _wait_all(train_procs, poll_s=10.0)
        _close_logs(train_procs)

        bad_train = {k: v for k, v in train_rc.items() if v != 0}
        if bad_train:
            raise RuntimeError(f"Training failed for ablation {tag}: {bad_train}")

        # -----------------
        # 2) TEST (3 datasets) in parallel (only after ALL train are done)
        # -----------------
        # Snapshot existing *_test.log count so we can sanity-check that new test logs are created.
        before_test_logs = set()
        if os.path.isdir(expe_log_dir):
            before_test_logs = {fn for fn in os.listdir(expe_log_dir) if fn.lower().endswith("_test.log")}

        test_procs: list[tuple[str, subprocess.Popen]] = []
        for ds, extra in test_jobs:
            out_log = None
            if not args.no_stdout_logs:
                out_log = os.path.join(repo_root, args.stdout_dir, f"test_{i:03d}_{ds}_{tag}.log")

            cmd = [
                "python",
                branch_test,
                "--data",
                ds,
                "--branch",
                "mamba_recon",
                "--batch_size",
                str(args.batch_size),
                "--gpu_id",
                str(args.gpu_id),
                "--log_dir",
                str(args.log_dir),
            ] + extra

            test_procs.append(_launch(f"test/{ds}", cmd, env=env, cwd=repo_root, log_file=out_log))

        test_rc = _wait_all(test_procs, poll_s=5.0)
        _close_logs(test_procs)

        bad_test = {k: v for k, v in test_rc.items() if v != 0}
        if bad_test:
            raise RuntimeError(f"Testing failed for ablation {tag}: {bad_test}")

        # Ensure test logs are produced (core requirement)
        after_test_logs = set()
        if os.path.isdir(expe_log_dir):
            after_test_logs = {fn for fn in os.listdir(expe_log_dir) if fn.lower().endswith("_test.log")}
        new_logs = after_test_logs - before_test_logs
        if not new_logs:
            raise RuntimeError(
                f"No new *_test.log detected under {expe_log_dir} for ablation {tag}. "
                "Check branch/test.py logging or permissions."
            )

    print("\nAll ablations completed.")


if __name__ == "__main__":
    main()
