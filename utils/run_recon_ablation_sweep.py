# -*- coding: utf-8 -*-
"""Run reconstruction-branch ablation sweeps without editing source code.

This launcher varies these 4 knobs (as requested):
  - num_mamba_layers
  - lss_residual
  - local_conv_variant
  - kernel_sizes

Under the hood it uses environment variable overrides in `model/mamba_recon.py`:
  - STAMP_RECON_NUM_MAMBA_LAYERS
  - STAMP_RECON_LSS_RESIDUAL
  - STAMP_RECON_LOCAL_CONV_VARIANT
  - STAMP_RECON_KERNEL_SIZES

It then launches `branch/run.py` (train) and optionally `branch/test.py` (test)
for each configuration.

Example:
  python utils/run_recon_ablation_sweep.py --data SWaT --gpu_id 0 --do_test True

Note:
- This script uses subprocess and is OS-agnostic.
- It doesn't require modifying any Python source during the sweep.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationCfg:
    num_mamba_layers: int
    lss_residual: bool
    local_conv_variant: str
    kernel_sizes: str  # comma-separated

    def to_env(self) -> dict[str, str]:
        return {
            "STAMP_RECON_NUM_MAMBA_LAYERS": str(self.num_mamba_layers),
            "STAMP_RECON_LSS_RESIDUAL": "1" if self.lss_residual else "0",
            "STAMP_RECON_LOCAL_CONV_VARIANT": self.local_conv_variant,
            "STAMP_RECON_KERNEL_SIZES": self.kernel_sizes,
        }

    def tag(self) -> str:
        ks = self.kernel_sizes.replace(",", "-")
        return f"mi{self.num_mamba_layers}_skip{int(self.lss_residual)}_{self.local_conv_variant}_k{ks}"


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep ReconAblationConfig knobs for branch mamba_recon")
    p.add_argument("--data", type=str, default="SWaT")
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--log_dir", type=str, default="expe_branch")
    p.add_argument("--do_test", type=eval, default=True, help="whether to run branch/test.py after training")

    p.add_argument("--num_mamba_layers", type=str, default="1,2,3")
    p.add_argument("--lss_residual", type=str, default="0,1")
    p.add_argument("--local_conv_variant", type=str, default="dwconv_1x1,only_dwconv")
    p.add_argument("--kernel_sizes", type=str, default="3,5;5,7")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    n_layers = [int(x) for x in str(args.num_mamba_layers).split(",") if x.strip() != ""]
    residuals = [bool(int(x)) for x in str(args.lss_residual).split(",") if x.strip() != ""]
    variants = [x.strip() for x in str(args.local_conv_variant).split(",") if x.strip() != ""]
    kernels = [x.strip() for x in str(args.kernel_sizes).split(";") if x.strip() != ""]

    combos = [AblationCfg(a, b, c, d) for a, b, c, d in itertools.product(n_layers, residuals, variants, kernels)]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    branch_run = os.path.join(repo_root, "branch", "run.py")
    branch_test = os.path.join(repo_root, "branch", "test.py")

    for cfg in combos:
        env = os.environ.copy()
        env.update(cfg.to_env())

        _run(
            [
                "python",
                branch_run,
                "--data",
                str(args.data),
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
            ],
            env,
        )

        if bool(args.do_test):
            _run(
                [
                    "python",
                    branch_test,
                    "--data",
                    str(args.data),
                    "--branch",
                    "mamba_recon",
                    "--gpu_id",
                    str(args.gpu_id),
                    "--batch_size",
                    str(args.batch_size),
                    "--log_dir",
                    str(args.log_dir),
                ],
                env,
            )


if __name__ == "__main__":
    main()
