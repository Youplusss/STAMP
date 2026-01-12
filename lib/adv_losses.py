"""Adversarial / coupled training losses for STAMP.

This file provides *drop-in* loss helpers used by trainer.py.

Why this exists
---------------
STAMP's original code implements a min-max game using MSE reconstruction losses:
  - pred branch tries to minimize prediction error and also make the AE reconstruct
    the generated window well.
  - AE tries to reconstruct real windows well, but reconstruct generated windows poorly.

That min-max game is known to be unstable (especially with strong backbones like Mamba).
So we provide several stabilized variants (hinge/softplus/BEGAN-style etc.), plus
*scale-aware* knobs (margin_floor, tau_mode) to prevent blow-ups.

Notation
--------
- L_pred: prediction loss on future steps (MSE)
- L_real: AE reconstruction loss on real window
- L_fake: AE reconstruction loss on generated window

We treat the AE as an *energy* model: lower energy means more "real".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------- scheduling --------------------------

def ramp_weight(epoch: int, warmup_epochs: int, ramp_epochs: int, max_weight: float) -> float:
    """Warmup + linear ramp for adversarial weights.

    epoch: 1-based epoch index.
    warmup_epochs: keep weight at 0 for the first `warmup_epochs` epochs.
    ramp_epochs: linearly increase from 0 -> max_weight over `ramp_epochs` epochs.
    """

    if max_weight <= 0:
        return 0.0

    e = int(epoch)
    warm = max(int(warmup_epochs), 0)
    ramp = max(int(ramp_epochs), 1)

    if e <= warm:
        return 0.0

    t = (e - warm) / float(ramp)
    t = max(0.0, min(1.0, t))
    return float(max_weight) * t


# -------------------------- utilities --------------------------

def _safe_clamp_min(x: torch.Tensor, min_val: float) -> torch.Tensor:
    if min_val <= 0:
        return x
    return torch.clamp(x, min=min_val)


def energy_transform(loss: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """Optional monotonic transform for losses used as *energies*.

    This does NOT change the base reconstruction loss used to train the AE on real windows.
    It only affects the adversarial *gap* and the penalties.

    Useful when L_fake can become very large (e.g., AE learns to strongly reject generated windows),
    which may destabilize gradients.
    """

    m = (mode or "none").lower()
    if m in ("none", "identity"):
        return loss

    # losses should be >= 0, but numerical issues can introduce tiny negatives
    loss_pos = torch.clamp(loss, min=0.0)

    if m in ("log", "log1p"):
        return torch.log1p(loss_pos)

    if m in ("sqrt", "root"):
        return torch.sqrt(loss_pos + 1e-12)

    raise ValueError(f"Unknown energy_transform: {mode}")


def _margin_value(margin: float, margin_mode: str, ref_energy: torch.Tensor, margin_floor: float) -> torch.Tensor:
    """Compute a tensor margin value (broadcastable) on the same device."""

    mode = (margin_mode or "rel").lower()

    if mode == "abs":
        m = torch.as_tensor(float(margin), device=ref_energy.device, dtype=ref_energy.dtype)
    elif mode == "rel":
        # enforce: gap >= margin * E_real   <=>   E_fake >= (1+margin) * E_real
        m = float(margin) * ref_energy.detach()
    else:
        raise ValueError(f"Unknown margin_mode: {margin_mode}")

    if margin_floor and margin_floor > 0:
        mf = torch.as_tensor(float(margin_floor), device=ref_energy.device, dtype=ref_energy.dtype)
        m = torch.maximum(m, mf)

    return m


def _tau_value(tau: float, tau_mode: str, ref_energy: torch.Tensor, tau_floor: float) -> torch.Tensor:
    """Compute a tensor temperature (tau) on the same device."""

    mode = (tau_mode or "abs").lower()

    if mode == "abs":
        t = torch.as_tensor(float(tau), device=ref_energy.device, dtype=ref_energy.dtype)
    elif mode == "rel":
        # scale tau to the current loss magnitude (prevents tau=1.0 from being huge when losses ~1e-2)
        t = float(tau) * ref_energy.detach()
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")

    # avoid divide-by-zero / overly sharp temperatures
    floor = float(tau_floor) if tau_floor is not None else 0.0
    if floor > 0:
        tf = torch.as_tensor(floor, device=ref_energy.device, dtype=ref_energy.dtype)
        t = torch.maximum(t, tf)

    return t


# -------------------------- pred objective --------------------------

def pred_total_loss(
    pred_loss: torch.Tensor,
    fake_recon_loss: torch.Tensor,
    lambda_pred: float,
    *,
    real_recon_loss: Optional[torch.Tensor] = None,
    pred_objective: str = "adv",
    margin: float = 0.2,
    margin_mode: str = "rel",
    margin_floor: float = 0.0,
    margin_high: float = -1.0,
    tau: float = 1.0,
    tau_mode: str = "abs",
    tau_floor: float = 1e-4,
    energy_transform_mode: str = "none",
    auto_balance: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total loss for pred branch.

    Parameters
    ----------
    pred_loss:
        Forecasting loss (e.g., MSE on future steps).
    fake_recon_loss:
        AE reconstruction error on generated windows ("fake" energy).
    lambda_pred:
        Weight for the adversarial term (after warmup/ramp).
    real_recon_loss:
        AE reconstruction error on real windows (needed for gap-based objectives).
        Should be detached or at least does not require grad w.r.t pred parameters.
    pred_objective:
        - 'adv'       : pred_loss + lambda * E_fake
        - 'gap_relu'  : pred_loss + lambda * relu(E_fake - E_real)
        - 'gap_hinge' : pred_loss + lambda * relu((E_fake - E_real) - margin_val)

    Returns
    -------
    total_loss, penalty, gap  (all torch.Tensors)
    """

    lam = float(lambda_pred)
    if lam <= 0:
        # keep return types consistent
        z = torch.zeros_like(pred_loss)
        return pred_loss, z, z

    obj = (pred_objective or "adv").lower()

    E_fake = energy_transform(fake_recon_loss, energy_transform_mode)

    if obj == "adv":
        penalty = E_fake
        gap = torch.zeros_like(penalty)
    else:
        if real_recon_loss is None:
            raise ValueError(f"pred_objective='{obj}' requires real_recon_loss")

        E_real = energy_transform(real_recon_loss, energy_transform_mode)
        gap = E_fake - E_real

        if obj == "gap_relu":
            penalty = F.relu(gap)
        elif obj == "gap_hinge":
            margin_val = _margin_value(margin, margin_mode, E_real, margin_floor)
            penalty = F.relu(gap - margin_val)
        else:
            raise ValueError(f"Unknown pred_objective: {pred_objective}")

    # optional auto-balancing to keep the adversarial term on the same scale as pred_loss
    lam_eff = lam
    if auto_balance:
        denom = penalty.detach().abs() + eps
        scale = (pred_loss.detach().abs() / denom).clamp(0.1, 10.0)
        lam_eff = float(lam) * float(scale.item())

    total = pred_loss + lam_eff * penalty
    return total, penalty, gap


# -------------------------- AE objective --------------------------

def ae_total_loss(
    real_recon_loss: torch.Tensor,
    fake_recon_loss: torch.Tensor,
    *,
    mode: str = "hinge",
    lambda_ae: float = 1.0,
    margin: float = 0.2,
    margin_mode: str = "rel",
    margin_floor: float = 0.0,
    margin_high: float = -1.0,
    tau: float = 1.0,
    tau_mode: str = "abs",
    tau_floor: float = 1e-4,
    energy_transform_mode: str = "none",
    auto_balance: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total loss for AE branch.

    The AE always minimizes `real_recon_loss` (fit normal data), plus an adversarial penalty that
    encourages *separation* between real and generated windows.

    Supported modes
    ---------------
    - 'legacy'    : L = L_real - lambda * E_fake  (original STAMP min-max)
    - 'hinge'     : L = L_real + lambda * relu(margin - (E_fake - E_real))
    - 'softplus'  : smooth hinge using softplus (may require proper tau scaling)
    - 'softplus0' : softplus hinge with zeroed baseline (more stable when tau is large)
    - 'exp'       : exponential penalty (aggressive; use with care)

    Returns
    -------
    total_loss, penalty, gap, margin_val, tau_val

    Notes
    -----
    - The returned `gap` is computed on transformed energies: gap = E_fake - E_real.
    - `margin_val` and `tau_val` are tensors on the right device/dtype.
    """

    lam = float(lambda_ae)
    if lam < 0:
        raise ValueError("lambda_ae must be >= 0")

    m = (mode or "hinge").lower()

    E_real = energy_transform(real_recon_loss, energy_transform_mode)
    E_fake = energy_transform(fake_recon_loss, energy_transform_mode)
    gap = E_fake - E_real

    margin_val = _margin_value(margin, margin_mode, E_real, margin_floor)
    tau_val = _tau_value(tau, tau_mode, E_real, tau_floor)

    if lam == 0:
        z = torch.zeros_like(E_real)
        return real_recon_loss, z, gap, margin_val, tau_val

    if m == "legacy":
        # original STAMP: minimize L_real while maximizing E_fake (by subtracting it)
        penalty = -E_fake

    elif m == "hinge":
        # want: gap >= margin
        penalty = F.relu(margin_val - gap)

    elif m == "band":
        # keep gap within [margin, margin_high] to avoid runaway fake energy
        if margin_high <= 0:
            raise ValueError("adv_mode='band' requires margin_high > 0 (set --adv_margin_high)")
        margin_high_val = _margin_value(margin_high, margin_mode, E_real, margin_floor=0.0)
        margin_high_val = torch.maximum(margin_high_val, margin_val)
        penalty = F.relu(margin_val - gap) + F.relu(gap - margin_high_val)

    elif m == "softplus":
        # softplus approximation of relu (note softplus(0)=log(2))
        x = (margin_val - gap) / tau_val
        penalty = tau_val * F.softplus(x)

    elif m == "softplus0":
        # smoother hinge with *zero* penalty at x=0 (gap==margin)
        # penalty = relu(tau*softplus(x) - tau*log(2))
        x = (margin_val - gap) / tau_val
        penalty = tau_val * F.softplus(x) - (tau_val * math.log(2.0))
        penalty = F.relu(penalty)

    elif m == "exp":
        x = (margin_val - gap) / tau_val
        x = torch.clamp(x, max=50.0)
        penalty = tau_val * torch.exp(x)

    else:
        raise ValueError(f"Unknown ae adv mode: {mode}")

    # optional auto-balancing to keep penalty on a similar scale as real_recon_loss
    lam_eff = lam
    if auto_balance:
        denom = penalty.detach().abs() + eps
        scale = (real_recon_loss.detach().abs() / denom).clamp(0.1, 10.0)
        lam_eff = float(lam) * float(scale.item())

    total = real_recon_loss + lam_eff * penalty
    return total, penalty, gap, margin_val, tau_val
