# -*- coding: utf-8 -*-
"""Adversarial / coupled losses for STAMP.

This repo trains two branches:

1) Forecast ("pred") branch: predicts the future segment.
2) Reconstruction ("AE") branch: reconstructs the whole window.

STAMP-style adversarial coupling treats the reconstruction loss on a **generated window**
(`generate = concat(history, pred(history))`) as an *energy*. The prediction branch acts
like a generator trying to make that energy low, while the reconstruction branch acts
like an energy-based discriminator trying to assign higher energy to generated windows.

The original ("legacy") objective is a min-max form:

    pred: minimize  L_pred + λ_pred * L_adv
    AE  : minimize  L_real - λ_ae   * L_adv

where L_adv is the reconstruction error on generated windows. This objective can be
unstable for high-capacity backbones (e.g. Mamba) because the AE part is *unbounded*
in the sense that it always benefits from increasing L_adv.

This module provides *bounded* alternatives that match the same intuition but avoid
runaway behaviour:

* hinge:   AE minimizes  L_real + λ * relu(margin - (L_adv - L_real))
           -> AE only pushes L_adv above L_real+margin, then stops.
* softplus: smooth hinge (same constraint, smoother gradient).
* exp:     AE minimizes  L_real + λ * exp(-(L_adv - L_real)/tau)
           -> encourages a large gap, but saturates.

These are inspired by energy-based GAN / margin-based energy objectives.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F


AdvMode = Literal["legacy", "hinge", "softplus", "exp"]
MarginMode = Literal["abs", "rel"]


def ramp_weight(epoch: int, warmup_epochs: int, ramp_epochs: int, max_weight: float) -> float:
    """Warmup + linear ramp schedule.

    - For epoch <= warmup_epochs: return 0.
    - For warmup_epochs < epoch <= warmup_epochs + ramp_epochs: linearly increase to max_weight.
    - After that: return max_weight.
    """

    if max_weight <= 0:
        return 0.0

    warmup_epochs = int(max(warmup_epochs, 0))
    ramp_epochs = int(max(ramp_epochs, 0))
    e = int(epoch)

    if e <= warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return float(max_weight)

    t = (e - warmup_epochs) / float(ramp_epochs)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return float(max_weight) * t


def pred_total_loss(pred_loss: torch.Tensor, adv_loss: torch.Tensor, lambda_pred: float) -> torch.Tensor:
    """Prediction-branch objective."""
    if lambda_pred <= 0:
        return pred_loss
    return pred_loss + float(lambda_pred) * adv_loss


def ae_total_loss(
    ae_loss: torch.Tensor,
    adv_loss: torch.Tensor,
    *,
    mode: AdvMode = "hinge",
    lambda_ae: float = 1.0,
    margin: float = 0.1,
    margin_mode: MarginMode = "rel",
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction-branch (AE) objective.

    Returns
    -------
    total_loss:
        The scalar to backprop.
    penalty:
        The *unweighted* adversarial penalty term (for logging).
    gap:
        gap = adv_loss - ae_loss.
    margin_val:
        The margin used (tensor on the same device).
    """

    lambda_ae = float(lambda_ae)
    if lambda_ae <= 0:
        zero = torch.zeros_like(ae_loss)
        gap = adv_loss - ae_loss
        margin_val = zero
        return ae_loss, zero, gap, margin_val

    # gap > 0 means generated window is reconstructed worse than real window.
    gap = adv_loss - ae_loss

    if margin_mode == "abs":
        margin_val = torch.as_tensor(float(margin), device=ae_loss.device, dtype=ae_loss.dtype)
    else:
        # relative to current scale of ae_loss (detach to avoid changing AE gradients)
        margin_val = float(margin) * ae_loss.detach()

    if mode == "legacy":
        # unbounded min-max (can be unstable): minimize ae_loss - λ * adv_loss
        penalty = -adv_loss
        total = ae_loss + lambda_ae * penalty
        return total, penalty, gap, margin_val

    if mode == "hinge":
        # Encourage gap >= margin; once satisfied, penalty becomes 0.
        penalty = F.relu(margin_val - gap)
        total = ae_loss + lambda_ae * penalty
        return total, penalty, gap, margin_val

    tau = float(tau)
    tau = tau if tau > 1e-8 else 1.0

    if mode == "softplus":
        # Smooth hinge: tau * log(1 + exp((margin-gap)/tau))
        penalty = F.softplus((margin_val - gap) / tau) * tau
        total = ae_loss + lambda_ae * penalty
        return total, penalty, gap, margin_val

    if mode == "exp":
        # Saturating penalty: exp(-(gap)/tau)
        # If gap is large positive, penalty -> 0. If gap is negative, penalty grows but we clamp.
        inv = (-gap / tau).clamp(min=-50.0, max=50.0)
        penalty = torch.exp(inv)
        total = ae_loss + lambda_ae * penalty
        return total, penalty, gap, margin_val

    raise ValueError(f"Unknown adv mode: {mode}")
