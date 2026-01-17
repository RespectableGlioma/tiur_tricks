from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TIURState:
    """Holds previous checkpoint statistics for incremental TIUR estimation."""

    prev_mean: Optional[Dict[str, torch.Tensor]] = None
    prev_var: Optional[Dict[str, torch.Tensor]] = None
    prev_loss_mean: Optional[float] = None
    prev_step: Optional[int] = None

    directed_integral: float = 0.0
    churn_integral: float = 0.0


def _param_mean_var_diagonal(
    params_by_model: List[Dict[str, torch.nn.Parameter]],
    *,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute diagonal mean and variance across models for each parameter.

    Returns CPU float32 tensors.
    """
    names = params_by_model[0].keys()
    mean: Dict[str, torch.Tensor] = {}
    var: Dict[str, torch.Tensor] = {}

    for name in names:
        vals = [pm[name].detach().float().cpu() for pm in params_by_model]
        stack = torch.stack(vals, dim=0)
        m = stack.mean(dim=0)
        v = stack.var(dim=0, unbiased=False).add_(eps)
        mean[name] = m
        var[name] = v

    return mean, var


def update_tiur(
    state: TIURState,
    *,
    step: int,
    losses: List[float],
    params_by_model: List[Dict[str, torch.nn.Parameter]],
    eps: float = 1e-12,
) -> Tuple[TIURState, Dict[str, float]]:
    """Update TIUR metrics at a checkpoint.

    Args:
      state: running state
      step: current training step (integer time)
      losses: per-replicate eval loss at this checkpoint
      params_by_model: list of dicts {name: parameter} for each replicate model
      eps: numerical epsilon for covariance / divisions

    Returns:
      (new_state, log_row)
    """

    loss_mean = float(torch.tensor(losses).mean().item())
    loss_std = float(torch.tensor(losses).std(unbiased=False).item())

    mean, var = _param_mean_var_diagonal(params_by_model, eps=eps)

    I_mu = float("nan")
    I_sigma = float("nan")
    I_total = float("nan")
    churn_frac = float("nan")
    bound = float("nan")
    dLdt_abs = float("nan")
    efficiency = float("nan")

    if state.prev_mean is not None and state.prev_var is not None and state.prev_step is not None:
        dt = float(step - state.prev_step)
        if dt <= 0:
            raise ValueError("Non-positive dt in TIUR update")

        I_mu_acc = 0.0
        I_sigma_acc = 0.0

        # Diagonal Gaussian approximation:
        #   I_mu = sum_j (dmu_j^2 / var_j)
        #   I_sigma = 1/2 sum_j (dvar_j^2 / var_j^2)
        # where dmu/dt and dvar/dt are finite differences.
        for name in mean.keys():
            prev_m = state.prev_mean[name]
            prev_v = state.prev_var[name]
            m = mean[name]
            v = var[name]

            dmu = (m - prev_m) / dt
            dvar = (v - prev_v) / dt

            I_mu_acc += float((dmu.square() / v).sum().item())
            I_sigma_acc += float(0.5 * (dvar.square() / (v.square())).sum().item())

        I_mu = I_mu_acc
        I_sigma = I_sigma_acc
        I_total = I_mu + I_sigma
        churn_frac = I_sigma / (I_total + 1e-30)

        if state.prev_loss_mean is not None:
            dLdt = (loss_mean - state.prev_loss_mean) / dt
            dLdt_abs = abs(float(dLdt))

        bound = loss_std * math.sqrt(I_total + 1e-30)
        efficiency = dLdt_abs / (bound + 1e-30)

        state.directed_integral += I_mu * dt
        state.churn_integral += I_sigma * dt

    # Update state
    state.prev_mean = mean
    state.prev_var = var
    state.prev_loss_mean = loss_mean
    state.prev_step = step

    row = dict(
        step=float(step),
        loss_mean=loss_mean,
        loss_std=loss_std,
        dLdt_abs=dLdt_abs,
        I_mu=I_mu,
        I_sigma=I_sigma,
        I_total=I_total,
        churn_frac=churn_frac,
        bound=bound,
        efficiency=efficiency,
        directed_integral=state.directed_integral,
        churn_integral=state.churn_integral,
    )

    return state, row
