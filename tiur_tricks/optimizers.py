from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) wrapper.

    This is a lightweight SAM implementation designed for experimentation.

    Reference: https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, adaptive: bool = False, **base_kwargs):
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer_cls(params, **base_kwargs)
        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if self.adaptive:
                    g = g * torch.abs(p)
                norms.append(torch.norm(g, p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure")
        # First forward-backward pass
        loss = closure()
        self.first_step(zero_grad=True)
        # Second forward-backward pass
        closure()
        self.second_step(zero_grad=True)
        return loss


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def create_base_optimizer(cfg, params) -> torch.optim.Optimizer:
    opt = cfg.optimizer.lower()

    if opt == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

    if opt == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    if opt == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except Exception as e:
            raise ImportError(
                "Adafactor requires 'transformers'. In Colab: pip install transformers"
            ) from e
        return Adafactor(
            params,
            lr=cfg.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )

    if opt == "shampoo":
        try:
            from pytorch_optimizer import Shampoo
        except Exception as e:
            raise ImportError(
                "Shampoo requires 'pytorch-optimizer'. In Colab: pip install pytorch-optimizer"
            ) from e
        return Shampoo(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def create_optimizer(cfg, params) -> torch.optim.Optimizer:
    """Create an optimizer (optionally wrapped with SAM)."""
    if cfg.use_sam:
        # Wrap the base optimizer class.
        opt = cfg.optimizer.lower()
        if opt == "adamw":
            base_cls = torch.optim.AdamW
            base_kwargs = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        elif opt == "sgd":
            base_cls = torch.optim.SGD
            base_kwargs = dict(lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        else:
            raise ValueError("SAM wrapper currently supports base optimizers: adamw, sgd")
        return SAM(params, base_cls, rho=cfg.sam_rho, adaptive=False, **base_kwargs)

    return create_base_optimizer(cfg, params)
