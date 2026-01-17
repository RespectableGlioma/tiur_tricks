from __future__ import annotations

from typing import Dict, Optional

import torch


class EMAHelper:
    """Exponential moving average of parameters.

    Stores EMA weights on CPU to keep GPU memory low.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _init_shadow(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().cpu().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.shadow[name].to(p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[name])
        self.backup = {}


class SWAHelper:
    """Stochastic Weight Averaging style running average.

    This is a minimal SWA implementation. It does NOT update BN statistics.
    For quick TIUR diagnostics it's often enough.
    """

    def __init__(self, model: torch.nn.Module):
        self.n: int = 0
        self.avg: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.avg[name] = p.detach().float().cpu().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.n += 1
        alpha = 1.0 / self.n
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.avg[name].add_(p.detach().float().cpu() - self.avg[name], alpha=alpha)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.avg[name].to(p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[name])
        self.backup = {}
