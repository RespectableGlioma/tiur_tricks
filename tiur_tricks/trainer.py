from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import RunConfig
from .data import build_data, make_eval_loader, make_train_loader, num_train_steps
from .models import make_model
from .optimizers import create_optimizer, get_lr, set_lr, SAM
from .regularizers import EMAHelper, SWAHelper
from .tiur_metrics import TIURState, update_tiur


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, device: str, max_batches: Optional[int] = None) -> float:
    model.eval()
    total = 0.0
    n = 0
    for b, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        total += float(loss.item())
        n += 1
        if max_batches is not None and (b + 1) >= max_batches:
            break
    return total / max(1, n)


@torch.no_grad()
def score_dataset_losses(model: nn.Module, loader, device: str, max_batches: Optional[int] = None) -> torch.Tensor:
    """Return per-sample loss vector aligned with local indices."""
    model.eval()
    # Determine dataset size from the underlying dataset wrapper (IndexedSubset)
    n_items = len(loader.dataset)
    losses = torch.zeros(n_items, dtype=torch.float32)
    for b, (x, y, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        batch_losses = nn.functional.cross_entropy(logits, y, reduction="none").detach().float().cpu()
        losses[idx] = batch_losses
        if max_batches is not None and (b + 1) >= max_batches:
            break
    return losses


class EnsembleTrainer:
    """Train an ensemble of replicate models and compute TIUR diagnostics."""

    def __init__(self, cfg: RunConfig, *, data_dir: str = "./data"):
        self.cfg = cfg
        self.device = cfg.device
        self.data_dir = data_dir

        # Build datasets once (shared across replicates)
        self.data = build_data(cfg, seed=cfg.seed_base, data_dir=data_dir)
        self.total_steps = num_train_steps(cfg, n_train=len(self.data.train))

        # Build eval loader once (shared)
        self.eval_loader = make_eval_loader(
            self.data.eval,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        # Prepare curriculum metadata if needed
        self.difficulty_order: Optional[List[int]] = None
        self.loss_weights: Optional[torch.Tensor] = None

        # Initialize ensemble replicates
        self.models: List[nn.Module] = []
        self.opts: List[torch.optim.Optimizer] = []
        self.ema: List[Optional[EMAHelper]] = []
        self.swa: List[Optional[SWAHelper]] = []
        self.train_loaders: List = []
        self.train_iters: List = []

        self._init_ensemble()

        # TIUR state
        self.tiur_state = TIURState()
        self.logs: List[Dict[str, float]] = []

        # Controller auto-calibration (if cfg.controller_target_churn is None)
        self._controller_target: Optional[float] = None

    def _init_ensemble(self):
        cfg = self.cfg

        # If using curriculum variants, compute ordering/weights once
        if cfg.sampler in {"easy2hard", "loss_mixed"}:
            self._prepare_sampling_metadata()

        for r in range(cfg.num_replicates):
            seed = cfg.seed_base + 10_000 * r
            seed_all(seed)

            model = make_model(cfg.model, cfg.dataset, cfg.num_classes).to(cfg.device)
            opt = create_optimizer(cfg, model.parameters())

            ema_helper = EMAHelper(model, cfg.ema_decay) if cfg.ema_decay is not None else None
            swa_helper = SWAHelper(model) if cfg.swa_start_frac is not None else None

            self.models.append(model)
            self.opts.append(opt)
            self.ema.append(ema_helper)
            self.swa.append(swa_helper)

            loader = self._make_train_loader_for_step(step=0, seed=seed)
            self.train_loaders.append(loader)
            self.train_iters.append(iter(loader))

    def _make_train_loader_for_step(self, step: int, seed: int):
        cfg = self.cfg
        if cfg.sampler == "iid":
            return make_train_loader(
                self.data.train,
                batch_size=cfg.batch_size,
                seed=seed,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )

        if cfg.sampler == "easy2hard":
            assert self.difficulty_order is not None
            n = len(self.difficulty_order)
            frac = min(1.0, max(0.05, step / max(1, cfg.curriculum_steps)))
            k = max(1, int(frac * n))
            allowed = self.difficulty_order[:k]
            return make_train_loader(
                self.data.train,
                batch_size=cfg.batch_size,
                seed=seed,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                indices=allowed,
            )

        if cfg.sampler == "loss_mixed":
            # Weighted sampler over the full train set
            if self.loss_weights is None:
                self.loss_weights = torch.ones(len(self.data.train), dtype=torch.float32)
            return make_train_loader(
                self.data.train,
                batch_size=cfg.batch_size,
                seed=seed,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                weights=self.loss_weights,
            )

        raise ValueError(f"Unknown sampler: {cfg.sampler}")

    def _prepare_sampling_metadata(self):
        cfg = self.cfg

        # Warmup on a single model to get a meaningful difficulty ranking
        seed_all(cfg.seed_base + 999)
        warm_model = make_model(cfg.model, cfg.dataset, cfg.num_classes).to(cfg.device)
        warm_opt = torch.optim.AdamW(warm_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        warm_loader = make_train_loader(
            self.data.train,
            batch_size=cfg.batch_size,
            seed=cfg.seed_base + 999,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        warm_iter = iter(warm_loader)

        warm_model.train()
        for _ in range(cfg.curriculum_warmup_steps):
            try:
                x, y, _ = next(warm_iter)
            except StopIteration:
                warm_iter = iter(warm_loader)
                x, y, _ = next(warm_iter)
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            warm_opt.zero_grad(set_to_none=True)
            logits = warm_model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            warm_opt.step()

        # Score per-sample difficulty by loss
        score_loader = make_eval_loader(
            self.data.train,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        losses = score_dataset_losses(warm_model, score_loader, cfg.device, max_batches=None)

        # easiest first
        order = torch.argsort(losses).tolist()
        self.difficulty_order = order

        # initial weights for loss_mixed
        self.loss_weights = torch.ones(len(self.data.train), dtype=torch.float32)

    def _maybe_refresh_sampler(self, step: int):
        cfg = self.cfg
        # Refresh at checkpoints for simplicity (cheap and deterministic)
        if step % cfg.checkpoint_every != 0:
            return

        if cfg.sampler == "easy2hard":
            for r in range(cfg.num_replicates):
                seed = cfg.seed_base + 10_000 * r
                self.train_loaders[r] = self._make_train_loader_for_step(step=step, seed=seed)
                self.train_iters[r] = iter(self.train_loaders[r])

        if cfg.sampler == "loss_mixed":
            # Update weights using replicate 0 model
            model0 = self.models[0]
            score_loader = make_eval_loader(
                self.data.train,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
            )
            # If EMA/SWA is enabled, use the base weights for scoring (simple choice)
            losses = score_dataset_losses(model0, score_loader, cfg.device, max_batches=None)
            losses = losses.clamp_min(0.0)

            # mixture of uniform + loss-proportional
            alpha = float(cfg.extra.get("loss_mixed_alpha", 0.5))
            # normalize losses into a stable range
            l = losses
            l = (l - l.min()) / (l.max() - l.min() + 1e-12)
            w = (1.0 - alpha) * torch.ones_like(l) + alpha * (l + 1e-3)
            self.loss_weights = w

            for r in range(cfg.num_replicates):
                seed = cfg.seed_base + 10_000 * r
                self.train_loaders[r] = self._make_train_loader_for_step(step=step, seed=seed)
                self.train_iters[r] = iter(self.train_loaders[r])

    def _criterion(self):
        if self.cfg.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        return nn.CrossEntropyLoss()

    def _apply_eval_weights_if_needed(self, r: int):
        # Apply EMA or SWA weights for evaluation (priority: EMA then SWA)
        if self.ema[r] is not None:
            self.ema[r].apply(self.models[r])
            return "ema"
        if self.swa[r] is not None:
            self.swa[r].apply(self.models[r])
            return "swa"
        return "base"

    def _restore_eval_weights_if_needed(self, r: int, which: str):
        if which == "ema" and self.ema[r] is not None:
            self.ema[r].restore(self.models[r])
        if which == "swa" and self.swa[r] is not None:
            self.swa[r].restore(self.models[r])

    def _controller_update(self, step: int, row: Dict[str, float]):
        cfg = self.cfg
        if not cfg.use_tiur_controller:
            return
        if math.isnan(row.get("churn_frac", float("nan"))):
            return

        churn = float(row["churn_frac"])
        eff = float(row.get("efficiency", 0.0))

        # Target churn: if None, lock to the first observed churn value for this run.
        if cfg.controller_target_churn is None:
            if self._controller_target is None:
                self._controller_target = churn
            target = float(self._controller_target)
        else:
            target = float(cfg.controller_target_churn)

        band = cfg.controller_band
        low = target - band
        high = target + band

        # Use the LR of replicate 0 as reference, then set all replicates.
        lr = get_lr(self.opts[0])

        if churn > high:
            lr = max(cfg.controller_lr_min, lr * cfg.controller_lr_down)
        elif churn < low and eff < 0.9:  # only speed up if not already highly efficient
            lr = min(cfg.controller_lr_max, lr * cfg.controller_lr_up)

        for opt in self.opts:
            set_lr(opt, lr)


    def train(self, *, log_csv_path: Optional[str] = None) -> List[Dict[str, float]]:
        cfg = self.cfg
        device = cfg.device
        criterion = self._criterion().to(device)

        live_f = None
        live_writer = None
        wrote_header = False
        if log_csv_path is not None:
            os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
            # Append mode: if a previous partial file exists, keep it.
            live_f = open(log_csv_path, "a", newline="", encoding="utf-8")

        swa_start_step = None
        if cfg.swa_start_frac is not None:
            swa_start_step = int(cfg.swa_start_frac * self.total_steps)

        pbar = tqdm(range(self.total_steps), desc=f"{cfg.name}", leave=False)

        for step in pbar:
            self._maybe_refresh_sampler(step)

            # One synchronized step across ensemble
            for r in range(cfg.num_replicates):
                model = self.models[r]
                opt = self.opts[r]

                model.train()

                try:
                    x, y, _ = next(self.train_iters[r])
                except StopIteration:
                    self.train_iters[r] = iter(self.train_loaders[r])
                    x, y, _ = next(self.train_iters[r])

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if isinstance(opt, SAM):
                    def closure():
                        opt.zero_grad(set_to_none=True)
                        logits = model(x)
                        loss = criterion(logits, y)
                        loss.backward()
                        # grad noise / clipping inside closure (applies to both SAM steps)
                        if cfg.grad_noise_std > 0:
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.add_(torch.randn_like(p.grad) * cfg.grad_noise_std)
                        if cfg.clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                        return loss

                    loss = opt.step(closure)

                else:
                    opt.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()

                    if cfg.grad_noise_std > 0:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.add_(torch.randn_like(p.grad) * cfg.grad_noise_std)

                    if cfg.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                    opt.step()

                # Update EMA
                if self.ema[r] is not None:
                    self.ema[r].update(model)

                # Update SWA
                if swa_start_step is not None and step >= swa_start_step and self.swa[r] is not None:
                    self.swa[r].update(model)

            # Checkpoint: evaluate + TIUR update
            if (step + 1) % cfg.checkpoint_every == 0 or (step + 1) == self.total_steps:
                eval_losses: List[float] = []
                for r in range(cfg.num_replicates):
                    which = self._apply_eval_weights_if_needed(r)
                    l = evaluate_loss(self.models[r], self.eval_loader, device, max_batches=cfg.eval_batches)
                    self._restore_eval_weights_if_needed(r, which)
                    eval_losses.append(l)

                # Collect params dicts
                params_by_model = [
                    {name: p for name, p in m.named_parameters() if p.requires_grad}
                    for m in self.models
                ]

                self.tiur_state, row = update_tiur(
                    self.tiur_state,
                    step=step + 1,
                    losses=eval_losses,
                    params_by_model=params_by_model,
                    eps=cfg.fisher_eps,
                )

                row.update({"name": cfg.name, "lr": get_lr(self.opts[0])})
                self.logs.append(row)

                # Persist checkpoint row immediately (useful on Colab / preemptible runtimes)
                if live_f is not None:
                    if (not wrote_header) and (live_writer is None):
                        live_writer = csv.DictWriter(live_f, fieldnames=list(row.keys()))
                        # Write header only if file is empty
                        if live_f.tell() == 0:
                            live_writer.writeheader()
                        wrote_header = True
                    assert live_writer is not None
                    live_writer.writerow(row)
                    live_f.flush()

                # Controller uses last interval's churn/efficiency to set LR for next interval
                self._controller_update(step=step + 1, row=row)

                pbar.set_postfix({
                    "loss": f"{row['loss_mean']:.3f}",
                    "ΔL": f"{row['loss_std']:.3f}",
                    "churn": f"{row.get('churn_frac', float('nan')):.2f}",
                    "η": f"{row.get('efficiency', float('nan')):.2f}",
                })

        if live_f is not None:
            live_f.close()

        return self.logs