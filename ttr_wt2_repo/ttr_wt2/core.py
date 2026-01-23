"""Core experiment code ported from the Colab notebook.

This is intentionally "not too clever": it keeps the same structure as the notebook,
while making it runnable as a normal Python package.

Key ideas:
- ManualAdamW lets us compute/apply/undo parameter updates so we can measure J for a
  candidate learning-rate (lr_sched) and then optionally shrink to hit a J_target.
- J is Jeffreys divergence between next-token distributions before/after an update.
- J-finder runs a short high-LR run while ramping J_target to choose a good J*.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

# -------------------------
# Defaults (mirrors notebook)
# -------------------------

# Training schedule
WARMUP_STEPS = 200
TOTAL_TRAIN_STEPS = 2000

# Eval
EVAL_EVERY = 200
EVAL_MAX_BATCHES = 50

# Data
BLOCK_SIZE = 128

# TTR
J_SAFETY = 0.05
J_BACKTRACK_MULT = 0.5
J_BACKTRACK_MAX_ITERS = 8

# AdamW
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1

# Misc
USE_BF16 = True
AMP_DTYPE = torch.bfloat16
CLIP_NORM = 1.0

# Model configs
MODEL_SIZES: Dict[str, Dict[str, Any]] = {
    "small":  {"n_layer": 6,  "n_head": 6,  "n_embd": 384,  "batch_size": 64},
    "medium": {"n_layer": 12, "n_head": 12, "n_embd": 768,  "batch_size": 32},
    "large":  {"n_layer": 24, "n_head": 16, "n_embd": 1024, "batch_size": 16},
}

DEFAULT_MODEL_SIZES = MODEL_SIZES

@dataclass
class ExperimentConfig:
    model_size: str = "small"
    mode: str = "ttr"
    seed: int = 0
    peak_lr: float = 0.03
    J_target: Optional[float] = None
    total_steps: int = 2000
    warmup_frac: float = 0.1
    block_size: int = 128
    train_batch_size: int = 64
    eval_batch_size: int = 64
    eval_every: int = 200
    eval_max_batches: int = 50
    use_bf16: bool = True
    clip_norm: float = 1.0
    weight_decay: float = 0.1
    keep_traces: bool = False
    device_str: str = "cuda"


# -------------------------
# Helpers
# -------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def enable_fast_math() -> None:
    # A100-friendly defaults
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# -------------------------
# Divergence (J)
# -------------------------

def jeffreys_div_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jeffreys divergence between distributions induced by logits.

    Returns a scalar tensor.
    """
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)
    kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(-1)
    return 0.5 * (kl_pq + kl_qp).mean()


# -------------------------
# Manual AdamW (for apply/undo)
# -------------------------

class ManualAdamW:
    """Minimal AdamW re-implementation that supports apply/undo.

    We keep optimizer state (m, v, t) and can apply an update for an arbitrary lr
    without calling torch.optim.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        betas: Tuple[float, float] = BETAS,
        eps: float = EPS,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.params = list(params)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}
        self.t = 0

    def _get_state(self, p: torch.nn.Parameter) -> Dict[str, Any]:
        sid = id(p)
        if sid not in self.state:
            self.state[sid] = {
                "exp_avg": torch.zeros_like(p.data),
                "exp_avg_sq": torch.zeros_like(p.data),
            }
        return self.state[sid]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def update_state(self) -> None:
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self._get_state(p)
            st["exp_avg"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            st["exp_avg_sq"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

    @torch.no_grad()
    def apply_update(self, lr: float) -> None:
        """Apply AdamW parameter update for the given lr."""
        b1, b2 = self.beta1, self.beta2
        t = self.t
        for p in self.params:
            if p.grad is None:
                continue
            st = self._get_state(p)
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]

            # Bias correction
            bias_correction1 = 1 - b1**t
            bias_correction2 = 1 - b2**t
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(self.eps)
            step = exp_avg / denom

            if self.weight_decay != 0:
                p.data.add_(p.data, alpha=-lr * self.weight_decay)
            p.data.add_(step, alpha=-step_size)

    @torch.no_grad()
    def undo_update(self, lr: float) -> None:
        """Undo an AdamW update previously applied with apply_update(lr).

        This is exact given fixed state and grads.
        """
        b1, b2 = self.beta1, self.beta2
        t = self.t
        for p in self.params:
            if p.grad is None:
                continue
            st = self._get_state(p)
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]

            bias_correction1 = 1 - b1**t
            bias_correction2 = 1 - b2**t
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(self.eps)
            step = exp_avg / denom

            # Undo in reverse order
            p.data.add_(step, alpha=step_size)
            if self.weight_decay != 0:
                p.data.add_(p.data, alpha=lr * self.weight_decay)


# -------------------------
# Data
# -------------------------

@dataclass
class DataBundle:
    tokenizer: Any
    train_loader: DataLoader
    val_loader: DataLoader


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # HuggingFace dataset yields dicts of lists/ints.
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.tensor([ex[k] for ex in batch], dtype=torch.long)
    return out


def make_loaders(
    train_batch_size: int,
    eval_batch_size: Optional[int] = None,
    block_size: int = BLOCK_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataBundle:
    """Load WikiText-2, tokenize, chunk, return DataLoaders."""

    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        return tok(examples["text"], truncation=False)

    tok_train = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    tok_val = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, Any]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_lm = tok_train.map(group_texts, batched=True)
    val_lm = tok_val.map(group_texts, batched=True)

    train_loader = DataLoader(
        train_lm,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_lm,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return DataBundle(tokenizer=tok, train_loader=train_loader, val_loader=val_loader)


# -------------------------
# Model
# -------------------------

def make_model(model_size: str, tokenizer: Any, seed: int, device: torch.device) -> GPT2LMHeadModel:
    cfg = MODEL_SIZES[model_size]
    seed_all(seed)
    gpt_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(gpt_cfg)
    model.to(device)
    model.train()
    # disable cache during training for memory
    model.config.use_cache = False
    return model


# -------------------------
# LR schedule
# -------------------------

def cosine_with_warmup(step: int, total_steps: int, peak_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    # cosine decay to ~0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * peak_lr * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def constant_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int = 0) -> float:
    # no warmup in default notebook "nowarmup"
    return float(peak_lr)


# -------------------------
# Probe positions
# -------------------------

def probe_positions() -> int:
    # Original notebook uses a single probe position; kept as a function for flexibility.
    return -2


# -------------------------
# Train steps
# -------------------------


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_step_baseline(
    model: nn.Module,
    opt: ManualAdamW,
    batch: Dict[str, torch.Tensor],
    lr: float,
    probe_pos: int,
    device: torch.device,
    compute_J: bool = True,
    use_bf16: bool = USE_BF16,
    amp_dtype: torch.dtype = AMP_DTYPE,
    clip_norm: Optional[float] = CLIP_NORM,
) -> Tuple[float, float]:
    """One training step for baseline (no J-targeting). Returns (loss, J_step)."""

    model.train()
    batch = _move_batch_to_device(batch, device)

    with torch.enable_grad():
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
                logits_before = out.logits.detach()
        else:
            out = model(**batch)
            loss = out.loss
            logits_before = out.logits.detach()

        if not loss.requires_grad:
            raise RuntimeError(
                "Loss has no grad_fn (grad tracking disabled). "
                "Check for an outer torch.no_grad()/torch.inference_mode()."
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

    # Update state then apply update
    opt.update_state()

    # Optional J measurement
    J_step = 0.0
    if compute_J:
        with torch.no_grad():
            opt.apply_update(lr)
            if use_bf16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits_after = model(**batch).logits
            else:
                logits_after = model(**batch).logits

            J_step = float(
                jeffreys_div_from_logits(logits_before[:, probe_pos], logits_after[:, probe_pos]).item()
            )
            opt.undo_update(lr)

    # Apply the real update
    with torch.no_grad():
        opt.apply_update(lr)

    return float(loss.detach().item()), float(J_step)


def train_step_ttr(
    model: nn.Module,
    opt: ManualAdamW,
    batch: Dict[str, torch.Tensor],
    lr_sched: float,
    J_target: float,
    probe_pos: int,
    device: torch.device,
    use_bf16: bool = USE_BF16,
    amp_dtype: torch.dtype = AMP_DTYPE,
    clip_norm: Optional[float] = CLIP_NORM,
    j_safety: float = J_SAFETY,
    backtrack_mult: float = J_BACKTRACK_MULT,
    backtrack_max_iters: int = J_BACKTRACK_MAX_ITERS,
) -> Tuple[float, float, float, float]:
    """One TTR step.

    Returns:
      loss, lr_eff, J_cand, J_applied
    """

    model.train()
    batch = _move_batch_to_device(batch, device)

    with torch.enable_grad():
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
                logits_before = out.logits.detach()
        else:
            out = model(**batch)
            loss = out.loss
            logits_before = out.logits.detach()

        if not loss.requires_grad:
            raise RuntimeError(
                "Loss has no grad_fn (grad tracking disabled). "
                "Check for an outer torch.no_grad()/torch.inference_mode()."
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

    opt.update_state()

    # Candidate update @ lr_sched to estimate J scaling
    with torch.no_grad():
        opt.apply_update(lr_sched)
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits_after_cand = model(**batch).logits
        else:
            logits_after_cand = model(**batch).logits

        J_cand_t = jeffreys_div_from_logits(logits_before[:, probe_pos], logits_after_cand[:, probe_pos])
        J_cand = float(J_cand_t.item())
        opt.undo_update(lr_sched)

    # Scale to hit J_target (approx assumes J ~ lr^2)
    if (J_target is not None) and (J_cand > J_target) and (J_cand > 0):
        lr_eff = lr_sched * math.sqrt(J_target / (J_cand + 1e-12))
    else:
        lr_eff = lr_sched

    # Backtracking: ensure applied J is within (1+j_safety)*J_target
    J_applied = float("nan")
    for _ in range(backtrack_max_iters + 1):
        with torch.no_grad():
            opt.apply_update(lr_eff)
            if use_bf16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits_after = model(**batch).logits
            else:
                logits_after = model(**batch).logits

            J_applied_t = jeffreys_div_from_logits(logits_before[:, probe_pos], logits_after[:, probe_pos])
            J_applied = float(J_applied_t.item())

            # If it's ok, keep update and break; otherwise undo and shrink lr_eff.
            if (not math.isnan(J_applied)) and (J_applied <= (1.0 + j_safety) * J_target):
                break

            opt.undo_update(lr_eff)
            lr_eff *= backtrack_mult

    # Ensure update remains applied; if we broke due to ok condition, update is still applied.
    # If we exhausted iters, update is applied at the final lr_eff (potentially tiny).

    return float(loss.detach().item()), float(lr_eff), float(J_cand), float(J_applied)


# -------------------------
# Eval
# -------------------------

@torch.no_grad()
def eval_lm(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool = USE_BF16,
    amp_dtype: torch.dtype = AMP_DTYPE,
    max_batches: int = EVAL_MAX_BATCHES,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
        else:
            out = model(**batch)
            loss = out.loss
        losses.append(float(loss.detach().item()))
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ppl = float(math.exp(mean_loss)) if math.isfinite(mean_loss) else float("nan")
    return {"val_loss": mean_loss, "val_ppl": ppl}


# -------------------------
# AutoTTR controller
# -------------------------

from collections import deque


class AutoTTRController:
    """Heuristic controller that adapts J_target based on stability of loss & J."""

    def __init__(
        self,
        J_init: float,
        J_min: float,
        J_max: float,
        window: int = 20,
        ema_beta: float = 0.98,
        step_frac: float = 0.02,
        spike_tol: float = 0.20,
    ):
        self.J = float(J_init)
        self.J_min = float(J_min)
        self.J_max = float(J_max)
        self.window = int(window)
        self.ema_beta = float(ema_beta)
        self.step_frac = float(step_frac)
        self.spike_tol = float(spike_tol)

        self.loss_ema: Optional[float] = None
        self.J_hist: deque = deque(maxlen=self.window)
        self.loss_hist: deque = deque(maxlen=self.window)

    def update(self, loss: float, J_applied: float) -> float:
        # EMA(loss)
        if self.loss_ema is None:
            self.loss_ema = float(loss)
        else:
            self.loss_ema = self.ema_beta * self.loss_ema + (1.0 - self.ema_beta) * float(loss)

        self.J_hist.append(float(J_applied))
        self.loss_hist.append(float(loss))

        if len(self.J_hist) < self.window:
            return self.J

        J_med = float(np.median(list(self.J_hist)))
        loss_med = float(np.median(list(self.loss_hist)))

        # Detect spikes
        J_spike = (float(J_applied) > (1.0 + self.spike_tol) * J_med) if math.isfinite(J_med) else False
        loss_spike = (float(loss) > (1.0 + self.spike_tol) * loss_med) if math.isfinite(loss_med) else False

        # If things look unstable, decrease J
        if J_spike or loss_spike:
            self.J *= (1.0 - self.step_frac)
        else:
            # Otherwise very gently increase J
            self.J *= (1.0 + 0.5 * self.step_frac)

        self.J = float(np.clip(self.J, self.J_min, self.J_max))
        return self.J


# -------------------------
# J-finder
# -------------------------

def _exp_ramp(J_min: float, J_max: float, frac: float) -> float:
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(J_min * ((J_max / J_min) ** frac))


def run_j_finder(
    model_size: str,
    seed: int,
    lr_high: float,
    J_min: float,
    J_max: float,
    data: DataBundle,
    device: torch.device,
    steps: int = 400,
    ema_beta: float = 0.98,
    increase_frac: float = 0.10,
    safety: float = 0.5,
    weight_decay: float = WEIGHT_DECAY,
    use_bf16: bool = USE_BF16,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Thermodynamic Range Test (TRT) / J-finder.

    IMPORTANT: must run with gradients enabled (do NOT wrap in torch.no_grad).

    Returns:
      J_star, info, traces
    """

    assert J_min > 0 and J_max > 0 and J_max > J_min
    seed_all(seed)

    train_iter = iter(data.train_loader)

    model = make_model(model_size, tokenizer=data.tokenizer, seed=seed, device=device)
    opt = ManualAdamW(model.parameters(), betas=BETAS, eps=EPS, weight_decay=weight_decay)
    probe_pos = probe_positions()

    t0 = time.time()

    Jt_trace: List[float] = []
    loss_trace: List[float] = []
    ema_trace: List[float] = []
    lr_eff_trace: List[float] = []
    J_app_trace: List[float] = []
    J_cand_trace: List[float] = []

    loss_ema: Optional[float] = None
    best_ema = float("inf")
    best_idx = 0

    for step in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train_loader)
            batch = next(train_iter)

        frac = step / max(1, steps - 1)
        Jt = _exp_ramp(J_min, J_max, frac)

        loss, lr_eff, J_cand, J_app = train_step_ttr(
            model=model,
            opt=opt,
            batch=batch,
            lr_sched=lr_high,
            J_target=Jt,
            probe_pos=probe_pos,
            device=device,
            use_bf16=use_bf16,
        )

        # EMA(loss)
        if loss_ema is None:
            loss_ema = float(loss)
        else:
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * float(loss)

        if loss_ema < best_ema:
            best_ema = loss_ema
            best_idx = step

        Jt_trace.append(float(Jt))
        loss_trace.append(float(loss))
        ema_trace.append(float(loss_ema))
        lr_eff_trace.append(float(lr_eff))
        J_app_trace.append(float(J_app))
        J_cand_trace.append(float(J_cand))

    elapsed = time.time() - t0

    # Choose stopping point: first time EMA rises > (1+increase_frac) above best_ema *after* the best point
    stop_idx: Optional[int] = None
    thresh = best_ema * (1.0 + increase_frac)
    for i in range(best_idx + 5, len(ema_trace)):
        if ema_trace[i] > thresh:
            stop_idx = i
            break
    if stop_idx is None:
        stop_idx = len(ema_trace) - 1

    # Conservative J*: take the J just before stop, and apply safety factor
    j_stop = Jt_trace[max(0, stop_idx - 1)]
    J_star = float(np.clip(j_stop * safety, J_min, J_max))

    info = dict(
        model_size=model_size,
        seed=seed,
        steps=steps,
        lr_high=lr_high,
        J_min=J_min,
        J_max=J_max,
        best_idx=int(best_idx),
        best_ema=float(best_ema),
        stop_idx=int(stop_idx),
        thresh=float(thresh),
        j_stop=float(j_stop),
        J_star=float(J_star),
        elapsed_s=float(elapsed),
    )

    traces = dict(
        Jt=Jt_trace,
        loss=loss_trace,
        loss_ema=ema_trace,
        lr_eff=lr_eff_trace,
        J_applied=J_app_trace,
        J_cand=J_cand_trace,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(J_star), info, traces


# -------------------------
# Experiment runner
# -------------------------


def run_one(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run a single training run from config."""
    device = torch.device(cfg.device_str) if cfg.device_str else get_device()
    
    data = make_loaders(
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        block_size=cfg.block_size,
    )
    
    warmup_steps = int(cfg.total_steps * cfg.warmup_frac)
    
    return _run_one_internal(
        model_size=cfg.model_size,
        mode=cfg.mode,
        seed=cfg.seed,
        peak_lr=cfg.peak_lr,
        data=data,
        device=device,
        J_target=cfg.J_target,
        keep_traces=cfg.keep_traces,
        total_steps=cfg.total_steps,
        warmup_steps=warmup_steps,
        eval_every=cfg.eval_every,
        eval_max_batches=cfg.eval_max_batches,
        use_bf16=cfg.use_bf16,
        clip_norm=cfg.clip_norm,
        weight_decay=cfg.weight_decay,
    )


def _run_one_internal(
    model_size: str,
    mode: str,
    seed: int,
    peak_lr: float,
    data: DataBundle,
    device: torch.device,
    J_target: Optional[float] = None,
    keep_traces: bool = False,
    total_steps: int = TOTAL_TRAIN_STEPS,
    warmup_steps: int = WARMUP_STEPS,
    eval_every: int = EVAL_EVERY,
    eval_max_batches: int = EVAL_MAX_BATCHES,
    use_bf16: bool = USE_BF16,
    clip_norm: float = CLIP_NORM,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict[str, Any]:
    """Internal implementation of the run logic."""

    assert model_size in MODEL_SIZES, f"Unknown model_size: {model_size}"
    assert mode in {"nowarmup", "warmup", "ttr", "autottr", "jfinder"}, f"Unknown mode: {mode}"

    enable_fast_math()
    seed_all(seed)

    # If requested, do J-finder first to get J_target.
    jfinder_info = None
    if mode == "jfinder":
        # Mirror notebook defaults
        J_star, info, _tr = run_j_finder(
            model_size=model_size,
            seed=seed,
            lr_high=peak_lr,
            J_min=1e-4,
            J_max=0.3,
            data=data,
            device=device,
            steps=400,
            ema_beta=0.98,
            increase_frac=0.10,
            safety=0.5,
            weight_decay=weight_decay,
            use_bf16=use_bf16,
        )
        J_target = J_star
        jfinder_info = info
        # Re-seed so the actual training run starts from the same init.
        seed_all(seed)

    model = make_model(model_size, tokenizer=data.tokenizer, seed=seed, device=device)
    opt = ManualAdamW(model.parameters(), betas=BETAS, eps=EPS, weight_decay=weight_decay)
    probe_pos = probe_positions()

    controller: Optional[AutoTTRController] = None
    if mode == "autottr":
        assert J_target is not None, "AutoTTR needs an initial J_target (pass --J_target or use --mode jfinder)."
        controller = AutoTTRController(J_init=J_target, J_min=1e-6, J_max=0.5)

    train_iter = iter(data.train_loader)

    traces: Dict[str, List[float]] = {
        "step": [],
        "train_loss": [],
        "lr_sched": [],
        "lr_eff": [],
        "J_target": [],
        "J_cand": [],
        "J_applied": [],
        "val_loss": [],
        "val_ppl": [],
    }

    best_val_loss = float("inf")
    best_val_ppl = float("inf")

    t0 = time.time()

    for step in range(total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train_loader)
            batch = next(train_iter)

        # LR schedule
        if mode == "warmup":
            lr_sched = cosine_with_warmup(step, total_steps, peak_lr, warmup_steps)
        else:
            lr_sched = constant_lr(step, total_steps, peak_lr)

        if mode in {"nowarmup", "warmup"}:
            loss, J_step = train_step_baseline(
                model=model,
                opt=opt,
                batch=batch,
                lr=lr_sched,
                probe_pos=probe_pos,
                device=device,
                compute_J=False,
                use_bf16=use_bf16,
                clip_norm=clip_norm,
            )
            lr_eff = lr_sched
            J_cand = float("nan")
            J_applied = float("nan")
            Jt = None

        else:
            assert J_target is not None, "TTR modes require a J_target (or use mode=jfinder)."

            Jt = float(J_target)
            loss, lr_eff, J_cand, J_applied = train_step_ttr(
                model=model,
                opt=opt,
                batch=batch,
                lr_sched=lr_sched,
                J_target=Jt,
                probe_pos=probe_pos,
                device=device,
                use_bf16=use_bf16,
                clip_norm=clip_norm,
            )

            if controller is not None:
                # Update J_target for next step
                J_target = float(controller.update(loss=loss, J_applied=J_applied))

        # Periodic eval
        val_loss = float("nan")
        val_ppl = float("nan")
        if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
            ev = eval_lm(
                model=model,
                val_loader=data.val_loader,
                device=device,
                max_batches=eval_max_batches,
                use_bf16=use_bf16,
            )
            val_loss = ev["val_loss"]
            val_ppl = ev["val_ppl"]
            best_val_loss = min(best_val_loss, val_loss)
            best_val_ppl = min(best_val_ppl, val_ppl)

        if keep_traces:
            traces["step"].append(step)
            traces["train_loss"].append(float(loss))
            traces["lr_sched"].append(float(lr_sched))
            traces["lr_eff"].append(float(lr_eff))
            traces["J_target"].append(float(Jt) if Jt is not None else float("nan"))
            traces["J_cand"].append(float(J_cand))
            traces["J_applied"].append(float(J_applied))
            traces["val_loss"].append(float(val_loss))
            traces["val_ppl"].append(float(val_ppl))

    elapsed = time.time() - t0

    summary: Dict[str, Any] = {
        "model_size": model_size,
        "mode": mode,
        "seed": int(seed),
        "peak_lr": float(peak_lr),
        "J_target_final": float(J_target) if J_target is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_val_ppl": float(best_val_ppl),
        "elapsed_s": float(elapsed),
    }
    if jfinder_info is not None:
        summary["jfinder"] = jfinder_info

    if keep_traces:
        summary["traces"] = traces

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)