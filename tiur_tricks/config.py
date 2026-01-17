from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RunConfig:
    """Config for one experiment run.

    This harness trains an *ensemble* of N replicate models (different seeds), and uses
    that ensemble as an empirical proxy for the time-evolving parameter distribution
    \rho(\theta,t). TIUR diagnostics (loss fluctuation, time-Fisher, drift/churn
    decomposition, and efficiency) are estimated at checkpoints.

    Defaults are set to be Colab-friendly (quick runs), but you can scale them up.
    """

    # Identity
    name: str = "baseline"

    # Repro / ensemble
    num_replicates: int = 3
    seed_base: int = 0

    # Dataset / model
    dataset: str = "cifar10"  # {cifar10, fashionmnist, mnist}
    model: str = "resnet18"  # {resnet18, small_cnn}
    num_classes: int = 10

    # Data size shortcuts for fast iteration
    subset_train: Optional[int] = 10_000   # None = full train
    subset_eval: Optional[int] = 2_000     # None = full test

    # Training
    batch_size: int = 128
    epochs: int = 3
    max_steps: Optional[int] = None        # if set, overrides epochs

    lr: float = 3e-4
    weight_decay: float = 0.05

    # Loss
    label_smoothing: float = 0.0

    # Optimizer
    optimizer: str = "adamw"  # {adamw, sgd, adafactor, shampoo(optional)}
    momentum: float = 0.9      # for SGD
    betas: tuple[float, float] = (0.9, 0.999)  # for AdamW
    eps: float = 1e-8

    # Noise / "temperature" knobs
    grad_noise_std: float = 0.0  # add N(0, std) to each grad tensor

    # Regularizers / training tricks
    clip_grad_norm: Optional[float] = None
    ema_decay: Optional[float] = None  # e.g., 0.999
    swa_start_frac: Optional[float] = None  # e.g., 0.6 means start SWA after 60% of steps

    # SAM (Sharpness-Aware Minimization)
    use_sam: bool = False
    sam_rho: float = 0.05

    # Data sampling / curriculum
    sampler: str = "iid"  # {iid, easy2hard, loss_mixed}
    curriculum_warmup_steps: int = 200  # used to score difficulty (easy2hard)
    curriculum_steps: int = 1000        # steps over which the easy->hard fraction ramps

    # TIUR checkpointing
    checkpoint_every: int = 200  # in optimizer steps
    eval_batches: int = 20       # how many eval batches per checkpoint (None/full is slow)

    # TIUR Fisher estimator (diag Gaussian approx)
    fisher_eps: float = 1e-12

    # TIUR-guided controller
    use_tiur_controller: bool = False
    controller_target_churn: float = 0.35
    controller_band: float = 0.10
    controller_lr_up: float = 1.10
    controller_lr_down: float = 0.70
    controller_lr_min: float = 1e-5
    controller_lr_max: float = 3e-3

    # Misc
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True

    # Extra free-form overrides (kept for convenience)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        # dataclass stores tuples fine, but JSON users may prefer lists
        d["betas"] = list(self.betas)
        return d
