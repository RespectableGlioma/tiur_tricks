from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import RunConfig
from .plotting import plot_suite_overview, plot_single_run
from .trainer import EnsembleTrainer
from .data import num_train_steps


def _approx_train_size(cfg: RunConfig) -> int:
    """Approximate training set size without instantiating datasets.

    Used to tune curriculum/EMA/controller defaults for very short Colab runs.
    """
    if cfg.subset_train is not None:
        return int(cfg.subset_train)

    ds = cfg.dataset.lower()
    if ds == "cifar10":
        return 50_000
    if ds in {"mnist", "fashionmnist", "fashion-mnist"}:
        return 60_000
    # Fallback: a reasonable default
    return 50_000


def make_experiment_suite_set1(base: Optional[RunConfig] = None, *, fast: bool = True) -> List[RunConfig]:
    """Create a suite of configs that cover Experiment Set 1.

    Notes:
      - In fast/Colab mode, we avoid redundant "baseline" duplicates to save time.
      - Some tricks (EMA, easy2hard curriculum) are sensitive to *total training steps*.
        We automatically adapt those defaults so quick runs don't look artificially bad.
    """
    if base is None:
        base = RunConfig()

    # Use frequent checkpoints for meaningful TIUR curves
    base = replace(base, checkpoint_every=50, eval_batches=20)

    # Approximate total steps (needed to pick sensible curriculum/EMA settings)
    n_train = _approx_train_size(base)
    total_steps = num_train_steps(base, n_train=n_train)

    suite: List[RunConfig] = []

    # Baseline (single)
    baseline = replace(
        base,
        name="baseline",
        optimizer="adamw",
        sampler="iid",
        grad_noise_std=0.0,
        ema_decay=None,
        swa_start_frac=None,
        clip_grad_norm=None,
        use_sam=False,
        use_tiur_controller=False,
    )
    suite.append(baseline)

    # 1) Optimizer comparison
    suite.append(replace(baseline, name="opt_sgd", optimizer="sgd", lr=0.05, weight_decay=1e-4))

    if not fast:
        suite.append(replace(baseline, name="opt_adafactor", optimizer="adafactor"))
        suite.append(replace(baseline, name="opt_shampoo", optimizer="shampoo"))

    # 2) Noise / temperature sweep (gradient noise)
    if not fast:
        suite.append(replace(baseline, name="noise_0", grad_noise_std=0.0))
    suite.append(replace(baseline, name="noise_1e-3", grad_noise_std=1e-3))
    suite.append(replace(baseline, name="noise_1e-2", grad_noise_std=1e-2))
    if not fast:
        suite.append(replace(baseline, name="noise_5e-2", grad_noise_std=5e-2))

    # 3) TIUR-guided controller vs baseline
    # In short runs, the absolute scale of churn_frac depends on the estimator.
    # If controller_target_churn=None (default), we auto-calibrate to the first observed churn.
    suite.append(replace(baseline, name="ctrl_on", use_tiur_controller=True))

    # 4) Curriculum / sampling
    # Important: For Colab-quick runs, the default curriculum_steps=1000 can exceed total_steps.
    # That would mean we *never* expose the model to most of the dataset.
    # Fix: make curriculum_steps track the run length so we reach the full set by the end.
    suite.append(replace(baseline, name="sampler_easy2hard", sampler="easy2hard", curriculum_steps=max(1, total_steps)))
    suite.append(replace(baseline, name="sampler_lossmixed", sampler="loss_mixed"))

    # 5) Regularizers
    # EMA: if total_steps is very small, a huge decay (0.999) will keep EMA near init.
    # Use a shorter-timescale EMA by default for quick runs, while still letting you override.
    ema_decay = 0.99 if total_steps < 500 else 0.999
    suite.append(replace(baseline, name="reg_ema", ema_decay=ema_decay))
    suite.append(replace(baseline, name="reg_clip", clip_grad_norm=1.0))

    if not fast:
        suite.append(replace(baseline, name="reg_swa", swa_start_frac=0.6))
        suite.append(replace(baseline, name="reg_sam", use_sam=True, optimizer="adamw"))

    return suite
def run_experiment_suite(
    suite: List[RunConfig],
    *,
    out_dir: str = "./tiur_out",
    show_plots: bool = True,
    data_dir: str = "./data",
    save_plots: bool = True,
    persist_checkpoints: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a suite of experiments.

    Returns:
      - logs_df: long-form per-checkpoint logs (one row per checkpoint per run)
      - summary_df: one row per run with aggregated metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    # Record a small manifest for provenance
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_runs": len(suite),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    all_rows: List[Dict] = []
    summaries: List[Dict] = []

    for cfg in suite:
        run_dir = os.path.join(out_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)

        # Save config for this run
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

        trainer = EnsembleTrainer(cfg, data_dir=data_dir)
        # If the runtime dies, we still have partial logs on disk.
        log_csv_path = os.path.join(run_dir, "logs_live.csv") if persist_checkpoints else None
        logs = trainer.train(log_csv_path=log_csv_path)

        for row in logs:
            all_rows.append(row)

        # Summary metrics
        last = logs[-1]
        # Approximate integrals are already tracked by TIURState
        summary = dict(
            name=cfg.name,
            optimizer=cfg.optimizer,
            sampler=cfg.sampler,
            grad_noise_std=cfg.grad_noise_std,
            ema_decay=cfg.ema_decay or 0.0,
            swa_start_frac=cfg.swa_start_frac or 0.0,
            clip_grad_norm=cfg.clip_grad_norm or 0.0,
            use_sam=int(cfg.use_sam),
            use_controller=int(cfg.use_tiur_controller),
            final_loss=float(last["loss_mean"]),
            final_loss_std=float(last["loss_std"]),
            final_efficiency=float(last.get("efficiency", float("nan"))),
            directed_integral=float(last.get("directed_integral", float("nan"))),
            churn_integral=float(last.get("churn_integral", float("nan"))),
            final_churn_frac=float(last.get("churn_frac", float("nan"))),
        )
        summaries.append(summary)

        # Save per-run CSV for easy inspection
        pd.DataFrame(logs).to_csv(os.path.join(run_dir, "logs.csv"), index=False)
        # Also save a flat copy at the top-level for convenience
        pd.DataFrame(logs).to_csv(os.path.join(out_dir, f"{cfg.name}_logs.csv"), index=False)

        if show_plots:
            plot_single_run(
                pd.DataFrame(logs),
                title=cfg.name,
                save_dir=run_dir if save_plots else None,
            )

    logs_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries).sort_values("final_loss")

    logs_df.to_csv(os.path.join(out_dir, "all_logs.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    if show_plots:
        plot_suite_overview(logs_df, summary_df, save_dir=out_dir if save_plots else None)

    return logs_df, summary_df


def run_set1_quick(
    *,
    device: str = "cuda",
    out_dir: str = "./tiur_out",
    data_dir: str = "./data",
    fast: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-liner to run a Colab-friendly Experiment Set 1 suite."""
    base = RunConfig(device=device)
    suite = make_experiment_suite_set1(base, fast=fast)
    return run_experiment_suite(
        suite,
        out_dir=out_dir,
        show_plots=True,
        data_dir=data_dir,
        save_plots=True,
        persist_checkpoints=True,
    )
