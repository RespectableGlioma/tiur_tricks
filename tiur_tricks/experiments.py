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


def make_experiment_suite_set1(base: Optional[RunConfig] = None, *, fast: bool = True) -> List[RunConfig]:
    """Create a suite of configs that cover Experiment Set 1.

    If fast=True, keeps the grid small and Colab-friendly.
    """
    if base is None:
        base = RunConfig()

    # Use frequent checkpoints for meaningful TIUR curves
    base = replace(base, checkpoint_every=50, eval_batches=20)

    suite: List[RunConfig] = []

    # 1) Optimizer comparison
    suite.append(replace(base, name="opt_adamw", optimizer="adamw"))
    suite.append(replace(base, name="opt_sgd", optimizer="sgd", lr=0.05, weight_decay=1e-4))

    if not fast:
        suite.append(replace(base, name="opt_adafactor", optimizer="adafactor"))
        suite.append(replace(base, name="opt_shampoo", optimizer="shampoo"))

    # 2) Noise / temperature sweep (gradient noise)
    suite.append(replace(base, name="noise_0", grad_noise_std=0.0))
    suite.append(replace(base, name="noise_1e-3", grad_noise_std=1e-3))
    suite.append(replace(base, name="noise_1e-2", grad_noise_std=1e-2))

    if not fast:
        suite.append(replace(base, name="noise_5e-2", grad_noise_std=5e-2))

    # 3) TIUR-guided controller vs baseline
    suite.append(replace(base, name="ctrl_off", use_tiur_controller=False))
    suite.append(replace(base, name="ctrl_on", use_tiur_controller=True))

    # 4) Curriculum / sampling
    suite.append(replace(base, name="sampler_iid", sampler="iid"))
    suite.append(replace(base, name="sampler_easy2hard", sampler="easy2hard"))
    suite.append(replace(base, name="sampler_lossmixed", sampler="loss_mixed"))

    # 5) Regularizers
    suite.append(replace(base, name="reg_baseline", ema_decay=None, swa_start_frac=None, clip_grad_norm=None, use_sam=False))
    suite.append(replace(base, name="reg_ema", ema_decay=0.999))
    suite.append(replace(base, name="reg_clip", clip_grad_norm=1.0))

    if not fast:
        suite.append(replace(base, name="reg_swa", swa_start_frac=0.6))
        suite.append(replace(base, name="reg_sam", use_sam=True, optimizer="adamw"))

    # Deduplicate configs by name (easy if user edits above)
    seen = set()
    unique: List[RunConfig] = []
    for c in suite:
        if c.name in seen:
            continue
        seen.add(c.name)
        unique.append(c)

    return unique


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
