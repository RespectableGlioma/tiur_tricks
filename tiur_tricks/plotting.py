from __future__ import annotations

import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:200] if len(s) > 200 else s


def _maybe_save(fig, save_path: Optional[str]) -> None:
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)


def plot_single_run(df: pd.DataFrame, *, title: str = "run", save_dir: Optional[str] = None) -> None:
    """Plot key TIUR signals for one run (one config).

    If save_dir is provided, saves PNGs alongside displaying them.
    """
    df = df.sort_values("step")
    tag = _safe_name(title)

    # Loss curve
    fig = plt.figure()
    plt.plot(df["step"], df["loss_mean"], label="mean loss")
    plt.fill_between(
        df["step"],
        df["loss_mean"] - df["loss_std"],
        df["loss_mean"] + df["loss_std"],
        alpha=0.2,
        label="±1 std across ensemble",
    )
    plt.xlabel("step")
    plt.ylabel("eval loss")
    plt.title(f"{title} | eval loss")
    plt.legend()
    _maybe_save(fig, os.path.join(save_dir, f"{tag}_loss.png") if save_dir else None)
    plt.show()
    plt.close(fig)

    # Drift vs churn
    fig = plt.figure()
    plt.plot(df["step"], df["I_mu"], label="I_mu (drift)")
    plt.plot(df["step"], df["I_sigma"], label="I_sigma (churn)")
    plt.xlabel("step")
    plt.ylabel("time-Fisher (diag est.)")
    plt.title(f"{title} | TIUR time-Fisher decomposition")
    plt.legend()
    _maybe_save(fig, os.path.join(save_dir, f"{tag}_fisher.png") if save_dir else None)
    plt.show()
    plt.close(fig)

    # Efficiency
    fig = plt.figure()
    plt.plot(df["step"], df["efficiency"], label="eta")
    if df["efficiency"].notna().any():
        ymax = max(1.05, float(df["efficiency"].dropna().max()) * 1.05)
    else:
        ymax = 1.05
    plt.ylim(0, ymax)
    plt.xlabel("step")
    plt.ylabel("efficiency")
    plt.title(f"{title} | speed-limit efficiency")
    plt.legend()
    _maybe_save(fig, os.path.join(save_dir, f"{tag}_efficiency.png") if save_dir else None)
    plt.show()
    plt.close(fig)

    # Bound vs realized |dL/dt|
    fig = plt.figure()
    plt.plot(df["step"], df["dLdt_abs"], label="|d<L>/dt|")
    plt.plot(df["step"], df["bound"], label="DeltaL * sqrt(I_F)")
    plt.xlabel("step")
    plt.ylabel("rate / bound")
    plt.title(f"{title} | realized rate vs TIUR bound")
    plt.legend()
    _maybe_save(fig, os.path.join(save_dir, f"{tag}_bound.png") if save_dir else None)
    plt.show()
    plt.close(fig)


def plot_suite_overview(
    logs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    save_dir: Optional[str] = None,
) -> None:
    """Plot overview comparisons across runs.

    If save_dir is provided, saves PNGs and a text summary.
    """
    # Loss curves
    fig = plt.figure()
    for name, g in logs_df.groupby("name"):
        g = g.sort_values("step")
        plt.plot(g["step"], g["loss_mean"], label=name)
    plt.xlabel("step")
    plt.ylabel("eval loss")
    plt.title("Suite overview | eval loss")
    plt.legend(fontsize=8)
    _maybe_save(fig, os.path.join(save_dir, "suite_loss.png") if save_dir else None)
    plt.show()
    plt.close(fig)

    # Churn fraction curves (if available)
    fig = plt.figure()
    any_churn = False
    for name, g in logs_df.groupby("name"):
        g = g.sort_values("step")
        if g["churn_frac"].notna().any():
            any_churn = True
            plt.plot(g["step"], g["churn_frac"], label=name)
    plt.xlabel("step")
    plt.ylabel("I_sigma / (I_mu+I_sigma)")
    plt.title("Suite overview | churn fraction")
    if any_churn:
        plt.legend(fontsize=8)
    _maybe_save(fig, os.path.join(save_dir, "suite_churn_frac.png") if save_dir else None)
    plt.show()
    plt.close(fig)

    # Print + optionally save summary table
    display_cols = [
        "name",
        "final_loss",
        "final_loss_std",
        "final_efficiency",
        "final_churn_frac",
        "directed_integral",
        "churn_integral",
    ]
    text = "\n=== Summary (sorted by final_loss) ===\n"
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        text += summary_df[display_cols].to_string(index=False) + "\n"

    print(text)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "summary_table.txt"), "w", encoding="utf-8") as f:
            f.write(text)
