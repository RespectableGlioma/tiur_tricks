from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_single_run(df: pd.DataFrame, *, title: str = "run") -> None:
    """Plot key TIUR signals for one run (one config)."""
    df = df.sort_values("step")

    # Loss curve
    plt.figure()
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
    plt.show()

    # Drift vs churn
    plt.figure()
    plt.plot(df["step"], df["I_mu"], label="I_mu (drift)")
    plt.plot(df["step"], df["I_sigma"], label="I_sigma (churn)")
    plt.xlabel("step")
    plt.ylabel("time-Fisher (diag est.)")
    plt.title(f"{title} | TIUR time-Fisher decomposition")
    plt.legend()
    plt.show()

    # Efficiency
    plt.figure()
    plt.plot(df["step"], df["efficiency"], label="eta")
    plt.ylim(0, max(1.05, float(df["efficiency"].dropna().max()) * 1.05) if df["efficiency"].notna().any() else 1.05)
    plt.xlabel("step")
    plt.ylabel("efficiency")
    plt.title(f"{title} | speed-limit efficiency")
    plt.legend()
    plt.show()

    # Bound vs realized |dL/dt|
    plt.figure()
    plt.plot(df["step"], df["dLdt_abs"], label="|d<L>/dt|")
    plt.plot(df["step"], df["bound"], label="DeltaL * sqrt(I_F)")
    plt.xlabel("step")
    plt.ylabel("rate / bound")
    plt.title(f"{title} | realized rate vs TIUR bound")
    plt.legend()
    plt.show()


def plot_suite_overview(logs_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Plot overview comparisons across runs."""
    # Loss curves
    plt.figure()
    for name, g in logs_df.groupby("name"):
        g = g.sort_values("step")
        plt.plot(g["step"], g["loss_mean"], label=name)
    plt.xlabel("step")
    plt.ylabel("eval loss")
    plt.title("Suite overview | eval loss")
    plt.legend(fontsize=8)
    plt.show()

    # Churn fraction curves (if available)
    plt.figure()
    for name, g in logs_df.groupby("name"):
        g = g.sort_values("step")
        if g["churn_frac"].notna().any():
            plt.plot(g["step"], g["churn_frac"], label=name)
    plt.xlabel("step")
    plt.ylabel("I_sigma / (I_mu+I_sigma)")
    plt.title("Suite overview | churn fraction")
    plt.legend(fontsize=8)
    plt.show()

    # Print summary table
    display_cols = [
        "name",
        "final_loss",
        "final_loss_std",
        "final_efficiency",
        "final_churn_frac",
        "directed_integral",
        "churn_integral",
    ]
    print("\n=== Summary (sorted by final_loss) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary_df[display_cols].to_string(index=False))
