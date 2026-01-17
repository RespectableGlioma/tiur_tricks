"""TIUR Training Tricks (Colab-friendly)

A small experimental harness for testing common "training tricks" using TIUR-style
metrics:
  - loss mean and ensemble loss fluctuations
  - time-Fisher estimate (diag Gaussian ensemble approximation)
  - drift vs churn decomposition (I_F^mu vs I_F^Sigma)
  - speed-limit efficiency eta = |d<L>/dt| / (DeltaL * sqrt(I_F))

The package is designed to run quickly on a single GPU (e.g. Colab A100) while still
supporting principled comparisons across training interventions.
"""

from .config import RunConfig
from .experiments import (
    make_experiment_suite_set1,
    run_experiment_suite,
    run_set1_quick,
)

__all__ = [
    "RunConfig",
    "make_experiment_suite_set1",
    "run_experiment_suite",
    "run_set1_quick",
]
