# TIUR Training Tricks - Colab Quickstart

This is a small, self-contained harness to test common training interventions ("training tricks")
using TIUR-style diagnostics:

- mean eval loss across an ensemble of replicate runs
- ensemble loss fluctuations (std across replicates)
- a diagonal-Gaussian *time-Fisher* estimate from the ensemble
- drift vs churn decomposition: `I_mu` vs `I_sigma`
- speed-limit efficiency: `eta = |d<L>/dt| / (DeltaL * sqrt(I_F))`

## Easiest option: run the included notebook

This repo ships with a **top-level Colab notebook**:

- `TIUR_Set1_Quickstart.ipynb`

Open it in Colab, flip Runtime → **GPU**, and run all cells. The notebook will:

1) mount Google Drive
2) install dependencies
3) run the Experiment Set 1 suite
4) **save all logs + plots to Drive** (so you don't lose results if the runtime dies)

## Minimal Colab steps (manual)

1) **Upload** the zip (or this folder) into Colab.

2) In a Colab cell:

```python
!pip install -q tqdm pandas matplotlib

# NOTE: Colab already includes torch + torchvision.
# Avoid reinstalling torchvision unless you also install a matching torch build.

import sys
sys.path.insert(0, "/content/tiur_tricks_colab")  # adjust if you unzip elsewhere

from tiur_tricks import run_set1_quick

logs_df, summary_df = run_set1_quick(
    device="cuda",
    out_dir="/content/drive/MyDrive/tiur_tricks_results/run1",
    fast=True,   # True = small grid; False = bigger grid (adds optional deps)
)
```

3) Inspect outputs:

- `<out_dir>/summary.csv`
- `<out_dir>/<run_name>/logs.csv`
- `<out_dir>/<run_name>/*.png` (plots)

Additionally, if `persist_checkpoints=True` (default in `run_set1_quick`), each run writes
`logs_live.csv` as it trains so partial progress is preserved on preemptible runtimes.

## Notes / Extensions

- For **Shampoo**: `pip install pytorch-optimizer`
- For **Adafactor**: `pip install transformers`

- For more stable TIUR estimates, increase `num_replicates` from 3 to 5–8.
- For faster debugging, reduce `subset_train` and `subset_eval`.


## Interpreting the TIUR plots

- `I_mu` is the *directed drift* term (movement of the ensemble mean).
- `I_sigma` is the *churn* term (shape/covariance reshaping).
- `churn_frac = I_sigma / (I_mu + I_sigma)` is a handy one-number diagnostic.
- `efficiency` close to 1 means you're close to the estimated speed limit; near 0 means you're moving in parameter space without improving loss much.

**Note:** the time-Fisher estimate here uses a *diagonal-Gaussian* approximation from a small ensemble, so the inequality may not be perfectly tight and `efficiency` can occasionally exceed 1 due to estimator noise.
