# ttr-wt2

Refactor of the Colab notebook into a runnable repo for **TIUR / TTR / J-finder** experiments on WikiText-2.

## What this repo gives you

- A **single-run** CLI (`python -m ttr_wt2.run`) that reproduces the notebook logic.
- A **multi-GPU sweep** CLI (`python -m ttr_wt2.sweep`) that can saturate an 8×A100 node via *task parallelism* (one experiment per GPU worker).
- Minimal deps and scripts to install on a vanilla Ubuntu instance.

## Airline proof this
o keep your work running if SSH drops, use a named tmux session.

1. Start a new named session:
```
tmux new -s my_work
```
(Replace my_work with any name you like)

2. Detach from the session (leave it running in background): Press Ctrl+b, then release both and press d.

3. Reattach to the session (after reconnecting SSH):
```
tmux attach -t my_work
```
4. List all running sessions:
```
tmux ls
```

## Quickstart (Ubuntu + 8×A100)

1) Clone and bootstrap

```bash
git clone https://github.com/RespectableGlioma/tiur_tricks.git
cd ttr-wt2
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

2) Sanity check: single run on GPU 0

```bash
CUDA_VISIBLE_DEVICES=0 python -m ttr_wt2.run \
  --model_size small \
  --mode ttr \
  --seed 0 \
  --peak_lr 0.03 \
  --J_target 0.02 \
  --out runs/sanity.json
```

3) Sweep on 8 GPUs (round‑robin over GPUs)

```bash
python -m ttr_wt2.sweep \
  --config configs/sweep.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --out_dir runs
```

4) Aggregate results

```bash
python -m ttr_wt2.aggregate --runs_dir runs --out results.csv
```

## Notes on scaling strategy

For *large grids of short-ish runs*, **task parallelism** almost always wins:
- each GPU independently runs full experiments end‑to‑end
- you get near-linear throughput scaling up to 8 GPUs

If you later want **DDP/FSDP within a single run**, we can add that, but it tends to be lower leverage when your bottleneck is “N runs” rather than “one giant run”.

## Reproducing notebook defaults

The defaults in `ttr_wt2.core.ExperimentConfig` are taken from the notebook:
- 2,000 training steps
- eval every 200 steps on up to 50 val batches
- GPT‑2 style configs for `small|medium|large`

Edit `configs/sweep.yaml` to match your exact sweep grid.

## Common perf tips on p4d

- Set HF cache to local NVMe:
  ```bash
  export HF_DATASETS_CACHE=/local/hf_datasets
  export TRANSFORMERS_CACHE=/local/hf_transformers
  ```
- Increase DataLoader workers (p4d has lots of CPU): try `--num_workers 8` or `16`.
- Keep `eval_max_batches` small while exploring; do a final full eval pass only for the best runs.

