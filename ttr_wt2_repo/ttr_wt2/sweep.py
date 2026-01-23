"""Multi-run sweep runner.

This is designed for a *single node* with multiple GPUs (e.g. AWS p4d.24xlarge with 8xA100).

It does *task parallelism*:
- each worker process is pinned to exactly 1 GPU
- each worker runs a list of experiments sequentially on that GPU

This tends to scale better than DDP for large grids of short-ish runs.

Example:
  python -m ttr_wt2.sweep --config configs/sweep.yaml --gpus 0,1,2,3,4,5,6,7

The output directory will contain one JSON summary per run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import queue
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm.auto import tqdm

from .core import ExperimentConfig, run_one


def _parse_gpus(s: str) -> List[int]:
    if s.strip() in {"", "auto"}:
        # Defer actual detection to torch inside workers.
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _task_name(t: Dict[str, Any]) -> str:
    parts = [
        t["model_size"],
        t["mode"],
        f"seed{t['seed']}",
        f"lr{t['peak_lr']}",
    ]
    return "__".join(parts)


def _worker(worker_id: int, gpu_id: int | None, tasks: List[Dict[str, Any]], base_cfg: Dict[str, Any], out_dir: str, progress_queue: mp.Queue) -> None:
    # IMPORTANT: set CUDA_VISIBLE_DEVICES before importing torch in this process.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for t in tasks:
        cfg = ExperimentConfig(**base_cfg)
        cfg.model_size = t["model_size"]
        cfg.mode = t["mode"]
        cfg.seed = int(t["seed"])
        cfg.peak_lr = float(t["peak_lr"])
        cfg.J_target = t.get("J_target", None)
        cfg.use_tqdm = False  # Disable per-run progress bar to avoid interleaving

        try:
            res = run_one(cfg)
            payload = {"task": t, "config": asdict(cfg), "result": res}
            out_file = out_path / f"{_task_name(t)}.json"
            out_file.write_text(json.dumps(payload, indent=2))
            progress_queue.put(("success", _task_name(t)))
        except Exception as e:
            out_file = out_path / f"{_task_name(t)}.FAILED.txt"
            out_file.write_text(str(e))
            progress_queue.put(("failed", _task_name(t), str(e)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML sweep file")
    ap.add_argument("--gpus", type=str, default="auto", help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
    ap.add_argument("--out_dir", type=str, default="runs", help="Directory to write one JSON per run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Handle flat config vs nested config
    if "sweep" not in cfg and "base" not in cfg:
        # Assume flat structure
        sweep_keys = {"model_sizes", "modes", "seeds", "peak_lrs"}
        sweep = {k: cfg[k] for k in sweep_keys if k in cfg}
        base_cfg = {k: v for k, v in cfg.items() if k not in sweep_keys}
    else:
        base_cfg = cfg.get("base", {})
        sweep = cfg.get("sweep", {})

    # Remap keys to match ExperimentConfig
    remaps = {
        "steps": "total_steps",
        "batch_size": "train_batch_size",
        "save_traces": "keep_traces",
    }
    for k, v in remaps.items():
        if k in base_cfg:
            base_cfg[v] = base_cfg.pop(k)

    # Filter base_cfg to only valid fields
    valid_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    base_cfg = {k: v for k, v in base_cfg.items() if k in valid_fields}

    model_sizes = sweep.get("model_sizes", ["small"])
    modes = sweep.get("modes", ["nowarmup"])
    seeds = sweep.get("seeds", [0])
    peak_lrs = sweep.get("peak_lrs", [0.03])

    tasks: List[Dict[str, Any]] = []
    for ms in model_sizes:
        for mode in modes:
            for seed in seeds:
                for lr in peak_lrs:
                    tasks.append({"model_size": ms, "mode": mode, "seed": int(seed), "peak_lr": float(lr)})

    gpus = _parse_gpus(args.gpus)

    # If user didn't specify, let each worker see all GPUs and pick gpu_id=None.
    # (This is slower / less safe, but useful for CPU debugging.)
    if len(gpus) == 0:
        # We'll just run in-process.
        print(f"Running {len(tasks)} tasks sequentially in-process...")
        for t in tqdm(tasks, desc="Sweep"):
            cfg0 = ExperimentConfig(**base_cfg)
            cfg0.model_size = t["model_size"]
            cfg0.mode = t["mode"]
            cfg0.seed = int(t["seed"])
            cfg0.peak_lr = float(t["peak_lr"])
            cfg0.use_tqdm = False
            
            try:
                res = run_one(cfg0)
                payload = {"task": t, "config": asdict(cfg0), "result": res}
                out_path = Path(args.out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                (out_path / f"{_task_name(t)}.json").write_text(json.dumps(payload, indent=2))
            except Exception as e:
                print(f"Task {_task_name(t)} failed: {e}")
        return

    # Round-robin distribute tasks to GPUs.
    buckets: List[List[Dict[str, Any]]] = [[] for _ in gpus]
    for i, t in enumerate(tasks):
        buckets[i % len(gpus)].append(t)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    progress_queue = manager.Queue()

    procs: List[mp.Process] = []
    for worker_id, (gpu_id, bucket) in enumerate(zip(gpus, buckets)):
        p = mp.Process(target=_worker, args=(worker_id, gpu_id, bucket, base_cfg, args.out_dir, progress_queue), daemon=False)
        p.start()
        procs.append(p)

    # Monitor progress
    completed = 0
    total = len(tasks)
    pbar = tqdm(total=total, desc="Sweep Progress")
    
    while completed < total:
        try:
            # Check if all processes are dead (and we might hang if we don't check)
            any_alive = any(p.is_alive() for p in procs)
            if not any_alive and progress_queue.empty():
                break

            item = progress_queue.get(timeout=1.0)
            status = item[0]
            task_name = item[1]

            completed += 1
            pbar.update(1)
            if status == "failed":
                error_msg = item[2] if len(item) > 2 else "Unknown error"
                pbar.write(f"Task failed: {task_name}\nError: {error_msg}")
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\nSweep interrupted. Terminating workers...")
            for p in procs:
                p.terminate()
            break

    pbar.close()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
