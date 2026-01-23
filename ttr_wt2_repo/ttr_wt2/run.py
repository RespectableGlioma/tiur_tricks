"""CLI for a single experiment run.

Example:
  python -m ttr_wt2.run --model_size small --mode ttr --seed 0 --peak_lr 0.03 --J_target 0.02

Notes:
- For mode=jfinder, J_target is ignored; we first run the J-finder to compute it.
- Results are written as a JSON file (default: runs/<auto-name>.json).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .core import ExperimentConfig, DEFAULT_MODEL_SIZES, run_one, save_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="small", choices=sorted(DEFAULT_MODEL_SIZES.keys()))
    p.add_argument(
        "--mode",
        type=str,
        default="ttr",
        choices=["nowarmup", "warmup", "ttr", "autottr", "jfinder"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peak_lr", type=float, default=0.03)
    p.add_argument("--J_target", type=float, default=None)

    p.add_argument("--total_steps", type=int, default=2000)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_max_batches", type=int, default=50)

    p.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)

    p.add_argument("--keep_traces", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cpu, or cuda:<index>",
    )

    p.add_argument("--out", type=str, default=None, help="Path to write JSON result")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = ExperimentConfig(
        model_size=args.model_size,
        mode=args.mode,
        seed=args.seed,
        peak_lr=args.peak_lr,
        J_target=args.J_target,
        total_steps=args.total_steps,
        warmup_frac=args.warmup_frac,
        block_size=args.block_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches,
        use_bf16=args.use_bf16,
        clip_norm=args.clip_norm,
        weight_decay=args.weight_decay,
        keep_traces=args.keep_traces,
        device_str=args.device,
    )

    out = run_one(cfg)

    if args.out is None:
        out_dir = Path("runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{cfg.model_size}_{cfg.mode}_seed{cfg.seed}_lr{cfg.peak_lr:g}.json"
        args.out = str(out_dir / fname)

    save_json(args.out, out)
    print(json.dumps({"saved": args.out, **{k: out.get(k) for k in ["best_val_ppl", "elapsed_s", "J_target_final"]}}, indent=2))


if __name__ == "__main__":
    main()
