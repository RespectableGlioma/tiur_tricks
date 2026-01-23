"""Aggregate JSON run outputs into a single CSV.

Example:
  python -m ttr_wt2.aggregate --runs_dir runs --out results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=kk + "."))
        else:
            out[kk] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        d_flat = _flatten(d)
        d_flat["_file"] = p.name
        rows.append(d_flat)

    if not rows:
        raise SystemExit(f"No JSON run files found in: {runs_dir}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
