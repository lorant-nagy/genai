#!/usr/bin/env python3
"""
flex_heatmap.py  —  Heatmap from the experiment database
=========================================================
Reads a DB file produced by run_db.py, plots a heatmap, and uploads
it to WandB.

All w1_* metrics are discovered automatically; no VALUE constant needed.

USAGE
-----
    python -m utils.flex_heatmap <db_path> <project_name>

EXAMPLES
--------
    python -m utils.flex_heatmap db/surface.json surface
"""

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import wandb

import argparse

try:
    from utils.run_db import load_db, AXIS_ALIASES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.run_db import load_db, AXIS_ALIASES




""" # ── Configuration ─────────────────────────────────────────────────────────────

# How many trailing finite values to use.  Set to "all" to use the entire series.
REDUCE_TAIL = "all"

# Aggregation function applied to each w1_k time-series: "median" or "min".
REDUCE_FN   = "min"

# Aggregation function applied across seeds per cell: "mean", "median", "min".
SEED_AGG    = "mean"

AXIS_X     = "T"
AXIS_Y     = "alpha"
AXIS_SLICE = ("n_steps", 100)   # (param, value)  — fixes the third dimension

# ───────────────────────────────────────────────────────────────────────────── """


def _reduce(series, REDUCE_TAIL, REDUCE_FN) -> float | None:
    """
    Collapse a single w1_k time-series to a scalar.

    REDUCE_TAIL selects which values to consider:
      int   — last N finite values
      "all" — all finite values in the series

    REDUCE_FN is then applied to that selection:
      "median" — median
      "min"    — minimum

    Returns None if there are no finite values.
    """
    fin = [v for v in series if isinstance(v, (int, float)) and np.isfinite(v)]
    if not fin:
        return None
    tail = fin if REDUCE_TAIL == "all" else fin[-REDUCE_TAIL:]
    if REDUCE_FN == "min":
        return float(np.min(tail))
    return float(np.median(tail))


def _seed_agg(values, SEED_AGG) -> float:
    """
    Collapse per-seed scalars for a single cell to one value.

    SEED_AGG controls the method:
      "mean"   — arithmetic mean  (default)
      "median" — median
      "min"    — minimum
    """
    if SEED_AGG == "min":
        return float(np.min(values))
    if SEED_AGG == "median":
        return float(np.median(values))
    return float(np.mean(values))


def _annotate(ax, grid: np.ndarray) -> None:
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            v = grid[ri, ci]
            if np.isfinite(v):
                ax.text(ci, ri, f"{v:.4f}",
                        ha="center", va="center",
                        fontsize=7, color="white",
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground="black")])


def plot(records, w1_keys, AXIS_SLICE, AXIS_X, AXIS_Y, REDUCE_TAIL, REDUCE_FN, SEED_AGG, REMARK) -> plt.Figure:
    slice_param, slice_val = AXIS_SLICE

    filtered = [r for r in records
                if r["params"].get(slice_param) == slice_val]

    if not filtered:
        raise ValueError(
            f"No records with {slice_param}={slice_val}. "
            f"Available: {sorted({r['params'].get(slice_param) for r in records})}"
        )

    x_vals = sorted({r["params"][AXIS_X] for r in filtered
                     if r["params"].get(AXIS_X) is not None})
    y_vals = sorted({r["params"][AXIS_Y] for r in filtered
                     if r["params"].get(AXIS_Y) is not None})

    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in filtered:
        xv = r["params"].get(AXIS_X)
        yv = r["params"].get(AXIS_Y)
        if xv is None or yv is None:
            continue
        for key in w1_keys:
            series = r["metrics"].get(key)
            if series is None:
                continue
            scalar = _reduce(series, REDUCE_TAIL, REDUCE_FN)
            if scalar is not None:
                cells[(xv, yv)].append(scalar)

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            vs = cells.get((x, y))
            if vs:
                grid[yi, xi] = _seed_agg(vs, SEED_AGG)

    finite = grid[np.isfinite(grid)]
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0

    fig, ax = plt.subplots(figsize=(max(5, len(x_vals) * 1.4),
                                    max(4, len(y_vals) * 1.0)))
    im = ax.imshow(grid, cmap=plt.cm.viridis_r,
                   vmin=vmin, vmax=vmax,
                   aspect="auto", origin="lower")
    _annotate(ax, grid)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(AXIS_X, fontsize=11)
    ax.set_ylabel(AXIS_Y, fontsize=11)

    n_seeds      = len(w1_keys)
    tail_label   = "all" if REDUCE_TAIL == "all" else f"tail({REDUCE_TAIL})"
    reduce_label = f"{REDUCE_FN}_{tail_label}"
    ax.set_title(
        f"W1 ({n_seeds} seeds, seed_agg={SEED_AGG})  —  {AXIS_Y} × {AXIS_X}"
        f"  |  {slice_param}={slice_val}"
        f"  |  series_reduce={reduce_label}",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if REMARK:
        fig.text(0.5, 0.01, f"Remark: {REMARK}",
                 ha="center", va="bottom", fontsize=9,
                 style="italic", color="#444444",
                 wrap=True)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        plt.tight_layout()

    return fig


def main() -> None:

    parser = argparse.ArgumentParser(
        description = "parse arguments for heatmap generator"
    )

    parser.add_argument("db_path")
    parser.add_argument("project_name")
    parser.add_argument("--reduce-tail", dest="REDUCE_TAIL", default="all")
    parser.add_argument("--reduce-fn", dest = "REDUCE_FN", default="min")
    parser.add_argument("--seed-agg", dest="SEED_AGG", default="mean")
    parser.add_argument("--axis-x", dest="AXIS_X")
    parser.add_argument("--axis-y", dest="AXIS_Y")
    parser.add_argument("--axis-slice-key", dest="AXIS_SLICE_KEY")
    parser.add_argument("--axis-slice-value", dest="AXIS_SLICE_VALUE")
    parser.add_argument("--remark", dest="remark", default=None)

    args = parser.parse_args()

    args.AXIS_SLICE = (args.AXIS_SLICE_KEY, int(args.AXIS_SLICE_VALUE))

    records = load_db(args.db_path)
    print(f"Loaded {len(records)} records from {args.db_path}")

    available = sorted({k for r in records for k in r["metrics"]})

    w1_keys = sorted(
        (k for k in available if re.fullmatch(r"w1_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    if not w1_keys:
        print(f"Error: no w1_* metrics found. Available: {available}")
        sys.exit(1)
    print(f"  Found {len(w1_keys)} w1 seed(s): {w1_keys}")

    fig = plot(records, w1_keys, args.AXIS_SLICE, args.AXIS_X, args.AXIS_Y, args.REDUCE_TAIL, args.REDUCE_FN, args.SEED_AGG, args.remark)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_seeds   = len(w1_keys)
    wandb.init(
        project=args.project_name,
        name=f"flex_heatmap_{args.AXIS_X}_vs_{args.AXIS_Y}_w1x{n_seeds}_{timestamp}",
        job_type="flex_heatmap",
        config={
            "x":           args.AXIS_X,
            "y":           args.AXIS_Y,
            "slice":       args.AXIS_SLICE,
            "w1_keys":     w1_keys,
            "seed_agg":    args.SEED_AGG,
            "reduce_fn":   args.REDUCE_FN,
            "reduce_tail": args.REDUCE_TAIL,
        },
    )
    log_key = f"w1/heatmap_{args.AXIS_X}_vs_{args.AXIS_Y}"
    wandb.log({log_key: wandb.Image(fig)})
    print(f"  Logged to WandB: {log_key}")
    wandb.finish()

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()