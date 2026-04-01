#!/usr/bin/env python3
"""
Heatmap Analysis Script
=======================
Produces a single heatmap from a sweep and uploads it to WandB.

The x and y axes are config parameters (T, alpha, n_steps).
The cell values are a metric from metrics_evo.json.
If multiple runs share the same (x, y) cell (because a third axis varies),
values are aggregated by taking the minimum (lower-is-better convention).

Usage (direct):
    python -m utils.heatmap_analysis <runs_folder> <project_name> \\
        --x <param> --y <param> --values <metric>

Usage (via analyse.py):
    python -m utils.analyse <runs_folder> <project_name> \\
        --heatmap --x <param> --y <param> --values <metric>

Valid axis params: T, alpha, n_steps
The metric name must match a key in metrics_evo.json (e.g. w1, fid, kid).

Examples:
    python -m utils.analyse run_logs/surface_response/runs surface_response \\
        --heatmap --x T --y alpha --values w1

    python -m utils.analyse run_logs/nsteps/runs nsteps \\
        --heatmap --x alpha --y n_steps --values w1
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml


# ── Config axis registry ──────────────────────────────────────────────────────

AXIS_EXTRACTORS = {
    "T":       lambda cfg: cfg["corruption"]["process_params"]["T"],
    "alpha":   lambda cfg: cfg["corruption"]["process_params"]["alpha"],
    "n_steps": lambda cfg: cfg["corruption"]["corruptor_params"]["n_steps"],
}

VALID_AXES = sorted(AXIS_EXTRACTORS.keys())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run_data(runs_dir: Path) -> list[dict]:
    """
    Load all completed runs from runs_dir.
    Each entry: {T, alpha, n_steps, best_metrics, run_name}
    best_metrics: median of last 5 finite values per metric key.
    """
    all_data = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path  = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics_evo.json"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics_evo = json.load(f)

        params = {name: extractor(cfg)
                  for name, extractor in AXIS_EXTRACTORS.items()}

        best_metrics = {}
        for key, values in metrics_evo.items():
            if key in ("epochs", "nan") or not values:
                continue
            finite = [v for v in values
                      if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                best_metrics[key] = float(np.median(finite[-5:]))

        all_data.append({
            **params,
            "best_metrics": best_metrics,
            "run_name":     run_dir.name,
        })

    return all_data


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _annotate_heatmap(ax, grid):
    """Print numeric values inside each cell."""
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            v = grid[ri, ci]
            if np.isfinite(v):
                ax.text(ci, ri, f"{v:.4f}",
                        ha="center", va="center",
                        fontsize=7, color="white",
                        path_effects=[
                            matplotlib.patheffects.withStroke(
                                linewidth=1.5, foreground="black")
                        ])


def plot_heatmap(x_param: str, y_param: str, metric: str,
                 all_data: list[dict], project_name: str):
    """
    Build and log a single heatmap.
    x_param / y_param : config axis names (T, alpha, n_steps)
    metric            : key in best_metrics
    """
    x_vals = sorted({d[x_param] for d in all_data})
    y_vals = sorted({d[y_param] for d in all_data})

    # Aggregate cells: collect all values, then take min (lower-is-better)
    cells: dict[tuple, list[float]] = defaultdict(list)
    for d in all_data:
        v = d["best_metrics"].get(metric)
        if v is not None and np.isfinite(v):
            cells[(d[x_param], d[y_param])].append(v)

    n_aggregated = sum(1 for vs in cells.values() if len(vs) > 1)
    if n_aggregated:
        print(f"  Warning: {n_aggregated} cell(s) had multiple runs — "
              f"aggregated by min (lower-is-better).")

    # shape: (len(y_vals), len(x_vals))
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            vs = cells.get((x, y))
            if vs:
                grid[yi, xi] = min(vs)

    finite = grid[np.isfinite(grid)]
    vmin, vmax = (float(finite.min()), float(finite.max())) \
        if len(finite) else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(max(5, len(x_vals) * 1.4),
                                    max(4, len(y_vals) * 1.0)))

    im = ax.imshow(
        grid, cmap=plt.cm.viridis_r, vmin=vmin, vmax=vmax,
        aspect="auto", origin="lower",
    )
    _annotate_heatmap(ax, grid)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(x_param, fontsize=11)
    ax.set_ylabel(y_param, fontsize=11)
    ax.set_title(f"{metric.upper()}  —  {y_param} × {x_param} heatmap",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    log_key = f"{metric}/heatmap_{x_param}_vs_{y_param}"
    wandb.log({log_key: wandb.Image(fig)})
    plt.close(fig)
    print(f"  Logged: {log_key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Produce a single heatmap and upload to WandB.",
        add_help=True,
    )
    parser.add_argument("runs_folder",  help="Path to sweep runs directory")
    parser.add_argument("project_name", help="WandB project name")
    parser.add_argument("--x",      required=True, choices=VALID_AXES,
                        help="Config parameter for the x-axis")
    parser.add_argument("--y",      required=True, choices=VALID_AXES,
                        help="Config parameter for the y-axis")
    parser.add_argument("--values", required=True,
                        help="Metric key for cell values (e.g. w1, fid)")
    args = parser.parse_args()

    if args.x == args.y:
        print("Error: --x and --y must be different parameters.")
        sys.exit(1)

    runs_dir = Path(args.runs_folder)
    if not runs_dir.exists():
        print(f"Error: {args.runs_folder} does not exist.")
        sys.exit(1)

    print(f"Loading runs from: {args.runs_folder}")
    all_data = load_run_data(runs_dir)

    if not all_data:
        print("No valid run data found (need config.yaml + metrics_evo.json).")
        sys.exit(1)

    print(f"Found {len(all_data)} runs")

    available_metrics = sorted({k for d in all_data for k in d["best_metrics"]})
    if args.values not in available_metrics:
        print(f"Error: metric '{args.values}' not found in run data.")
        print(f"Available metrics: {available_metrics}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=args.project_name,
        name=f"heatmap_{args.x}_vs_{args.y}_{args.values}_{timestamp}",
        job_type="heatmap_analysis",
        config={
            "runs_folder":  args.runs_folder,
            "n_runs":       len(all_data),
            "x_param":      args.x,
            "y_param":      args.y,
            "metric":       args.values,
        },
    )

    plot_heatmap(args.x, args.y, args.values, all_data, args.project_name)

    wandb.finish()
    print("Heatmap analysis complete!")


if __name__ == "__main__":
    main()