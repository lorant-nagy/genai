#!/usr/bin/env python3
"""
N-Steps Analysis Script
=======================
Plots metric quality vs. corruptor n_steps, with one curve per terminal time T.
Produces one figure per metric and uploads them all to WandB.

Designed for sweeps where the varying axes are:
  - corruptor_params.n_steps   (x-axis)
  - corruption.process_params.T  (one curve per value)

Any other sweep axes (e.g. alpha) are aggregated by taking the median over
all runs that share the same (T, n_steps) pair, so the script stays useful
even when more axes are present.

Usage (direct):
    python -m utils.nsteps_analysis <runs_folder> <project_name>

Usage (via experiment.py):
    python experiment.py <runs_folder> <project_name> --nsteps
"""

import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93',
           '#1982C4', '#8AC926', '#FF595E', '#6A994E']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*']

def _color(i):  return PALETTE[i % len(PALETTE)]
def _marker(i): return MARKERS[i % len(MARKERS)]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run_data(runs_dir: Path):
    """
    Load all completed runs.  For each run, extract:
      - T           : terminal time
      - n_steps     : corruptor n_steps
      - best_metrics: median of the last 5 finite values for every metric
    """
    all_data = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path  = run_dir / 'config.yaml'
        metrics_path = run_dir / 'metrics_evo.json'

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics_evo = json.load(f)

        T       = config['corruption']['process_params']['T']
        n_steps = config['corruption']['corruptor_params']['n_steps']

        best_metrics = {}
        for key, values in metrics_evo.items():
            if key in ('epochs', 'nan') or not values:
                continue
            finite = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                best_metrics[key] = float(np.median(finite[-5:]))

        all_data.append({
            'T':            T,
            'n_steps':      n_steps,
            'best_metrics': best_metrics,
            'run_name':     run_dir.name,
        })

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation  (in case multiple runs share the same (T, n_steps) pair)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_data):
    """
    For each (T, n_steps) pair, aggregate best_metrics by median.
    Returns a list of dicts with the same schema as load_run_data output,
    but with one entry per unique (T, n_steps).
    """
    bucket = defaultdict(list)          # (T, n_steps) -> [best_metrics dict, ...]
    for d in all_data:
        bucket[(d['T'], d['n_steps'])].append(d['best_metrics'])

    aggregated = []
    for (T, n_steps), metric_list in bucket.items():
        all_keys = set(k for m in metric_list for k in m)
        agg_metrics = {}
        for k in all_keys:
            vals = [m[k] for m in metric_list if k in m]
            if vals:
                agg_metrics[k] = float(np.median(vals))
        aggregated.append({'T': T, 'n_steps': n_steps, 'best_metrics': agg_metrics})

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_nsteps(metric_key: str, aggregated: list, project_name: str):
    """
    One figure: x = n_steps, y = best metric value.
    One curve per distinct T value.
    """
    # Collect data grouped by T
    data_by_T = defaultdict(list)   # T -> [(n_steps, value), ...]
    for d in aggregated:
        if metric_key in d['best_metrics']:
            data_by_T[d['T']].append((d['n_steps'], d['best_metrics'][metric_key]))

    if not data_by_T:
        print(f"  No data for {metric_key}, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, T in enumerate(sorted(data_by_T)):
        pts = sorted(data_by_T[T])          # sort by n_steps
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]

        ax.plot(
            xs, ys,
            color=_color(i), marker=_marker(i),
            markersize=8, linewidth=2,
            label=f'T = {T}',
        )

    ax.set_xlabel('n_steps', fontsize=12)
    ax.set_ylabel(metric_key.upper(), fontsize=12)
    ax.set_title(f'{metric_key.upper()} vs n_steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.log({f'{metric_key}_vs_nsteps': wandb.Image(fig)})
    plt.close(fig)
    print(f"  Logged: {metric_key}_vs_nsteps")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m utils.nsteps_analysis <runs_folder> <project_name>")
        sys.exit(1)

    runs_folder  = sys.argv[1]
    project_name = sys.argv[2]

    runs_dir = Path(runs_folder)
    if not runs_dir.exists():
        print(f"Error: {runs_folder} does not exist")
        sys.exit(1)

    print(f"Loading runs from: {runs_folder}")
    all_data = load_run_data(runs_dir)

    if not all_data:
        print("No valid run data found")
        sys.exit(1)

    print(f"Found {len(all_data)} runs")

    # Aggregate over any extra axes (e.g. alpha)
    aggregated = aggregate(all_data)

    T_values      = sorted({d['T']       for d in aggregated})
    nsteps_values = sorted({d['n_steps'] for d in aggregated})
    print(f"T values:      {T_values}")
    print(f"n_steps range: {nsteps_values}")

    # Collect all metric keys
    all_metric_keys = sorted({
        k for d in aggregated for k in d['best_metrics']
    })
    print(f"Metrics: {all_metric_keys}")

    if not all_metric_keys:
        print("No metrics found")
        sys.exit(1)

    # Init WandB
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project=project_name,
        name=f'nsteps_analysis_{timestamp}',
        job_type='nsteps_analysis',
        config={
            'runs_folder': runs_folder,
            'n_runs':      len(all_data),
            'T_values':    T_values,
            'nsteps':      nsteps_values,
        },
    )

    for metric_key in all_metric_keys:
        plot_metric_vs_nsteps(metric_key, aggregated, project_name)

    wandb.finish()
    print("n_steps analysis complete!")


if __name__ == "__main__":
    main()