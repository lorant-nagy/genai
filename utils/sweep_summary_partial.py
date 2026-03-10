#!/usr/bin/env python3
"""
Profiler Analysis Script (Partial-Run Mode)
============================================
Like profiler_analysis.py, but tolerates an incomplete sweep.

Key additions
-------------
* Infers the *expected* full (T, α) grid from the runs that ARE present,
  combined with every T value that appears in at least one alpha group.
* For each metric, plots a scatter/line chart where:
    - completed (T, α) pairs  → solid filled marker + connected line
    - missing   (T, α) pairs  → hollow × marker on the α-row baseline,
                                 with a light vertical guide line from the
                                 x-axis to where the point would sit.
* A "completion badge" in the top-right corner shows e.g. "55 / 84 runs".
* Validation is relaxed: missing pairs are expected, duplicates still error.

Usage (same as profiler_analysis.py):
    python -m utils.profiler_analysis_partial <runs_folder> <project_name>
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
import matplotlib.patheffects as pe
import numpy as np
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def load_run_data(runs_dir: Path):
    """Load all run configs and metrics from a runs directory."""
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
            metrics = json.load(f)

        power = config['corruption']['process_params']['alpha']
        T     = config['corruption']['process_params']['T']

        best_metrics = {}
        for key, values in metrics.items():
            if key in ['epochs', 'nan'] or not values:
                continue
            finite = [v for v in values
                      if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                K    = 5
                tail = finite[-K:]
                best_metrics[key] = float(np.median(tail))

        all_data.append({
            'power':        power,
            'T':            T,
            'best_metrics': best_metrics,
            'run_name':     run_dir.name,
        })

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Grid inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_full_grid(all_data):
    """
    Return (all_T_sorted, all_alphas_sorted, present_pairs_set).

    The "full" grid is the union of every (T, α) we can infer was *intended*:
    every T value seen across ANY alpha, crossed with every alpha seen.
    """
    present_pairs = {(d['T'], d['power']) for d in all_data}
    all_T      = sorted({d['T']     for d in all_data})
    all_alphas = sorted({d['power'] for d in all_data})
    return all_T, all_alphas, present_pairs


def compute_grid_stats(all_T, all_alphas, present_pairs):
    total    = len(all_T) * len(all_alphas)
    done     = len(present_pairs)
    missing  = {(t, a) for t in all_T for a in all_alphas} - present_pairs
    return total, done, missing


# ─────────────────────────────────────────────────────────────────────────────
# Validation  (relaxed – no consistency check on T counts across alphas)
# ─────────────────────────────────────────────────────────────────────────────

def validate_data(all_data):
    """Only check there are no duplicate (power, T) pairs."""
    pairs = [(d['power'], d['T']) for d in all_data]
    assert len(pairs) == len(set(pairs)), "Duplicate (power, T) pairs found"


# ─────────────────────────────────────────────────────────────────────────────
# Colour / style helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93',
            '#1982C4', '#8AC926', '#FF595E', '#6A994E']
MARKERS  = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*']


def _color(i):  return PALETTE[i % len(PALETTE)]
def _marker(i): return MARKERS[i % len(MARKERS)]


# ─────────────────────────────────────────────────────────────────────────────
# Core plot function
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_partial(
    metric_key: str,
    data_by_power: dict,          # power -> [(T, value), ...]  (only present runs)
    all_T: list,
    all_alphas: list,
    present_pairs: set,
    total_runs: int,
    done_runs: int,
    project_name: str,
):
    """
    Create one metric plot with ghost markers for missing (T, α) cells.

    Visual language
    ---------------
    Present  → filled marker, solid line segment between consecutive points
    Missing  → hollow ✕ at the α-row baseline, subtle vertical ghost line
    Grid     → light horizontal lines at each α (the "expected row")
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    # ── collect value range for ghost line height ──────────────────────────
    all_vals = [v for pts in data_by_power.values() for _, v in pts]
    if not all_vals:
        plt.close(fig)
        return

    v_min, v_max = min(all_vals), max(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0

    # ── background grid lines at each T value (light, dotted) ─────────────
    for t in all_T:
        ax.axvline(t, color='#cccccc', linewidth=0.4, zorder=0, linestyle=':')

    # ── per-alpha series ───────────────────────────────────────────────────
    for i, alpha in enumerate(all_alphas):
        col = _color(i)
        mrk = _marker(i)

        pts_present = sorted(data_by_power.get(alpha, []))
        T_present   = [p[0] for p in pts_present]
        V_present   = [p[1] for p in pts_present]

        missing_T = [t for t in all_T if (t, alpha) not in present_pairs]

        # Ghost vertical guides for missing T values
        for t in missing_T:
            ax.axvline(
                t,
                ymin=0.0, ymax=0.04,           # tiny tick at the bottom
                color=col, linewidth=1.2,
                alpha=0.35, zorder=1,
            )

        # Solid line through present points
        if len(T_present) > 1:
            ax.plot(
                T_present, V_present,
                linestyle='-', linewidth=1.8,
                color=col, alpha=0.85, zorder=3,
            )

        # Filled markers for present points
        if T_present:
            ax.scatter(
                T_present, V_present,
                marker=mrk, s=70,
                color=col, edgecolors='white', linewidths=0.8,
                zorder=4, label=f'α = {alpha}',
            )

        # Hollow ✕ for missing points – placed at a fixed "row baseline"
        # We use a small y-offset per alpha so they don't stack on top of each other
        baseline = v_min - v_range * (0.07 + 0.045 * i)
        if missing_T:
            ax.scatter(
                missing_T,
                [baseline] * len(missing_T),
                marker='x', s=55,
                color=col, linewidths=1.4,
                alpha=0.55, zorder=2,
            )
            # small label on the left so the reader knows which α it is
            ax.text(
                all_T[0] - (all_T[-1] - all_T[0]) * 0.015,
                baseline,
                f'α={alpha}',
                ha='right', va='center',
                fontsize=7.5, color=col, alpha=0.6,
            )

    # ── axis labels & title ────────────────────────────────────────────────
    ax.set_xlabel('T  (Terminal Time)', fontsize=12)
    ax.set_ylabel(f'Best {metric_key.upper()}', fontsize=12)
    ax.set_title(
        f'Best {metric_key.upper()} vs T',
        fontsize=14, fontweight='bold',
    )

    # ── completion badge ───────────────────────────────────────────────────
    pct = 100.0 * done_runs / total_runs if total_runs else 0
    badge_text = f'{done_runs} / {total_runs} runs  ({pct:.0f}%)'
    ax.text(
        0.98, 0.98, badge_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='#fffbe6', edgecolor='#e0c040',
            linewidth=1.2, alpha=0.92,
        ),
        zorder=5,
    )

    # ── missing-data legend entry ─────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles_extra = [
        Line2D([0], [0], marker='x', color='grey', linestyle='None',
               markersize=7, markeredgewidth=1.4,
               label='missing run  (×  below axis)'),
    ]

    legend = ax.legend(
        fontsize=10,
        handles=ax.get_legend_handles_labels()[0] + legend_handles_extra,
        labels=ax.get_legend_handles_labels()[1]  + ['missing run  (×  below axis)'],
        loc='upper left',
        framealpha=0.88,
    )

    # ── grid & layout ──────────────────────────────────────────────────────
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.set_xlim(
        all_T[0]  - (all_T[-1] - all_T[0]) * 0.04,
        all_T[-1] + (all_T[-1] - all_T[0]) * 0.04,
    )

    plt.tight_layout()

    # ── upload to WandB ────────────────────────────────────────────────────
    wandb.log({f'{metric_key}_vs_T_partial': wandb.Image(fig)})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Completion heatmap  (bonus: shows the full grid at a glance)
# ─────────────────────────────────────────────────────────────────────────────

def plot_completion_heatmap(all_T, all_alphas, present_pairs, total, done):
    """
    A binary heatmap: green = done, red-hatched = missing.
    Gives an instant overview of which cells are still outstanding.
    """
    grid = np.zeros((len(all_alphas), len(all_T)), dtype=float)
    for ai, a in enumerate(all_alphas):
        for ti, t in enumerate(all_T):
            grid[ai, ti] = 1.0 if (t, a) in present_pairs else 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(all_T) * 0.55), 3.5))

    # Green / red colourmap
    cmap_bw = matplotlib.colors.ListedColormap(['#ffdddd', '#c8f0c8'])
    ax.imshow(grid, cmap=cmap_bw, vmin=0, vmax=1,
              aspect='auto', origin='lower')

    # Hatch pattern over missing cells
    for ai in range(len(all_alphas)):
        for ti in range(len(all_T)):
            if grid[ai, ti] == 0:
                rect = plt.Rectangle(
                    (ti - 0.5, ai - 0.5), 1, 1,
                    fill=False, hatch='////',
                    edgecolor='#cc4444', linewidth=0, alpha=0.6,
                )
                ax.add_patch(rect)
            else:
                ax.text(ti, ai, '✓', ha='center', va='center',
                        fontsize=9, color='#2a7a2a', fontweight='bold')

    ax.set_xticks(range(len(all_T)))
    ax.set_xticklabels([f'{t:.2f}' for t in all_T],
                       rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(all_alphas)))
    ax.set_yticklabels([f'α={a}' for a in all_alphas], fontsize=9)
    ax.set_xlabel('T  (Terminal Time)', fontsize=11)
    ax.set_title(
        f'Sweep Completion Grid  —  {done}/{total} runs done'
        f'  ({100*done/total:.0f}%)',
        fontsize=12, fontweight='bold',
    )

    plt.tight_layout()
    wandb.log({'completion_heatmap': wandb.Image(fig)})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python profiler_analysis_partial.py <runs_folder> <project_name>")
        sys.exit(1)

    runs_folder  = sys.argv[1]
    project_name = sys.argv[2]

    print(f"Loading runs from: {runs_folder}")
    runs_dir = Path(runs_folder)

    if not runs_dir.exists():
        print(f"Error: Directory {runs_folder} does not exist")
        sys.exit(1)

    # ── load & validate ────────────────────────────────────────────────────
    all_data = load_run_data(runs_dir)

    if not all_data:
        print("No valid run data found")
        sys.exit(1)

    print(f"Found {len(all_data)} completed runs")
    validate_data(all_data)

    # ── infer the full intended grid ───────────────────────────────────────
    all_T, all_alphas, present_pairs = infer_full_grid(all_data)
    total, done, missing_pairs = compute_grid_stats(all_T, all_alphas, present_pairs)

    print(f"Inferred grid: {len(all_alphas)} α values × {len(all_T)} T values = {total} cells")
    print(f"Done: {done}  |  Missing: {len(missing_pairs)}")

    if missing_pairs:
        by_alpha = defaultdict(list)
        for (t, a) in sorted(missing_pairs):
            by_alpha[a].append(t)
        for a in sorted(by_alpha):
            print(f"  α={a}: missing T = {by_alpha[a]}")

    # ── init WandB ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project=project_name,
        name=f'partial_profiling_{timestamp}',
        job_type='analysis',
        config={
            'runs_folder':   runs_folder,
            'total_planned': total,
            'done':          done,
            'missing':       len(missing_pairs),
            'completion_pct': round(100 * done / total, 1) if total else 0,
        },
    )

    # ── completion heatmap (always produced first) ─────────────────────────
    plot_completion_heatmap(all_T, all_alphas, present_pairs, total, done)

    # ── collect metrics ────────────────────────────────────────────────────
    all_metrics: set = set()
    for d in all_data:
        all_metrics.update(d['best_metrics'].keys())

    # ── one plot per metric ────────────────────────────────────────────────
    for metric_key in sorted(all_metrics):
        data_by_power: dict = defaultdict(list)
        for d in all_data:
            if metric_key in d['best_metrics']:
                data_by_power[d['power']].append(
                    (d['T'], d['best_metrics'][metric_key])
                )

        if data_by_power:
            plot_metric_partial(
                metric_key     = metric_key,
                data_by_power  = data_by_power,
                all_T          = all_T,
                all_alphas     = all_alphas,
                present_pairs  = present_pairs,
                total_runs     = total,
                done_runs      = done,
                project_name   = project_name,
            )
            print(f"  Plotted: {metric_key}")

    wandb.finish()
    print("Partial analysis complete!")


if __name__ == "__main__":
    main()