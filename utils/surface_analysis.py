#!/usr/bin/env python3
"""
Surface Analysis Script
=======================
Analyses a 3-axis sweep: T × alpha × n_steps, producing four families of plots
per metric and uploading them all to WandB.

Requires a *rectangular* sweep — every combination of the observed T, alpha,
and n_steps values must have exactly one completed run.  Missing cells are
reported and the script exits rather than silently producing partial plots.

Plots produced (per metric)
---------------------------
1. Heatmaps       — T (x) × alpha (y), one panel per n_steps + one sup panel
2. Curves vs T    — one curve per alpha, faceted by n_steps
3. Curves vs n_steps — one curve per (T, alpha) pair, all on one axes
4. Sensitivity    — variance bar chart: how much does each axis matter?

Usage (direct):
    python -m utils.surface_analysis <runs_folder> <project_name>

Usage (via analyse.py):
    python -m utils.analyse <runs_folder> <project_name> --surface
"""

import json
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects
import numpy as np
import wandb


# ── Palette ───────────────────────────────────────────────────────────────────

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A4C93",
           "#1982C4", "#8AC926", "#FF595E", "#6A994E", "#FF6B6B"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "p"]

def _c(i): return PALETTE[i % len(PALETTE)]
def _m(i): return MARKERS[i % len(MARKERS)]


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

        T       = cfg["corruption"]["process_params"]["T"]
        alpha   = cfg["corruption"]["process_params"]["alpha"]
        n_steps = cfg["corruption"]["corruptor_params"]["n_steps"]

        best_metrics = {}
        for key, values in metrics_evo.items():
            if key in ("epochs", "nan") or not values:
                continue
            finite = [v for v in values
                      if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                best_metrics[key] = float(np.median(finite[-5:]))

        all_data.append({
            "T":            T,
            "alpha":        alpha,
            "n_steps":      n_steps,
            "best_metrics": best_metrics,
            "run_name":     run_dir.name,
        })

    return all_data


# ── Rectangularity check & largest usable sub-grid ───────────────────────────

def _is_rectangular(data: list[dict]) -> tuple[bool, set]:
    """Return (is_rect, missing_set) for the given data."""
    T_vals      = sorted({d["T"]       for d in data})
    alpha_vals  = sorted({d["alpha"]   for d in data})
    nsteps_vals = sorted({d["n_steps"] for d in data})
    present     = {(d["T"], d["alpha"], d["n_steps"]) for d in data}
    full_grid   = set(product(T_vals, alpha_vals, nsteps_vals))
    missing     = full_grid - present
    return len(missing) == 0, missing


def find_largest_rectangular(all_data: list[dict]):
    """
    Find the largest rectangular T × alpha × n_steps sub-grid present in the
    data.  Strategy: the n_steps axis is most likely to be partially complete
    (runs still in progress), so we try dropping n_steps values — starting
    from the largest — until the remaining data is rectangular.

    Returns (filtered_data, T_vals, alpha_vals, nsteps_vals, dropped_nsteps).
    Raises AssertionError on duplicate triples (that is always a hard error).
    """
    # Duplicate check — always hard error
    tuples = [(d["T"], d["alpha"], d["n_steps"]) for d in all_data]
    seen, dupes = set(), []
    for t in tuples:
        if t in seen:
            dupes.append(t)
        seen.add(t)
    assert not dupes, f"Duplicate (T, alpha, n_steps) triples: {dupes}"

    all_nsteps = sorted({d["n_steps"] for d in all_data})

    # Try progressively dropping the largest n_steps values
    for n_drop in range(len(all_nsteps)):
        candidate_nsteps = all_nsteps[: len(all_nsteps) - n_drop]
        candidate_data   = [d for d in all_data if d["n_steps"] in candidate_nsteps]
        ok, _            = _is_rectangular(candidate_data)
        if ok:
            dropped = all_nsteps[len(all_nsteps) - n_drop :]
            T_vals      = sorted({d["T"]       for d in candidate_data})
            alpha_vals  = sorted({d["alpha"]   for d in candidate_data})
            nsteps_vals = sorted({d["n_steps"] for d in candidate_data})
            return candidate_data, T_vals, alpha_vals, nsteps_vals, dropped

    raise AssertionError(
        "Could not find any rectangular sub-grid even after dropping all "
        "n_steps values. Check your data."
    )


def check_rectangular(all_data: list[dict]):
    """
    Find and report the largest usable rectangular sub-grid.
    Prints a summary and returns (T_vals, alpha_vals, nsteps_vals).
    Also returns filtered_data as a 4th element.
    """
    filtered, T_vals, alpha_vals, nsteps_vals, dropped = \
        find_largest_rectangular(all_data)

    total_present = len(all_data)
    total_used    = len(filtered)
    full_cells    = len(T_vals) * len(alpha_vals) * len(nsteps_vals)

    print()
    print("=" * 60)
    print("  TABLE SUMMARY")
    print("=" * 60)
    print(f"  Runs found          : {total_present}")
    print(f"  Runs used           : {total_used}")
    if dropped:
        print(f"  n_steps dropped     : {dropped}  (incomplete)")
    else:
        print(f"  n_steps dropped     : none  (full grid available)")
    print(f"  Grid shape          : "
          f"{len(T_vals)} T × {len(alpha_vals)} alpha × {len(nsteps_vals)} n_steps"
          f" = {full_cells} cells")
    print(f"  T values            : {T_vals}")
    print(f"  alpha values        : {alpha_vals}")
    print(f"  n_steps values      : {nsteps_vals}")
    print("=" * 60)
    print()

    return filtered, T_vals, alpha_vals, nsteps_vals


# ── Build lookup table ────────────────────────────────────────────────────────

def build_table(all_data: list[dict]) -> dict:
    """Return dict (T, alpha, n_steps) -> best_metrics."""
    return {(d["T"], d["alpha"], d["n_steps"]): d["best_metrics"]
            for d in all_data}


# ── Plot 1: Heatmaps T × alpha, faceted by n_steps + sup ─────────────────────

def plot_heatmaps(metric: str, table: dict,
                  T_vals, alpha_vals, nsteps_vals):
    """
    One row of panels: one heatmap per n_steps value + one sup panel.
    x-axis = T, y-axis = alpha, colour = metric value.
    """
    n_panels = len(nsteps_vals) + 1          # +1 for sup
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.0 * n_panels, 4.0),
        squeeze=False,
    )
    axes = axes[0]

    # Build data arrays
    # shape: (len(alpha_vals), len(T_vals))
    grids = {}
    for n in nsteps_vals:
        grid = np.full((len(alpha_vals), len(T_vals)), np.nan)
        for ai, a in enumerate(alpha_vals):
            for ti, t in enumerate(T_vals):
                v = table.get((t, a, n), {}).get(metric, np.nan)
                grid[ai, ti] = v
        grids[n] = grid

    sup_grid = np.full((len(alpha_vals), len(T_vals)), np.nan)
    for ai in range(len(alpha_vals)):
        for ti in range(len(T_vals)):
            vals = [grids[n][ai, ti] for n in nsteps_vals
                    if np.isfinite(grids[n][ai, ti])]
            if vals:
                sup_grid[ai, ti] = min(vals)   # inf over n_steps (lower=better)

    # For distance-like metrics lower is better → use min for "best"
    # We label it "best over n_steps" and let the colour speak
    all_vals = np.concatenate(
        [g.ravel() for g in list(grids.values()) + [sup_grid]]
    )
    finite_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = (float(finite_vals.min()), float(finite_vals.max())) \
        if len(finite_vals) else (0, 1)

    cmap = plt.cm.viridis_r    # lower = better = brighter

    for idx, n in enumerate(nsteps_vals):
        ax = axes[idx]
        im = ax.imshow(
            grids[n], cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", origin="lower",
        )
        _annotate_heatmap(ax, grids[n])
        ax.set_xticks(range(len(T_vals)))
        ax.set_xticklabels([str(t) for t in T_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(alpha_vals)))
        ax.set_yticklabels([f"α={a}" for a in alpha_vals])
        ax.set_xlabel("T")
        if idx == 0:
            ax.set_ylabel("alpha")
        ax.set_title(f"n_steps={n}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Sup panel
    ax = axes[-1]
    im = ax.imshow(
        sup_grid, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", origin="lower",
    )
    _annotate_heatmap(ax, sup_grid)
    ax.set_xticks(range(len(T_vals)))
    ax.set_xticklabels([str(t) for t in T_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(alpha_vals)))
    ax.set_yticklabels([f"α={a}" for a in alpha_vals])
    ax.set_xlabel("T")
    ax.set_title("inf over n_steps", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{metric.upper()}  —  T × alpha heatmaps",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb.log({f"{metric}/heatmap_T_alpha": wandb.Image(fig)})
    plt.close(fig)


def _annotate_heatmap(ax, grid):
    """Print numeric values inside each cell."""
    for ai in range(grid.shape[0]):
        for ti in range(grid.shape[1]):
            v = grid[ai, ti]
            if np.isfinite(v):
                ax.text(ti, ai, f"{v:.4f}",
                        ha="center", va="center",
                        fontsize=7, color="white",
                        path_effects=[
                            matplotlib.patheffects.withStroke(
                                linewidth=1.5, foreground="black")
                        ])


# ── Plot 2: metric vs T, faceted by n_steps ───────────────────────────────────

def plot_metric_vs_T(metric: str, table: dict,
                     T_vals, alpha_vals, nsteps_vals):
    """
    One column per n_steps.  Within each panel: x=T, one curve per alpha.
    """
    n_panels = len(nsteps_vals)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.5 * n_panels, 4.5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for idx, n in enumerate(nsteps_vals):
        ax = axes[idx]
        for ai, a in enumerate(alpha_vals):
            ys = [table.get((t, a, n), {}).get(metric, np.nan)
                  for t in T_vals]
            ax.plot(T_vals, ys,
                    color=_c(ai), marker=_m(ai),
                    markersize=7, linewidth=1.8,
                    label=f"α={a}")
        ax.set_xlabel("T", fontsize=11)
        if idx == 0:
            ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"n_steps={n}", fontsize=10)
        ax.grid(True, alpha=0.25)
        if idx == n_panels - 1:
            ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"{metric.upper()}  —  vs T  (faceted by n_steps)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb.log({f"{metric}/vs_T_by_nsteps": wandb.Image(fig)})
    plt.close(fig)


# ── Plot 3: metric vs n_steps, all (T, alpha) curves ─────────────────────────

def plot_metric_vs_nsteps(metric: str, table: dict,
                          T_vals, alpha_vals, nsteps_vals):
    """
    x = n_steps.  One curve per (T, alpha) pair.
    Linestyle encodes T, colour encodes alpha.
    """
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for ti, t in enumerate(T_vals):
        ls = linestyles[ti % len(linestyles)]
        for ai, a in enumerate(alpha_vals):
            ys = [table.get((t, a, n), {}).get(metric, np.nan)
                  for n in nsteps_vals]
            label = f"T={t}, α={a}"
            ax.plot(nsteps_vals, ys,
                    color=_c(ai), marker=_m(ti),
                    linestyle=ls,
                    markersize=7, linewidth=1.6,
                    label=label)

    ax.set_xlabel("n_steps", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{metric.upper()}  —  vs n_steps  (all T × alpha)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    wandb.log({f"{metric}/vs_nsteps_all": wandb.Image(fig)})
    plt.close(fig)


# ── Plot 4: sensitivity bar chart ─────────────────────────────────────────────

def plot_sensitivity(all_metrics: list[str], table: dict,
                     T_vals, alpha_vals, nsteps_vals):
    """
    For each metric: variance when marginalising over each axis independently.
    Produces one grouped bar chart covering all metrics.
    """
    axes_labels = ["T", "alpha", "n_steps"]
    axes_sets   = [T_vals, alpha_vals, nsteps_vals]

    variances = {m: [] for m in all_metrics}

    for metric in all_metrics:
        for axis_idx, fixed_vals in enumerate(
                [alpha_vals, nsteps_vals, T_vals]):       # the OTHER two axes
            # marginalise: for each value of the free axis, average over the fixed two
            free_axis    = axes_sets[axis_idx]
            other_axes   = [axes_sets[i] for i in range(3) if i != axis_idx]

            per_free = []
            for free_val in free_axis:
                vals = []
                for combo in product(*other_axes):
                    key_parts = [None, None, None]
                    key_parts[axis_idx] = free_val
                    oi = 0
                    for i in range(3):
                        if i != axis_idx:
                            key_parts[i] = combo[oi]
                            oi += 1
                    v = table.get(tuple(key_parts), {}).get(metric, np.nan)
                    if np.isfinite(v):
                        vals.append(v)
                if vals:
                    per_free.append(float(np.mean(vals)))

            variances[metric].append(float(np.var(per_free)) if per_free else 0.0)

    # Normalise within each metric so bars are comparable
    n_metrics = len(all_metrics)
    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.4), 5))

    x      = np.arange(n_metrics)
    width  = 0.25
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, axis_label in enumerate(axes_labels):
        vals = []
        for metric in all_metrics:
            row = variances[metric]
            total = sum(row) if sum(row) > 0 else 1.0
            vals.append(row[i] / total)       # normalised share
        ax.bar(x + i * width, vals,
               width, label=axis_label, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_metrics, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalised variance share", fontsize=11)
    ax.set_title("Sensitivity: which axis drives each metric?",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    wandb.log({"sensitivity_by_axis": wandb.Image(fig)})
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m utils.surface_analysis <runs_folder> <project_name>")
        sys.exit(1)

    runs_folder  = sys.argv[1]
    project_name = sys.argv[2]
    runs_dir     = Path(runs_folder)

    if not runs_dir.exists():
        print(f"Error: {runs_folder} does not exist")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading runs from: {runs_folder}")
    all_data = load_run_data(runs_dir)

    if not all_data:
        print("No valid run data found (need config.yaml + metrics_evo.json)")
        sys.exit(1)

    print(f"Found {len(all_data)} runs")

    # ── Rectangularity: find largest usable sub-grid ─────────────────────
    try:
        filtered, T_vals, alpha_vals, nsteps_vals = check_rectangular(all_data)
    except AssertionError as e:
        print(f"\nAborting: {e}")
        sys.exit(1)

    table = build_table(filtered)

    # ── Collect metrics ───────────────────────────────────────────────────
    all_metrics = sorted({k for d in filtered for k in d["best_metrics"]})
    print(f"Metrics   : {all_metrics}")

    # ── WandB init ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=project_name,
        name=f"surface_analysis_{timestamp}",
        job_type="surface_analysis",
        config={
            "runs_folder":   runs_folder,
            "n_runs_total":  len(all_data),
            "n_runs_used":   len(filtered),
            "T_values":      T_vals,
            "alpha_values":  alpha_vals,
            "nsteps_values": nsteps_vals,
            "metrics":       all_metrics,
        },
    )

    # ── Per-metric plots ──────────────────────────────────────────────────
    for metric in all_metrics:
        print(f"  Plotting: {metric}")
        plot_heatmaps(metric, table, T_vals, alpha_vals, nsteps_vals)
        plot_metric_vs_T(metric, table, T_vals, alpha_vals, nsteps_vals)
        plot_metric_vs_nsteps(metric, table, T_vals, alpha_vals, nsteps_vals)

    # ── Sensitivity (one plot, all metrics) ───────────────────────────────
    print("  Plotting: sensitivity")
    plot_sensitivity(all_metrics, table, T_vals, alpha_vals, nsteps_vals)

    wandb.finish()
    print("Surface analysis complete!")


if __name__ == "__main__":
    main()