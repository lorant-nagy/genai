#!/usr/bin/env python3
"""
Epoch Analysis Script - Plot metrics evolution through epochs and upload to WandB

This script is called by profiler.py with --anal_through_epochs flag.
It loads metrics_evo.json from all runs and plots how metrics evolve over epochs.
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


def load_run_data(runs_dir):
    """Load all run configs and full metrics evolution from a runs directory."""
    all_data = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        config_path = run_dir / 'config.yaml'
        metrics_path = run_dir / 'metrics_evo.json'
        
        if not config_path.exists() or not metrics_path.exists():
            continue
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        # Extract key parameters
        power = config['corruption']['process_params']['alpha']
        T = config['corruption']['process_params']['T']
        
        # Store full evolution
        all_data.append({
            'power': power, 
            'T': T, 
            'metrics_evo': metrics,
            'run_name': run_dir.name
        })
    
    return all_data


def get_metric_keys(all_data):
    """Get all metric keys that have numeric data (excluding epochs and nan)."""
    all_metric_keys = set()
    for d in all_data:
        metrics = d['metrics_evo']
        for key, values in metrics.items():
            if key in ['epochs', 'nan']:
                continue
            # Check if this metric has numeric values
            if values and any(isinstance(v, (int, float)) for v in values):
                all_metric_keys.add(key)
    return sorted(all_metric_keys)


def plot_metric_evolution_by_power(metric_key, all_data, project_name):
    """
    Plot metric evolution through epochs, grouped by power (alpha).
    One line per (power, T) combination.
    """
    # Group data by power
    data_by_power = defaultdict(list)
    
    for d in all_data:
        metrics = d['metrics_evo']
        if metric_key not in metrics or not metrics[metric_key]:
            continue
        
        epochs = metrics.get('epochs', list(range(1, len(metrics[metric_key]) + 1)))
        metric_values = metrics[metric_key]
        
        # Filter out non-finite values
        clean_data = [(e, v) for e, v in zip(epochs, metric_values) 
                      if isinstance(v, (int, float)) and np.isfinite(v)]
        
        if clean_data:
            epochs_clean, values_clean = zip(*clean_data)
            data_by_power[d['power']].append({
                'T': d['T'],
                'epochs': epochs_clean,
                'values': values_clean,
                'run_name': d['run_name']
            })
    
    if not data_by_power:
        print(f"No valid data for {metric_key}, skipping")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot each power group
    for i, (power, runs) in enumerate(sorted(data_by_power.items())):
        # Sort by T within power group
        runs_sorted = sorted(runs, key=lambda x: x['T'])
        
        for j, run_data in enumerate(runs_sorted):
            label = f'α={power}, T={run_data["T"]}'
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            
            ax.plot(run_data['epochs'], run_data['values'],
                   label=label,
                   color=color,
                   linestyle=linestyle,
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_key.upper(), fontsize=12)
    ax.set_title(f'{metric_key.upper()} Evolution Through Epochs', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    wandb.log({f'{metric_key}_vs_epochs': wandb.Image(fig)})
    plt.close(fig)


def plot_metric_evolution_by_T(metric_key, all_data, project_name):
    """
    Plot metric evolution through epochs, grouped by T.
    One line per (power, T) combination, but organized differently.
    """
    # Group data by T
    data_by_T = defaultdict(list)
    
    for d in all_data:
        metrics = d['metrics_evo']
        if metric_key not in metrics or not metrics[metric_key]:
            continue
        
        epochs = metrics.get('epochs', list(range(1, len(metrics[metric_key]) + 1)))
        metric_values = metrics[metric_key]
        
        # Filter out non-finite values
        clean_data = [(e, v) for e, v in zip(epochs, metric_values) 
                      if isinstance(v, (int, float)) and np.isfinite(v)]
        
        if clean_data:
            epochs_clean, values_clean = zip(*clean_data)
            data_by_T[d['T']].append({
                'power': d['power'],
                'epochs': epochs_clean,
                'values': values_clean,
                'run_name': d['run_name']
            })
    
    if not data_by_T:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot each T group
    for i, (T, runs) in enumerate(sorted(data_by_T.items())):
        # Sort by power within T group
        runs_sorted = sorted(runs, key=lambda x: x['power'])
        
        for j, run_data in enumerate(runs_sorted):
            label = f'T={T}, α={run_data["power"]}'
            color = colors[i % len(colors)]
            marker = markers[j % len(markers)]
            
            # Plot with markers every N epochs for clarity
            marker_every = max(1, len(run_data['epochs']) // 10)
            
            ax.plot(run_data['epochs'], run_data['values'],
                   label=label,
                   color=color,
                   marker=marker,
                   markersize=6,
                   markevery=marker_every,
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_key.upper(), fontsize=12)
    ax.set_title(f'{metric_key.upper()} Evolution Through Epochs (Grouped by T)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    wandb.log({f'{metric_key}_vs_epochs_by_T': wandb.Image(fig)})
    plt.close(fig)


def plot_final_metrics_comparison(all_data, metric_keys):
    """
    Create a summary plot showing final metric values for all runs.
    Useful for quick comparison of convergence quality.
    """
    # Collect final values (last epoch)
    final_data = []
    
    for d in all_data:
        metrics = d['metrics_evo']
        final_metrics = {}
        
        for key in metric_keys:
            if key in metrics and metrics[key]:
                values = [v for v in metrics[key] 
                         if isinstance(v, (int, float)) and np.isfinite(v)]
                if values:
                    # Take median of last 5 epochs for stability
                    K = min(5, len(values))
                    final_metrics[key] = np.median(values[-K:])
        
        if final_metrics:
            final_data.append({
                'power': d['power'],
                'T': d['T'],
                'run_name': d['run_name'],
                'metrics': final_metrics
            })
    
    if not final_data:
        return
    
    # Create figure with subplots for each metric
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric_key in zip(axes, metric_keys):
        # Group by power
        data_by_power = defaultdict(list)
        for d in final_data:
            if metric_key in d['metrics']:
                data_by_power[d['power']].append((d['T'], d['metrics'][metric_key]))
        
        # Plot
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
        for i, (power, data) in enumerate(sorted(data_by_power.items())):
            data_sorted = sorted(data, key=lambda x: x[0])
            T_vals = [x[0] for x in data_sorted]
            metric_vals = [x[1] for x in data_sorted]
            
            ax.plot(T_vals, metric_vals,
                   marker='o',
                   markersize=8,
                   linewidth=2,
                   label=f'α={power}',
                   color=colors[i % len(colors)])
        
        ax.set_xlabel('T', fontsize=11)
        ax.set_ylabel(f'Final {metric_key.upper()}', fontsize=11)
        ax.set_title(f'Final {metric_key.upper()} vs T', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({'final_metrics_summary': wandb.Image(fig)})
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print("Usage: python epoch_analysis.py <runs_folder> <project_name>")
        sys.exit(1)
    
    runs_folder = sys.argv[1]
    project_name = sys.argv[2]
    
    print(f"Loading runs from: {runs_folder}")
    runs_dir = Path(runs_folder)
    
    if not runs_dir.exists():
        print(f"Error: Directory {runs_folder} does not exist")
        sys.exit(1)
    
    # Load data
    all_data = load_run_data(runs_dir)
    
    if not all_data:
        print("No valid run data found")
        sys.exit(1)
    
    print(f"Found {len(all_data)} runs")
    
    # Get all available metrics
    metric_keys = get_metric_keys(all_data)
    print(f"Found metrics: {', '.join(metric_keys)}")
    
    if not metric_keys:
        print("No metrics to plot")
        sys.exit(1)
    
    # Initialize WandB
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project=project_name, 
        name=f'epoch_analysis_{timestamp}', 
        job_type='epoch_analysis'
    )
    
    # Create plots for each metric
    for metric_key in metric_keys:
        print(f"Plotting {metric_key}...")
        plot_metric_evolution_by_power(metric_key, all_data, project_name)
        plot_metric_evolution_by_T(metric_key, all_data, project_name)
    
    # Create final summary comparison
    print("Creating final metrics summary...")
    plot_final_metrics_comparison(all_data, metric_keys)
    
    wandb.finish()
    print('Epoch analysis complete!')


if __name__ == "__main__":
    main()