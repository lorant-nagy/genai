#!/usr/bin/env python3
"""
Profiler Analysis Script - Analyze training runs and upload plots to WandB

This script is called by profiler.py with runs folder and project name as arguments.
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
    """Load all run configs and metrics from a runs directory."""
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
        
        power = config['corruption']['process_params']['alpha']
        T = config['corruption']['process_params']['T']
        
        best_metrics = {}
        for key, values in metrics.items():
            if key in ['epochs', 'nan'] or not values:
                continue
            finite = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                K = 5
                tail = finite[-K:]                 
                best_metrics[key] = float(np.median(tail))
        
        all_data.append({
            'power': power, 
            'T': T, 
            'best_metrics': best_metrics, 
            'run_name': run_dir.name
        })
    
    return all_data


def validate_data(all_data):
    """Validate that data has consistent structure."""
    power_T_pairs = [(d['power'], d['T']) for d in all_data]
    assert len(power_T_pairs) == len(set(power_T_pairs)), \
        "Duplicate (power, T) pairs found"
    
    power_groups = defaultdict(set)
    for d in all_data:
        power_groups[d['power']].add(d['T'])
    T_counts = [len(t_set) for t_set in power_groups.values()]
    assert len(set(T_counts)) == 1, \
        "Inconsistent number of T values across powers"


def plot_metric(metric_key, data_by_power, project_name):
    """Create and log a single metric plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (power, data) in enumerate(sorted(data_by_power.items())):
        data_sorted = sorted(data, key=lambda x: x[0])
        T_vals = [d[0] for d in data_sorted]
        metric_vals = [d[1] for d in data_sorted]
        ax.plot(T_vals, metric_vals, 
                marker=markers[i % len(markers)], 
                markersize=8,
                linewidth=2, 
                label=f'α = {power}', 
                color=colors[i % len(colors)])
    
    ax.set_xlabel('T (Terminal Time)', fontsize=12)
    ax.set_ylabel(f'{metric_key.upper()}', fontsize=12)
    ax.set_title(f'{metric_key.upper()} vs T', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    wandb.log({f'{metric_key}_vs_T': wandb.Image(fig)})
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print("Usage: python profiler_analysis.py <runs_folder> <project_name>")
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
    
    # Validate
    validate_data(all_data)
    
    # Initialize WandB
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project=project_name, 
        name=f'profiling_results_{timestamp}', 
        job_type='analysis'
    )
    
    # Collect all metrics
    all_metrics = set()
    for d in all_data:
        all_metrics.update(d['best_metrics'].keys())
    
    # Plot each metric
    for metric_key in sorted(all_metrics):
        data_by_power = defaultdict(list)
        for d in all_data:
            if metric_key in d['best_metrics']:
                data_by_power[d['power']].append((d['T'], d['best_metrics'][metric_key]))
        
        if data_by_power:
            plot_metric(metric_key, data_by_power, project_name)
    
    wandb.finish()
    print('Analysis complete!')


if __name__ == "__main__":
    main()