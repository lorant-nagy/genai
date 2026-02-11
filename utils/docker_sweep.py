#!/usr/bin/env python3
"""
Docker Sweep Runner - runs each config in a fresh container

Usage:
    python3 docker_sweep_runner.py config_folder/ --device cuda:0
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
import yaml
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def find_configs(config_dir):
    """Find all .yml files in directory."""
    config_dir = Path(config_dir)
    configs = sorted([f for f in config_dir.glob("*.yml") if f.name != "manifest.yml"])
    return configs


def infer_log_path(config_dir):
    """Infer log file path from the first config file."""
    configs = find_configs(config_dir)
    if not configs:
        return None
    
    # Load first config to extract env settings
    with open(configs[0], 'r') as f:
        config = yaml.safe_load(f)
    
    wandb_project = config.get('wandb', {}).get('project', 'default_project')
    
    log_path = Path('run_logs') / wandb_project / 'global_logs.txt'
    return str(log_path)


def format_timedelta(td):
    """Format a timedelta as HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_eta_info(completed, total, avg_time_per_config, elapsed_time):
    """Format ETA information for display."""
    remaining = total - completed
    if completed == 0:
        return "ETA: Calculating..."
    
    estimated_remaining = avg_time_per_config * remaining
    eta_time = datetime.now() + estimated_remaining
    
    return (f"Progress: {completed}/{total} | "
            f"Avg time/config: {format_timedelta(avg_time_per_config)} | "
            f"Elapsed: {format_timedelta(elapsed_time)} | "
            f"ETA: {eta_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(~{format_timedelta(estimated_remaining)} remaining)")


def run_config_in_docker(config_path, device, compose_file="compose.yml"):
    """Run training in a fresh Docker container."""
    # Docker compose run command
    cmd = [
        "docker", "compose",
        "-f", compose_file,
        "run",
        "--rm",  # Remove container after completion
        "train",
        "python", "-m", "ml.train",
        str(config_path)
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    config_start = time.time()
    result = subprocess.run(cmd)
    config_duration = time.time() - config_start
    
    return result.returncode, config_duration

def main():
    parser = argparse.ArgumentParser(description='Run all configs in separate Docker containers')
    parser.add_argument('config_dir', help='Directory containing config files')
    parser.add_argument('--log-file', default=None, help='Path to log file (auto-inferred if not provided)')
    parser.add_argument('--device', default=None, help='Device (e.g., cuda:0)')
    parser.add_argument('--compose-file', default='compose.yml', help='Docker compose file')
    args = parser.parse_args()
    
    # Find configs
    configs = find_configs(args.config_dir)
    if not configs:
        print(f"No config files found in: {args.config_dir}")
        sys.exit(1)

    # Infer log file path if not provided
    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        inferred_log = infer_log_path(args.config_dir)
        if not inferred_log:
            print("Could not infer log file path and none provided")
            sys.exit(1)
        log_file_path = Path(inferred_log)
        print(f"Inferred log file: {log_file_path}")

    random.shuffle(configs)
    
    print(f"Found {len(configs)} configs")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Each will run in a fresh Docker container")
    
    start_time = datetime.now()
    sweep_start = time.time()
    
    # Run each config
    failed = []
    succeeded = []
    config_durations = []
    
    # Create log file with header for this sweep
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"SWEEP STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {args.device} | Configs: {len(configs)}\n")
        f.write(f"{'='*80}\n")
    
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        
        # Calculate and display ETA
        if config_durations:
            avg_time = timedelta(seconds=sum(config_durations) / len(config_durations))
            elapsed = timedelta(seconds=time.time() - sweep_start)
            eta_info = format_eta_info(i, len(configs), avg_time, elapsed)
            print(eta_info)
            print()
        
        returncode, duration = run_config_in_docker(config_path, args.device, args.compose_file)
        config_durations.append(duration)
        
        # Log result immediately
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration_str = format_timedelta(timedelta(seconds=duration))
        
        if returncode != 0:
            print(f"FAILED: {config_path.name} (took {duration_str})")
            failed.append(config_path.name)
            with open(log_file_path, "a") as f:
                f.write(f"[{timestamp}] FAILED: {config_path.name} (duration: {duration_str})\n")
        else:
            print(f"SUCCESS: {config_path.name} (took {duration_str})")
            succeeded.append(config_path.name)
            with open(log_file_path, "a") as f:
                f.write(f"[{timestamp}] SUCCESS: {config_path.name} (duration: {duration_str})\n")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Calculate average time per config
    if config_durations:
        avg_duration = timedelta(seconds=sum(config_durations) / len(config_durations))
        min_duration = timedelta(seconds=min(config_durations))
        max_duration = timedelta(seconds=max(config_durations))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration: {format_timedelta(total_duration)}")
    print(f"Total configs: {len(configs)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    
    if config_durations:
        print(f"\nTiming Statistics:")
        print(f"  Average time per config: {format_timedelta(avg_duration)}")
        print(f"  Fastest config: {format_timedelta(min_duration)}")
        print(f"  Slowest config: {format_timedelta(max_duration)}")
    
    if succeeded:
        print(f"\nSuccessful configs:")
        for name in succeeded:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed configs:")
        for name in failed:
            print(f"  ✗ {name}")
    
    print(f"{'='*80}")
    
    # Write summary to log
    with open(log_file_path, "a") as f:
        f.write(f"SWEEP ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | Duration: {total_duration} | Success: {len(succeeded)}/{len(configs)}\n")
        if config_durations:
            f.write(f"Average time per config: {format_timedelta(avg_duration)} | Min: {format_timedelta(min_duration)} | Max: {format_timedelta(max_duration)}\n")
    
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()