#!/usr/bin/env python3
"""
Docker Sweep Runner - runs each config in a fresh container

Usage:
    python3 docker_sweep_runner.py config_folder/ --device cuda:0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import random
import yaml

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
    
    results_subdir = config.get('env', {}).get('results_subdir', 'default')
    
    log_path = Path('/data/lorantnagy/storage/genai/runs') / results_subdir / 'global_log.txt'
    return str(log_path)


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
    
    result = subprocess.run(cmd)
    return result.returncode


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
    
    # Run each config
    failed = []
    succeeded = []
    
    # Create log file with header for this sweep
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"SWEEP STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {args.device} | Configs: {len(configs)}\n")
        f.write(f"{'='*80}\n")
    
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        returncode = run_config_in_docker(config_path, args.device, args.compose_file)
        
        # Log result immediately
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if returncode != 0:
            print(f"FAILED: {config_path.name}")
            failed.append(config_path.name)
            with open(log_file_path, "a") as f:
                f.write(f"[{timestamp}] FAILED: {config_path.name}\n")
        else:
            print(f"SUCCESS: {config_path.name}")
            succeeded.append(config_path.name)
            with open(log_file_path, "a") as f:
                f.write(f"[{timestamp}] SUCCESS: {config_path.name}\n")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Duration: {duration}")
    print(f"Total configs: {len(configs)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    
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
        f.write(f"SWEEP ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | Duration: {duration} | Success: {len(succeeded)}/{len(configs)}\n")
    
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()