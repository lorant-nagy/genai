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


def find_configs(config_dir):
    """Find all .yml files in directory."""
    config_dir = Path(config_dir)
    configs = sorted([f for f in config_dir.glob("*.yml") if f.name != "manifest.yml"])
    return configs


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
    parser.add_argument('--device', default=None, help='Device (e.g., cuda:0)')
    parser.add_argument('--compose-file', default='compose.yml', help='Docker compose file')
    args = parser.parse_args()
    
    # Find configs
    configs = find_configs(args.config_dir)
    if not configs:
        print(f"No config files found in: {args.config_dir}")
        sys.exit(1)

    random.shuffle(configs)
    
    print(f"Found {len(configs)} configs")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Each will run in a fresh Docker container")
    
    start_time = datetime.now()
    
    # Run each config
    failed = []
    succeeded = []
    
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        returncode = run_config_in_docker(config_path, args.device, args.compose_file)
        
        if returncode != 0:
            print(f"FAILED: {config_path.name}")
            failed.append(config_path.name)
        else:
            print(f"SUCCESS: {config_path.name}")
            succeeded.append(config_path.name)
    
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
    
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()