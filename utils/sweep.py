#!/usr/bin/env python3
"""
Simple Sweep Runner - runs all .yml configs in a folder

Usage:
    python3 sweep_runner.py config_folder/ --device cuda:0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_configs(config_dir):
    """Find all .yml files in directory."""
    config_dir = Path(config_dir)
    configs = sorted([f for f in config_dir.glob("*.yml") if f.name != "manifest.yml"])
    return configs


def run_config(config_path, device):
    """Run training for one config."""
    cmd = ["python", "-m", "ml.train", str(config_path)]
    if device:
        cmd.extend(["--device", device])
    
    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run all configs in a folder')
    parser.add_argument('config_dir', help='Directory containing config files')
    parser.add_argument('--device', default=None, help='Device (e.g., cuda:0)')
    args = parser.parse_args()
    
    # Find configs
    configs = find_configs(args.config_dir)
    if not configs:
        print(f"No config files found in: {args.config_dir}")
        sys.exit(1)
    
    print(f"Found {len(configs)} configs")
    print(f"Device: {args.device or 'auto-detect'}")
    
    # Run each config
    failed = []
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        returncode = run_config(config_path, args.device)
        
        if returncode != 0:
            print(f"FAILED: {config_path.name}")
            failed.append(config_path.name)
        else:
            print(f"SUCCESS: {config_path.name}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Completed: {len(configs) - len(failed)}/{len(configs)}")
    if failed:
        print(f"Failed configs:")
        for name in failed:
            print(f"  - {name}")
    print(f"{'='*80}")
    
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()