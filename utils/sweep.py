#!/usr/bin/env python3
"""
Simple Sweep Runner - runs all .yml configs in a folder

Usage:
    python3 sweep_runner.py config_folder/ --device 0
    python3 sweep_runner.py config_folder/ --device cuda:0
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_configs(config_dir):
    """Find all .yml files in directory."""
    config_dir = Path(config_dir)
    return sorted([f for f in config_dir.glob("*.yml") if f.name != "manifest.yml"])


def parse_device(s):
    """
    Our ml/train.py expects an int CUDA device index.
    Accepts: 0, "0", "cuda:0", "cuda" -> 0
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    if s == "cuda":
        return 0
    if s.startswith("cuda:"):
        return int(s.split(":", 1)[1])
    return int(s)


def run_config(config_path, device_idx):
    """Run training for one config."""
    cmd = [sys.executable, "-m", "ml.train", str(config_path)]
    if device_idx is not None:
        cmd.extend(["--device", str(device_idx)])

    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all configs in a folder")
    parser.add_argument("config_dir", help="Directory containing config files")
    parser.add_argument("--device", default=None, type=parse_device,
                        help="CUDA device index (e.g., 0 or cuda:0). If omitted, train.py default applies.")
    args = parser.parse_args()

    configs = find_configs(args.config_dir)
    if not configs:
        print(f"No config files found in: {args.config_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} configs")
    if args.device is None:
        print("Device: train.py default")
    else:
        print(f"Device: cuda:{args.device} (passed as --device {args.device})")

    failed = []
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        rc = run_config(config_path, args.device)
        if rc != 0:
            print(f"FAILED: {config_path.name}")
            failed.append(config_path.name)
        else:
            print(f"SUCCESS: {config_path.name}")

    print(f"\n{'='*80}")
    print(f"Completed: {len(configs) - len(failed)}/{len(configs)}")
    if failed:
        print("Failed configs:")
        for name in failed:
            print(f"  - {name}")
    print(f"{'='*80}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
