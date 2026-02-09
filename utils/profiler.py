#!/usr/bin/env python3
"""
Docker Profile Plotter - Analyze training runs and upload plots to WandB

USAGE:
    python profiler.py <runs_folder> <project_name>

EXAMPLE:
    python profiler.py run_logs/profiling/runs profiling
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python profiler.py <runs_folder> <project_name>")
        print("Example: python profiler.py run_logs/profiling/runs profiling")
        sys.exit(1)
    
    runs_folder = sys.argv[1]
    project_name = sys.argv[2]
    
    # Simple, clean command - just run the analysis script with arguments
    cmd = [
        "docker", "compose",
        "run", "--rm",
        "train",
        "python", "-m", "utils.profiler_analysis",
        runs_folder,
        project_name
    ]
    
    print(f"Running analysis: {runs_folder} -> WandB project '{project_name}'")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()