#!/usr/bin/env python3
"""
Docker Profile Plotter - Analyze training runs and upload plots to WandB

USAGE:
    python profiler.py <runs_folder> <project_name>
    python profiler.py <runs_folder> <project_name> --anal_through_epochs

EXAMPLE:
    python profiler.py run_logs/profiling/runs profiling
    python profiler.py run_logs/profiling/runs profiling --anal_through_epochs
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python profiler.py <runs_folder> <project_name> [--anal_through_epochs]")
        print("Example: python profiler.py run_logs/profiling/runs profiling")
        print("Example: python profiler.py run_logs/profiling/runs profiling --anal_through_epochs")
        sys.exit(1)
    
    runs_folder = sys.argv[1]
    project_name = sys.argv[2]
    
    # Check for epoch analysis flag
    if len(sys.argv) > 3 and sys.argv[3] == '--anal_through_epochs':
        analysis_module = "utils.epoch_analysis"
        print(f"Running epoch-wise analysis: {runs_folder} -> WandB project '{project_name}'")
    else:
        analysis_module = "utils.profiler_analysis"
        print(f"Running analysis: {runs_folder} -> WandB project '{project_name}'")
    
    # Simple, clean command - just run the analysis script with arguments
    cmd = [
        "docker", "compose",
        "run", "--rm",
        "train",
        "python", "-m", analysis_module,
        runs_folder,
        project_name
    ]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()