#!/usr/bin/env python3
"""
Experiment - Analyze training runs and upload plots to WandB

Dispatches to the appropriate analysis script based on the flag passed.

USAGE:
    python -m utils.analyse <runs_folder> <project_name> --sweep-summary
    python -m utils.analyse <runs_folder> <project_name> --sweep-summary-partial
    python -m utils.analyse <runs_folder> <project_name> --nsteps
    python -m utils.analyse <runs_folder> <project_name> --surface

ANALYSIS MODES:
    --sweep-summary          best metric vs T, one curve per alpha (complete sweeps)
    --sweep-summary-partial  like --sweep-summary but tolerates missing runs
    --nsteps                 best metric vs n_steps, one curve per T
    --surface                full T × alpha × n_steps surface analysis

EXAMPLES:
    python -m utils.analyse run_logs/profiling/runs profiling --sweep-summary
    python -m utils.analyse run_logs/profiling/runs profiling --sweep-summary-partial
    python -m utils.analyse run_logs/nsteps/runs nsteps --nsteps
    python -m utils.analyse run_logs/surface_response/runs surface_response --surface
"""

import subprocess
import sys

MODES = {
    '--nsteps':             ('utils.nsteps_analysis',    'n-steps analysis'),
    '--sweep-summary':      ('utils.sweep_summary',      'sweep summary'),
    '--sweep-summary-partial': ('utils.sweep_summary_partial', 'partial sweep summary'),
    '--surface':            ('utils.surface_analysis',   'T × alpha × n_steps surface analysis'),
}

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    runs_folder  = sys.argv[1]
    project_name = sys.argv[2]
    flags        = set(sys.argv[3:])

    # Pick mode (first recognised flag wins)
    module, label = None, None
    for flag, (mod, lbl) in MODES.items():
        if flag in flags:
            module, label = mod, lbl
            break

    if module is None:
        print("No valid analysis mode specified. Please use one of the following flags:")
        for flag in MODES.keys():
            print(f"  {flag}")
        sys.exit(1)

    print(f"Running {label}: {runs_folder} -> WandB project '{project_name}'")

    cmd = [
        "docker", "compose",
        "run", "--rm",
        "train",
        "python", "-m", module,
        runs_folder,
        project_name,
    ]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()