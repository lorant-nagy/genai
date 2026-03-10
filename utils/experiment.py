#!/usr/bin/env python3
"""
Experiment - Analyze training runs and upload plots to WandB

Dispatches to the appropriate analysis script based on the flag passed.

USAGE:
    python experiment.py <runs_folder> <project_name>
    python experiment.py <runs_folder> <project_name> --epochs
    python experiment.py <runs_folder> <project_name> --partial
    python experiment.py <runs_folder> <project_name> --nsteps

ANALYSIS MODES:
    (none)     profiler_analysis          - best metric vs T, one curve per alpha
    --epochs   epoch_analysis             - metric evolution through training epochs
    --partial  profiler_analysis_partial  - like default but tolerates incomplete sweeps
    --nsteps   nsteps_analysis            - best metric vs n_steps, one curve per T

EXAMPLES:
    python experiment.py run_logs/profiling/runs profiling
    python experiment.py run_logs/nsteps/runs nsteps --nsteps
    python experiment.py run_logs/profiling/runs profiling --epochs
    python experiment.py run_logs/profiling/runs profiling --partial
"""

import subprocess
import sys

MODES = {
    '--nsteps':  ('utils.nsteps_analysis',             'n-steps analysis'),
    '--epochs':  ('utils.epoch_analysis',              'epoch-wise analysis'),
    '--partial': ('utils.profiler_analysis_partial',   'partial-run analysis'),
}
DEFAULT = ('utils.profiler_analysis', 'standard analysis')

# Keep the old flag name working as an alias
ALIASES = {'--anal_through_epochs': '--epochs'}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    runs_folder  = sys.argv[1]
    project_name = sys.argv[2]
    flags        = {ALIASES.get(f, f) for f in sys.argv[3:]}

    # Pick mode (first recognised flag wins)
    module, label = DEFAULT
    for flag, (mod, lbl) in MODES.items():
        if flag in flags:
            module, label = mod, lbl
            break

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