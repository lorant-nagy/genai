#!/usr/bin/env python3
"""
Docker Sweep Runner - runs each config in a fresh container

Usage:
    # Normal sweep
    python3 utils/docker_sweep.py config_folder/ --device 2

    # Retry from a failed_configs.txt written by a previous sweep
    python3 utils/docker_sweep.py config_folder/ --device 2 \\
        --retry-failed /path/to/runs/my_project/failed_configs_20260310_141549.txt

Each container is pinned to the requested physical GPU via CUDA_VISIBLE_DEVICES,
which remaps it to cuda:0 inside the container so train.py always uses --device 0.

Code snapshot
-------------
At sweep start the repo is copied to <log_dir>/snapshot/ so that branch switches
or edits on the host during a long sweep do not affect running containers.
Pass --no-snapshot to disable and use the live workspace.
"""

import argparse
import subprocess
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
import yaml
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Config discovery ──────────────────────────────────────────────────────────

def find_configs(config_dir):
    """Find all .yml files in directory."""
    config_dir = Path(config_dir)
    return sorted([f for f in config_dir.glob("*.yml") if f.name != "manifest.yml"])


def filter_configs_by_names(all_configs, names):
    """Return only configs whose filename is in the given set of names."""
    return [c for c in all_configs if c.name in names]


# ── Failed-list helpers ───────────────────────────────────────────────────────

def write_failed_list(failed_names, log_file_path):
    """
    Write a failed_configs_<timestamp>.txt next to the global log.
    Returns the path written, or None if there were no failures.
    """
    if not failed_names:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_list_path = log_file_path.parent / f"failed_configs_{timestamp}.txt"
    with open(failed_list_path, "w") as f:
        for name in sorted(failed_names):
            f.write(name + "\n")
    return failed_list_path


def read_failed_list(path):
    """Read a failed_configs_*.txt and return a set of config filenames."""
    path = Path(path)
    names = set()
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


# ── Log path inference ────────────────────────────────────────────────────────

def infer_log_path(config_dir, retry=False):
    """Infer log file path from the first config file."""
    configs = find_configs(config_dir)
    if not configs:
        return None
    with open(configs[0], "r") as f:
        config = yaml.safe_load(f)
    wandb_project = config.get("wandb", {}).get("project", "default_project")
    project_dir   = os.path.join("/data/lorantnagy/storage/genai/runs", wandb_project)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix        = "_retry" if retry else ""
    return os.path.join(project_dir, f"global_logs_{timestamp}{suffix}.txt")


# ── Code snapshot ─────────────────────────────────────────────────────────────

def get_git_info(repo_root):
    """Return (commit_hash, branch) or ('unknown', 'unknown') on failure."""
    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root)] + list(args),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
    return _git("rev-parse", "--short", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD")


def snapshot_repo(repo_root, dest_dir):
    """
    Copy the repo into dest_dir/snapshot/, respecting .dockerignore patterns.
    Returns the Path to the snapshot directory.
    """
    repo_root = Path(repo_root).resolve()
    snapshot  = Path(dest_dir) / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)

    SKIP_DIRS = {".git", "__pycache__", "wandb", "runs", "checkpoints",
                 "data", "outputs", ".venv", ".ipynb_checkpoints", ".vscode"}
    SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".pt", ".pth", ".ckpt",
                 ".npy", ".npz", ".DS_Store"}

    def _ignore(src, names):
        ignored = set()
        for name in names:
            if name in SKIP_DIRS:
                ignored.add(name)
            elif Path(name).suffix in SKIP_EXTS:
                ignored.add(name)
        return ignored

    shutil.copytree(repo_root, snapshot, ignore=_ignore, dirs_exist_ok=True)
    return snapshot


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_eta_info(completed, total, avg_time_per_config, elapsed_time):
    if completed == 0:
        return "ETA: Calculating..."
    remaining           = total - completed
    estimated_remaining = avg_time_per_config * remaining
    eta_time            = datetime.now() + estimated_remaining
    return (
        f"Progress: {completed}/{total} | "
        f"Avg time/config: {format_timedelta(avg_time_per_config)} | "
        f"Elapsed: {format_timedelta(elapsed_time)} | "
        f"ETA: {eta_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(~{format_timedelta(estimated_remaining)} remaining)"
    )


def _parse_device_index(device):
    if device is None:
        return None
    s = str(device).strip()
    if s.lower().startswith("cuda:"):
        return s.split(":")[-1]
    return s


# ── Docker runner ─────────────────────────────────────────────────────────────

def run_config_in_docker(config_path, device, compose_file, snapshot_dir):
    """
    Run training in a fresh Docker container.
    If snapshot_dir is set, overrides the workspace volume with the snapshot.
    """
    device_idx = _parse_device_index(device)

    cmd = ["docker", "compose", "-f", compose_file, "run", "--rm"]

    if device_idx is not None:
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={device_idx}"]

    if snapshot_dir is not None:
        cmd += ["-v", f"{snapshot_dir}:/workspace"]

    cmd += ["train", "python", "-m", "ml.train", str(config_path), "--device", "0"]

    print(f"\n{'='*80}")
    print(f"Running : {config_path.name}")
    if device_idx is not None:
        print(f"GPU     : physical device {device_idx}  ->  cuda:0 inside container")
    if snapshot_dir is not None:
        print(f"Code    : {snapshot_dir}  (snapshot)")
    print(f"Command : {' '.join(cmd)}")
    print(f"{'='*80}\n")

    config_start = time.time()
    result       = subprocess.run(cmd)
    duration     = time.time() - config_start

    return result.returncode, duration


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all configs in separate Docker containers"
    )
    parser.add_argument("config_dir",
                        help="Directory containing config .yml files")
    parser.add_argument("--log-file",      default=None,
                        help="Path to log file (auto-inferred if not provided)")
    parser.add_argument("--device",        default=None,
                        help="Physical GPU index (e.g. 0, 1, 2)")
    parser.add_argument("--compose-file",  default="compose.yml")
    parser.add_argument("--retry-failed",  default=None, metavar="FAILED_LIST",
                        help="Path to a failed_configs_*.txt from a previous sweep. "
                             "Only those configs will be run. "
                             "A new global log is created with a _retry suffix.")
    parser.add_argument("--no-snapshot",   action="store_true",
                        help="Skip code snapshot and use the live workspace.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # ── Collect configs ───────────────────────────────────────────────────────
    all_configs = find_configs(args.config_dir)
    if not all_configs:
        print(f"No config files found in: {args.config_dir}")
        sys.exit(1)

    # ── Filter to failed list if retrying ─────────────────────────────────────
    is_retry = args.retry_failed is not None
    if is_retry:
        failed_list_path = Path(args.retry_failed)
        if not failed_list_path.exists():
            print(f"Failed list not found: {failed_list_path}")
            sys.exit(1)
        failed_names = read_failed_list(failed_list_path)
        configs      = filter_configs_by_names(all_configs, failed_names)

        missing = failed_names - {c.name for c in configs}
        print(f"\nRetry mode  —  failed list: {failed_list_path}")
        print(f"  configs in list : {len(failed_names)}")
        print(f"  found in dir    : {len(configs)}")
        if missing:
            print(f"  not found ({len(missing)}) :")
            for m in sorted(missing):
                print(f"    - {m}")
        if not configs:
            print("\nNo matching configs found. Exiting.")
            sys.exit(1)
    else:
        configs = all_configs

    # ── Log file path ─────────────────────────────────────────────────────────
    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        inferred = infer_log_path(args.config_dir, retry=is_retry)
        if not inferred:
            print("Could not infer log file path and none provided")
            sys.exit(1)
        log_file_path = Path(inferred)
        print(f"Inferred log file: {log_file_path}")

    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Code snapshot ─────────────────────────────────────────────────────────
    snapshot_dir = None
    commit, branch = get_git_info(repo_root)

    if args.no_snapshot:
        print(f"\nSnapshot : disabled (live workspace)  [{branch} @ {commit}]")
    else:
        print(f"\nSnapshotting repo  [{branch} @ {commit}] ...", end=" ", flush=True)
        snapshot_dir = snapshot_repo(repo_root, log_file_path.parent)
        print(f"done.\n  -> {snapshot_dir}")

    random.shuffle(configs)

    print(f"\nConfigs to run : {len(configs)}")
    print(f"Device         : {args.device or 'auto-detect'}")

    start_time  = datetime.now()
    sweep_start = time.time()

    failed           = []
    succeeded        = []
    config_durations = []

    # ── Log header ────────────────────────────────────────────────────────────
    retry_note    = f" | Retry of: {args.retry_failed}" if is_retry else ""
    snapshot_note = f" | Snapshot: {snapshot_dir}" if snapshot_dir else " | Snapshot: disabled"
    with open(log_file_path, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(
            f"SWEEP STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Device: {args.device} | Configs: {len(configs)}{retry_note}\n"
        )
        f.write(f"Git: {branch} @ {commit}{snapshot_note}\n")
        f.write(f"{'='*80}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, config_path in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")

        if config_durations:
            avg_time = timedelta(seconds=sum(config_durations) / len(config_durations))
            elapsed  = timedelta(seconds=time.time() - sweep_start)
            print(format_eta_info(i, len(configs), avg_time, elapsed))
            print()

        returncode, duration = run_config_in_docker(
            config_path, args.device, args.compose_file, snapshot_dir
        )
        config_durations.append(duration)

        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    # ── Summary ───────────────────────────────────────────────────────────────
    end_time       = datetime.now()
    total_duration = end_time - start_time

    avg_duration = min_duration = max_duration = None
    if config_durations:
        avg_duration = timedelta(seconds=sum(config_durations) / len(config_durations))
        min_duration = timedelta(seconds=min(config_durations))
        max_duration = timedelta(seconds=max(config_durations))

    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration : {format_timedelta(total_duration)}")
    print(f"Total configs  : {len(configs)}")
    print(f"Succeeded      : {len(succeeded)}")
    print(f"Failed         : {len(failed)}")

    if avg_duration is not None:
        print(f"\nTiming Statistics:")
        print(f"  Average time per config : {format_timedelta(avg_duration)}")
        print(f"  Fastest config          : {format_timedelta(min_duration)}")
        print(f"  Slowest config          : {format_timedelta(max_duration)}")

    if succeeded:
        print(f"\nSuccessful configs:")
        for name in succeeded:
            print(f"  v {name}")

    if failed:
        print(f"\nFailed configs:")
        for name in failed:
            print(f"  x {name}")

    print(f"{'='*80}")

    with open(log_file_path, "a") as f:
        f.write(
            f"SWEEP ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Duration: {total_duration} | Success: {len(succeeded)}/{len(configs)}\n"
        )
        if avg_duration is not None:
            f.write(
                f"Average time per config: {format_timedelta(avg_duration)} | "
                f"Min: {format_timedelta(min_duration)} | Max: {format_timedelta(max_duration)}\n"
            )

    # ── Write failed list ─────────────────────────────────────────────────────
    if failed:
        failed_list_path = write_failed_list(failed, log_file_path)
        print(f"\nFailed list written to: {failed_list_path}")
        print(f"To retry:  python3 utils/docker_sweep.py {args.config_dir} "
              f"--retry-failed {failed_list_path}"
              + (f" --device {args.device}" if args.device else ""))

    if snapshot_dir:
        print(f"\nSnapshot at: {snapshot_dir}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()