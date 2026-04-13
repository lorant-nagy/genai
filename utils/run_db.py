#!/usr/bin/env python3
"""
run_db.py  —  Experiment database builder
==========================================
Parses all run folders and materialises a structured, flat JSON database —
one record per run. A pure mirror: no derivation, no scalar reduction.
Raw time-series and raw config values only.

Re-running overwrites the existing DB file.

USAGE
-----
    python -m utils.run_db <runs_folder> <db_path>

EXAMPLES
--------
    python -m utils.run_db run_logs/surface/runs db/surface.json

DB SCHEMA (one record per run)
-------------------------------
{
    "run_name":  str,       # folder name
    "parsed_at": str,       # ISO timestamp of when this record was built
    "config":    { ... },   # full raw config (nested dict, as in config.yaml)
    "params":    {          # aliased flat params — convenient shorthand
        "T":       float,
        "alpha":   float,
        "n_steps": int,
        ...                 # extend AXIS_ALIASES below to add more
    },
    "metrics":   {          # raw time-series from metrics_evo.json
        "w1":     [float|null, ...],
        "loss":   [float|null, ...],
        "trash%": [float|null, ...],
        "epochs": [int, ...],
        "nan":    [str, ...],
        ...
    }
}

EXTENDING THE PARAM ALIASES
----------------------------
To track a new sweep dimension, add one entry to AXIS_ALIASES:

    AXIS_ALIASES["lr"] = "training.optimizer_params.lr"

The key becomes the short name used everywhere (in the DB "params" block
and in flex_heatmap AXIS_X / AXIS_Y / AXIS_SLICE).
The value is the dotted YAML path into config.yaml.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Param alias registry ──────────────────────────────────────────────────────
# Maps short name  →  dotted path into the config YAML.
# This is the single source of truth for all analysis scripts.

AXIS_ALIASES: dict[str, str] = {
    "T":       "corruption.process_params.T",
    "alpha":   "corruption.process_params.alpha",
    "n_steps": "corruption.corruptor_params.n_steps",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_nested(d: dict, dotted_path: str):
    """Traverse a nested dict with a dotted path. Returns None if missing."""
    cur = d
    for k in dotted_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _sanitise_series(values: list) -> list:
    """
    Normalise a raw metrics_evo series for JSON storage.
    Keeps None, strings, ints, floats as-is.
    Attempts to cast anything exotic (numpy scalars etc.) to float.
    """
    out = []
    for v in values:
        if v is None or v == "":
            out.append(None)
        elif isinstance(v, str):
            out.append(v)
        elif isinstance(v, bool):
            out.append(v)
        elif isinstance(v, (int, float)):
            out.append(v)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(None)
    return out


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_run(run_dir: Path) -> dict | None:
    """
    Parse one run folder into a DB record.
    Returns None if the folder is incomplete or its files are unreadable.
    """
    config_path  = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics_evo.json"

    if not config_path.exists() or not metrics_path.exists():
        return None

    try:
        with open(config_path) as f:
            cfg: dict = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics_evo: dict = json.load(f)
    except (json.JSONDecodeError, yaml.YAMLError, OSError) as e:
        print(f"  Warning: skipping {run_dir.name} — {e}")
        return None

    params: dict = {
        alias: _get_nested(cfg, path)
        for alias, path in AXIS_ALIASES.items()
    }

    metrics: dict = {
        key: _sanitise_series(series) if isinstance(series, list) else [series]
        for key, series in metrics_evo.items()
    }

    return {
        "run_name":  run_dir.name,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "config":    cfg,
        "params":    params,
        "metrics":   metrics,
    }


# ── Build ─────────────────────────────────────────────────────────────────────

def build_db(runs_dir: Path) -> list[dict]:
    """Parse all run folders in runs_dir and return the record list."""
    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    print(f"  Run folders found : {len(run_dirs)}")

    records, skipped = [], 0
    for run_dir in run_dirs:
        record = parse_run(run_dir)
        if record is None:
            skipped += 1
        else:
            records.append(record)

    if skipped:
        print(f"  Skipped (incomplete or unreadable) : {skipped}")
    print(f"  Records built : {len(records)}")
    return records


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_db(records: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "w") as f:
        json.dump(records, f, indent=2)


def load_db(db_path: Path) -> list[dict]:
    """Load a DB file produced by this script."""
    with open(db_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{db_path} does not contain a JSON array.")
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m utils.run_db <runs_folder> <db_path>")
        print("Example: python -m utils.run_db run_logs/surface/runs db/surface.json")
        sys.exit(1)

    runs_dir = Path(sys.argv[1])
    db_path  = Path(sys.argv[2])

    if not runs_dir.exists():
        print(f"Error: runs folder '{runs_dir}' does not exist.")
        sys.exit(1)

    print()
    print("=" * 50)
    print("  RUN DB")
    print("=" * 50)
    print(f"  Runs folder : {runs_dir}")
    print(f"  DB file     : {db_path}")
    print("=" * 50)

    records = build_db(runs_dir)

    if not records:
        print("  No valid records — nothing written.")
        sys.exit(0)

    save_db(records, db_path)
    print(f"  Saved → {db_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()