#!/usr/bin/env python3
"""
ml/train_profiled.py
--------------------
Drop-in replacement for ml/train.py that wraps the entire run in cProfile
and prints a ranked table of the top time-consuming functions at the end.

Usage (identical to train.py):
    python -m ml.train_profiled config.yml --device 0

Extra flags:
    --profile-top N     How many rows to show in the final table (default 50)
    --profile-out FILE  If given, also dump the raw .prof file for later use
                        with e.g. `python -m pstats myrun.prof`

WandB is suppressed by default to avoid polluting the profile with network I/O.
Pass --enable-wandb to re-enable.
"""

import cProfile
import pstats
import io
import os
import sys
sys.path.append(os.path.abspath(".."))

import argparse

# ── Parse our extra flags BEFORE handing control to training logic ───────────

_profiler_parser = argparse.ArgumentParser(add_help=False)
_profiler_parser.add_argument("--profile-top",  type=int, default=50)
_profiler_parser.add_argument("--profile-out",  type=str, default=None)
_profiler_parser.add_argument("--enable-wandb", action="store_true")
_profiler_args, _remaining = _profiler_parser.parse_known_args()

if not _profiler_args.enable_wandb:
    os.environ.setdefault("WANDB_MODE", "disabled")

sys.argv = [sys.argv[0]] + _remaining

# ── Run the full training script under cProfile ──────────────────────────────

_prof = cProfile.Profile()
_prof.enable()

try:
    import runpy
    runpy.run_module("ml.train", run_name="__main__", alter_sys=True)
finally:
    _prof.disable()

# ── Optionally save the raw .prof file ───────────────────────────────────────

_TOP_N    = _profiler_args.profile_top
_PROF_OUT = _profiler_args.profile_out

if _PROF_OUT:
    _prof.dump_stats(_PROF_OUT)

# ── Collect pstats output for both sort orders ───────────────────────────────

def _collect_stats(prof, sort_key, top_n):
    buf = io.StringIO()
    ps  = pstats.Stats(prof, stream=buf)
    ps.strip_dirs()
    ps.sort_stats(sort_key)
    ps.print_stats(top_n)
    lines = buf.getvalue().splitlines()
    start = next((i for i, l in enumerate(lines) if "ncalls" in l), 0)
    return lines, start, ps.total_calls, ps.total_tt

_cum_lines, _cum_start, _total_calls, _total_tt = _collect_stats(_prof, "cumulative", _TOP_N)
_self_lines, _self_start, _, _                  = _collect_stats(_prof, "tottime",    _TOP_N)

# ── Per-metric cost breakdown ─────────────────────────────────────────────────
#
# cProfile keys: (filename, lineno, funcname)
# cProfile does NOT record class names — a method `def compute(self, ...)` in
# class InceptionFIDMetric appears as just `compute` at its line number.
#
# Strategy: match by (filename_substr, lineno) which is exact and unambiguous.
# Line numbers come from metrics.py in this repo — update if the file changes.
#
# For each metric we also list fallback callee fingerprints (filename_substr,
# funcname_substr) for the unique inner functions, in case the outer compute()
# frame doesn't appear in the stats (e.g. too fast, inlined, or cached).
#
# We take MAX cumtime across all fingerprints to pick the outermost frame and
# avoid double-counting subcalls.

# Line numbers from metrics.py (update these if the file is edited)
_METRICS_FILE = "metrics.py"
_LINENO = {
    "w1_compute":           224,
    "w1_slice_compute":     243,
    "fid_compute":          277,
    "kid_compute":          303,
    "lenet_fid_compute":    327,
    "lenet_kid_compute":    347,
    "lenet_entropy_compute":382,
    "encode":               119,
    "classify":             126,
    "_ot_w1":                38,
    "_fid_from_feats":      139,
    "_kid_from_feats":      194,
    "_kid_from_feats_gpu":  171,
}

# Each entry: metric label → list of (file_substr, lineno_or_None, func_substr)
# lineno=None means match by funcname only (for external library functions)
_METRIC_FINGERPRINTS: dict[str, list[tuple[str, int | None, str]]] = {
    "w1": [
        (_METRICS_FILE, _LINENO["w1_compute"],   "compute"),
        (_METRICS_FILE, _LINENO["_ot_w1"],       "_ot_w1"),       # fallback: inner fn
        ("_network_simplex", None,               "f"),             # POT network simplex
    ],
    "w1_slice": [
        (_METRICS_FILE, _LINENO["w1_slice_compute"], "compute"),
    ],
    "fid": [
        (_METRICS_FILE, _LINENO["fid_compute"],  "compute"),
        ("fid.py",       None,                   "compute"),       # torchmetrics fid
    ],
    "kid": [
        (_METRICS_FILE, _LINENO["kid_compute"],  "compute"),
        ("kid.py",       None,                   "compute"),       # torchmetrics kid
    ],
    "lenet_fid": [
        (_METRICS_FILE, _LINENO["lenet_fid_compute"],  "compute"),
        (_METRICS_FILE, _LINENO["_fid_from_feats"],    "_fid_from_feats"),  # fallback
        ("",             None,                          "linalg_eigvals"),   # FID sqrt(cov)
    ],
    "lenet_kid": [
        (_METRICS_FILE, _LINENO["lenet_kid_compute"],    "compute"),
        (_METRICS_FILE, _LINENO["_kid_from_feats_gpu"],  "_kid_from_feats_gpu"),
        (_METRICS_FILE, _LINENO["_kid_from_feats"],      "_kid_from_feats"),
    ],
    "lenet_entropy": [
        (_METRICS_FILE, _LINENO["lenet_entropy_compute"], "compute"),
    ],
    # Shared backbone — included inside each lenet_* cumtime above,
    # shown separately so you can decompose backbone vs math cost.
    "  └─ lenet backbone [encode+classify]": [
        (_METRICS_FILE, _LINENO["encode"],    "encode"),
        (_METRICS_FILE, _LINENO["classify"],  "classify"),
    ],
    # ── Training loop context ─────────────────────────────────────────────────
    "model forward": [
        ("model.py",    None, "forward"),
        ("unet.py",     None, "forward"),
    ],
    "loss backward": [
        ("",            None, "run_backward"),
        ("",            None, "backward"),
    ],
    "reverse simulate": [
        ("sim_core.py", None, "simulate"),
    ],
    "score table build": [
        ("score_table.py", None, "build"),
        ("score_table.py", None, "build_score_table"),
    ],
    "data loading": [
        ("dataloader",  None, "__iter__"),
        ("dataloader",  None, "_worker_loop"),
    ],
}


def _lookup_cumtime(stats_dict, file_sub: str, lineno: int | None, func_sub: str) -> float:
    """
    Sum cumtime over matching pstats entries.
    If lineno is given, require exact match — this disambiguates methods that
    share the same name (e.g. multiple `compute` functions in metrics.py).
    If lineno is None, match by filename + funcname substrings only.
    """
    total = 0.0
    for (fname, ln, funcname), (_pc, _nc, _tt, cumtime, _callers) in stats_dict.items():
        fname_match = file_sub.lower() in fname.lower()
        func_match  = func_sub.lower() in funcname.lower()
        line_match  = (lineno is None) or (ln == lineno)
        if fname_match and func_match and line_match:
            total += cumtime
    return total


_ps_obj = pstats.Stats(_prof)
_ps_obj.strip_dirs()
_stats = _ps_obj.stats

_metric_costs: list[tuple[str, float]] = []
for _mname, _fps in _METRIC_FINGERPRINTS.items():
    _best = max(
        (_lookup_cumtime(_stats, fsub, ln, fn) for fsub, ln, fn in _fps),
        default=0.0,
    )
    if _best > 0.0:
        _metric_costs.append((_mname, _best))

# ── Pretty-print helpers ──────────────────────────────────────────────────────

_W    = 110
_SEP  = "=" * _W
_SEP2 = "-" * _W


def _banner(title):
    inner = f"  {title}  "
    pad_l = (_W - len(inner)) // 2
    pad_r = _W - len(inner) - pad_l
    return f"\n{_SEP}\n{'─'*pad_l}{inner}{'─'*pad_r}\n{_SEP}"


def _print_table(lines, start):
    for line in lines[start:]:
        print(line)


# ── Output ───────────────────────────────────────────────────────────────────

print()
print(_banner("PROFILER  REPORT"))
print(f"\n  Total function calls : {_total_calls:>12,}")
print(f"  Total CPU time       : {_total_tt:>12.3f} s")

print(_banner(f"TOP {_TOP_N} BY CUMULATIVE TIME  —  slow call chains · I/O · forward/backward"))
print()
_print_table(_cum_lines, _cum_start)

print(_banner(f"TOP {_TOP_N} BY SELF TIME  —  actual compute hotspots · inner loops"))
print()
_print_table(_self_lines, _self_start)

print(_banner("PER-METRIC COST SUMMARY  —  cumtime of the outermost compute() per metric"))
print()
if _metric_costs:
    _col1 = max(len(n) for n, _ in _metric_costs) + 2
    _col1 = max(_col1, 44)
    print(f"  {'metric':<{_col1}} {'cumtime (s)':>12}   {'% of total':>10}   note")
    print("  " + "-" * (_col1 + 42))
    for _mname, _ct in sorted(_metric_costs, key=lambda x: -x[1]):
        _pct  = 100.0 * _ct / _total_tt if _total_tt > 0 else 0.0
        _note = ""
        if _mname == "w1" and _ct > 30:
            _note = "← heavy; consider w1_slice"
        elif "backbone" in _mname:
            _note = "← shared across all lenet_* metrics"
        print(f"  {_mname:<{_col1}} {_ct:>12.3f}   {_pct:>9.1f}%   {_note}")
    print()
    print(f"  {'total profiled run':<{_col1}} {_total_tt:>12.3f}   {'100.0%':>10}")
    print()
    print("  Note: lenet_* cumtimes INCLUDE backbone cost.")
    print("  The '└─ lenet backbone' row shows how much of that is backbone vs math.")
    print("  Unaccounted time = training loop overhead, optimizer, logging, etc.")
else:
    print("  No metric fingerprints matched — run may not have reached an eval step.")
    print("  Check the cumulative table above for metrics.py entries.")

print()
print(_SEP)
print("  HOW TO READ")
print(_SEP2)
print("  ncalls   – how many times the function was called")
print("  tottime  – time spent INSIDE this function only  (self; excludes subcalls)")
print("  percall  – tottime / ncalls")
print("  cumtime  – tottime + all subcalls  (total wall time attributed to this entry)")
print("  percall  – cumtime / ncalls")
print()
print("  Cumulative table → find slow call chains (model forward, metric compute, data loading)")
print("  Self-time  table → find the actual hotspot (innermost function eating CPU cycles)")
print("  Per-metric table → directly compare metric evaluation cost across the active set")
print()
print("  Common patterns:")
print("    _network_simplex / _ot_w1   exact W1 is O(n³) CPU — consider w1_slice instead")
print("    score_table build           one-time startup cost — large N_x/N_t makes this slow")
print("    DataLoader / _worker_loop   I/O bound — try pin_memory=True or more num_workers")
print("    run_backward                gradient pass — expected to dominate on GPU")
print("    encode / classify           LeNet backbone — shared cost across lenet_* metrics")
print("    linalg_eigvals              matrix sqrt in FID computation (lenet_fid)")
print(_SEP)

if _PROF_OUT:
    print(f"\n  Raw .prof saved to : {_PROF_OUT}")
    print(f"  Interactive shell  : python -m pstats {_PROF_OUT}")
    print(f"  Visual flamegraph  : snakeviz {_PROF_OUT}   (pip install snakeviz)")
    print(_SEP)