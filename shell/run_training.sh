#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./run_configs.sh <config.yaml | folder_with_yamls>

if [[ $# -ne 1 ]]; then
  echo "ERROR: Exactly one argument required: a YAML config file OR a folder containing YAMLs." >&2
  exit 1
fi

INPUT="$1"

# ---- helpers ----
die() { echo "ERROR: $*" >&2; exit 1; }

# Parse YAML using PyYAML (no silent fallbacks).
# Required keys (new schema):
#   env.root         -> REPO_ROOT
#   env.int_env      -> ENV_NAME
#   env.results_dir  -> RESULTS_BASE
parse_yaml () {
  local cfg="$1"
  local out
  out="$(
    python3 - "$cfg" <<'PY' || exit 9
import sys, json
try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required. Install it (e.g., `python3 -m pip install pyyaml`).\n")
    sys.exit(2)

cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    sys.stderr.write(f"Failed to read YAML: {e}\n")
    sys.exit(3)

def need(path):
    d = cfg
    for k in path.split('.'):
        if not isinstance(d, dict) or k not in d or d[k] is None:
            sys.stderr.write(f"Missing required key '{path}' in {cfg_path}\n")
            sys.exit(4)
        d = d[k]
    return d

root         = need("env.root")
env_name     = need("env.int_env")
results_base = need("env.results_dir")

print(json.dumps({"root": root, "env": env_name, "results_base": results_base}))
PY
  )" || die "Parsing failed for $cfg"

  REPO_ROOT=$(printf '%s' "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin)["root"])')
  ENV_NAME=$(printf '%s' "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin)["env"])')
  RESULTS_BASE=$(printf '%s' "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin)["results_base"])')

  [[ -n "$REPO_ROOT"     ]] || die "env.root is empty in $cfg"
  [[ -n "$ENV_NAME"      ]] || die "env.int_env is empty in $cfg"
  [[ -n "$RESULTS_BASE"  ]] || die "env.results_dir is empty in $cfg"
}

run_one_config () {
  local cfg="$1"

  [[ -f "$cfg" ]] || die "Config path is not a file: $cfg"
  [[ "$cfg" == *.yml || "$cfg" == *.yaml ]] || die "Config must be .yml or .yaml: $cfg"

  parse_yaml "$cfg"

  # Validate repo layout and entry point
  [[ -d "$REPO_ROOT" ]] || die "env.root does not exist: $REPO_ROOT"
  local ENTRY="$REPO_ROOT/ml/train.py"
  [[ -f "$ENTRY" ]] || die "Entry script not found: $ENTRY (expected train.py under <env.root>/ml/)"

  # Prepare results dir: <env.results_dir>/<config_stem>/<YYYYmmdd_HHMMSS>
  local stem ts RESULTS_DIR LOG_FILE
  stem=$(basename "$cfg"); stem="${stem%.*}"
  ts=$(date +"%Y%m%d_%H%M%S")
  RESULTS_DIR="$RESULTS_BASE/$stem/$ts"
  mkdir -p "$RESULTS_DIR" || die "Failed to create results dir: $RESULTS_DIR"
  LOG_FILE="$RESULTS_DIR/shell_log"

  # Environment setup (no defaults; errors if conda/env missing)
  command -v conda >/dev/null 2>&1 || die "conda not found in PATH"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || die "Failed to source conda.sh"
  conda activate "$ENV_NAME" || die "Failed to activate env: $ENV_NAME"

  export MPLBACKEND=Agg
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

  # Run (live log with tee; preserve python exit code)
  echo "==> Running config: $cfg" | tee -a "$LOG_FILE"
  echo "    env: $ENV_NAME" | tee -a "$LOG_FILE"
  echo "    repo: $REPO_ROOT" | tee -a "$LOG_FILE"
  echo "    results: $RESULTS_DIR" | tee -a "$LOG_FILE"

  cd "$REPO_ROOT/ml" || die "Cannot cd to $REPO_ROOT/ml"

  set +e
  python -u "$ENTRY" "$cfg" "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    echo "FAILED (exit $status) for config: $cfg" | tee -a "$LOG_FILE"
    exit $status
  fi

  echo "SUCCESS for config: $cfg" | tee -a "$LOG_FILE"
}

# ---- main: single file or folder of configs ----
if [[ -d "$INPUT" ]]; then
  mapfile -t CFGS < <(find "$INPUT" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) | sort)
  [[ ${#CFGS[@]} -gt 0 ]] || die "No *.yml/*.yaml files found in folder: $INPUT"
  for cfg in "${CFGS[@]}"; do
    run_one_config "$cfg"
  done
elif [[ -f "$INPUT" ]]; then
  run_one_config "$INPUT"
else
  die "Not a file or folder: $INPUT"
fi
