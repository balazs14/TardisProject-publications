#!/usr/bin/env bash
set -euo pipefail

# Locate project root by walking up from this script until both of these exist:
#   - a "tardis" subdirectory
#   - a "pyproject.toml" file
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SELF_DIR"
ROOT_DIR=""

while [[ "$SEARCH_DIR" != "/" ]]; do
  if [[ -d "$SEARCH_DIR/tardis" && -f "$SEARCH_DIR/pyproject.toml" ]]; then
    ROOT_DIR="$SEARCH_DIR"
    break
  fi
  SEARCH_DIR="$(dirname -- "$SEARCH_DIR")"
done

if [[ -z "$ROOT_DIR" ]]; then
  echo "[PUB] error: could not find ROOT_DIR (expected a directory containing both 'tardis/' and 'pyproject.toml')." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/venv" ]]; then
  echo "[PUB] creating venv at $ROOT_DIR/venv"
  python3 -m venv "$ROOT_DIR/venv"
fi

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  "$ROOT_DIR/venv/bin/python" -m pip install --upgrade pip >/dev/null
  "$ROOT_DIR/venv/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" >/dev/null
  "$ROOT_DIR/venv/bin/python" -m pip install ipykernel >/dev/null
  echo "[PUB] installed requirements from $ROOT_DIR/requirements.txt"
else
  echo "[PUB] requirements.txt not found at $ROOT_DIR/requirements.txt; skipping dependency install"
fi

KERNEL_NAME="${PUBLICATIONS_KERNEL_NAME:-tardisproject-venv}"
KERNEL_DISPLAY_NAME="${PUBLICATIONS_KERNEL_DISPLAY_NAME:-Python (TardisProject venv)}"
"$ROOT_DIR/venv/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME" >/dev/null
echo "[PUB] registered Jupyter kernel: $KERNEL_DISPLAY_NAME ($KERNEL_NAME)"

echo "[PUB] root: $ROOT_DIR"