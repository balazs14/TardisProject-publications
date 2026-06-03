#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CORE_SCRIPT="$SELF_DIR/bootstrap_publications_env.py"

if [[ ! -f "$CORE_SCRIPT" ]]; then
  echo "[PUB] error: core bootstrap script not found: $CORE_SCRIPT" >&2
  exit 1
fi

if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
  PYTHON_CMD="$PYTHON_BOOTSTRAP"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[PUB] error: no python interpreter found for bootstrap" >&2
  exit 1
fi

if [[ "${1:-}" == "--emit-shell" ]]; then
  shift
  "$PYTHON_CMD" "$CORE_SCRIPT" --emit-shell "$@"
  exit $?
fi

eval "$($PYTHON_CMD "$CORE_SCRIPT" --emit-shell "$@")"

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "[PUB] error: bootstrap did not export ROOT_DIR" >&2
  exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[PUB] root: $ROOT_DIR"
  if [[ -n "${PUBLICATIONS_VENV_PYTHON:-}" ]]; then
    echo "[PUB] python: $PUBLICATIONS_VENV_PYTHON"
  fi
fi