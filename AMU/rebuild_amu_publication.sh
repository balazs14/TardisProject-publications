#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# OPERATOR REMINDER
# - Run this from anywhere; it auto-detects repo root from script location.
# - Expected data location: <repo>/datasets/{deribit,okex}/*_aligned_options_*_5min.parquet
# - Regenerated artifacts: AMU PNG figures + executed notebook outputs in publications/AMU.
# - If AMU compacted parquet is missing, this script will copy datasets/compacted.parquet when available.
#   Otherwise set AMU_RECREATE_PCPB=1 to rebuild compacted data from raw aligned files.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# HASH LOG (manual "best-so-far" updates)
# TARDISPROJECT_COMMIT=
# PUBLICATIONS_COMMIT=
# PYTHON_VERSION=3.12.x
# NOTES=
# -----------------------------------------------------------------------------

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SELF_DIR/../.." && pwd)"
AMU_DIR="$ROOT_DIR/publications/AMU"
NOTEBOOK="$AMU_DIR/multi_exchange_plots.ipynb"
PDF_BUILD_DIR="$AMU_DIR/build"

DRY_RUN="${DRY_RUN:-0}"
AMU_RECREATE_PCPB="${AMU_RECREATE_PCPB:-0}"
BUILD_PDF="${BUILD_PDF:-0}"

echo "[AMU] root: $ROOT_DIR"
echo "[AMU] notebook: $NOTEBOOK"
echo "[AMU] DRY_RUN=$DRY_RUN AMU_RECREATE_PCPB=$AMU_RECREATE_PCPB BUILD_PDF=$BUILD_PDF"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[AMU] dry run only; no changes made."
  exit 0
fi

if [[ ! -d "$ROOT_DIR/venv" ]]; then
  python3 -m venv "$ROOT_DIR/venv"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$ROOT_DIR/requirements.txt" >/dev/null
python -m pip install nbconvert ipykernel >/dev/null
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if [[ -f "$ROOT_DIR/datasets/compacted.parquet" && ! -f "$AMU_DIR/compacted.parquet" ]]; then
  cp "$ROOT_DIR/datasets/compacted.parquet" "$AMU_DIR/compacted.parquet"
  echo "[AMU] seeded publications/AMU/compacted.parquet from datasets/compacted.parquet"
fi

export AMU_RECREATE_PCPB
python -m nbconvert --to notebook --execute --inplace "$NOTEBOOK"

if [[ "$BUILD_PDF" == "1" ]]; then
  (
    cd "$AMU_DIR"
    mkdir -p "$PDF_BUILD_DIR"

    # Task-like LaTeX build: keep intermediates in build/, emit SyncTeX metadata,
    # and keep file:line error formatting for easier source navigation.
    latexmk \
      -pdf \
      -synctex=1 \
      -interaction=nonstopmode \
      -file-line-error \
      -outdir=build \
      main.tex

    # Convenience copy: keep top-level PDF while intermediates stay in build/.
    cp "$PDF_BUILD_DIR/main.pdf" "$AMU_DIR/main.pdf"
    cp "$PDF_BUILD_DIR/main.synctex.gz" "$AMU_DIR/main.synctex.gz"
  )
fi

echo "[AMU] done. Expected artifacts in publications/AMU:"
echo "  - summary_daily_option_coverage.png"
echo "  - amu_summary.png"
echo "  - multi_exchange_amu_bps_by_date.png"
echo "  - multi_exchange_amu_bps_by_rel_strike.png"
echo "  - multi_exchange_amu_bps_by_tte.png"
