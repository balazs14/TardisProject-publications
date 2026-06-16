#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAPER_DIR="$ROOT_DIR/publications/AMU-liquidity-premium"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
FIGURE_DIR="$PAPER_DIR/build/liquidity_figures"
TABLE_DIR="$PAPER_DIR/build/regressions"

FROM_DATE=${FROM_DATE:-2020-01-01}
TO_DATE=${TO_DATE:-2026-06-05}
BUILD_PDF=${BUILD_PDF:-1}

mkdir -p "$FIGURE_DIR" "$TABLE_DIR"

cd "$PAPER_DIR"

"$VENV_PYTHON" - <<'PY'
import os
from pathlib import Path

from panel_figures import generate_all_figures
from panel_regressions import write_regression_tables

from_date = os.environ["FROM_DATE"]
to_date = os.environ["TO_DATE"]

write_regression_tables(Path("build/regressions"), from_date=from_date, to_date=to_date)
generate_all_figures(Path("build/liquidity_figures"), from_date=from_date, to_date=to_date)
PY

if [[ "$BUILD_PDF" == "1" ]]; then
	latexmk -C >/dev/null 2>&1 || true
	rm -rf build/latex
	mkdir -p build/latex
	latexmk \
	  -pdf \
	  -synctex=1 \
	  -interaction=nonstopmode \
	  -file-line-error \
	  -outdir=build/latex \
	  liquidity-premium.tex
	cp build/latex/liquidity-premium.pdf liquidity-premium.pdf
	cp build/latex/liquidity-premium.synctex.gz liquidity-premium.synctex.gz
fi