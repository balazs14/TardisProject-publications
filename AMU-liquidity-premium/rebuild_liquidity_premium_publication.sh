#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAPER_DIR="$ROOT_DIR/publications/AMU-liquidity-premium"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
FIGURE_DIR="$PAPER_DIR/liquidity_figures"
TABLE_DIR="$PAPER_DIR/regressions"

FROM_DATE=${FROM_DATE:-2020-01-01}
TO_DATE=${TO_DATE:-2026-06-05}
BUILD_PDF=${BUILD_PDF:-1}
LOG_LEVEL=${LOG_LEVEL:-INFO}

export FROM_DATE
export TO_DATE
export BUILD_PDF
export LOG_LEVEL

mkdir -p "$FIGURE_DIR" "$TABLE_DIR"

cd "$PAPER_DIR"

"$VENV_PYTHON" - <<'PY'
import os
import logging
from pathlib import Path

from amu_statistics import generate_all_statistics
from amu_cache import build_shared_cache_path
from panel_figures import generate_all_figures
from panel_regressions import write_regression_tables

from_date = os.environ["FROM_DATE"]
to_date = os.environ["TO_DATE"]
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
matplotlib_log_level = max(log_level, logging.INFO)
cache_path = build_shared_cache_path(Path("."), from_date, to_date)

logging.basicConfig(
	level=log_level,
	format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Force publication pipeline loggers to requested level after module imports.
logging.getLogger("tardis").setLevel(log_level)
logging.getLogger("tardis.process_pcp").setLevel(log_level)
logging.getLogger("amu_statistics").setLevel(log_level)
logging.getLogger("amu_panel").setLevel(log_level)
logging.getLogger("panel_regressions").setLevel(log_level)
logging.getLogger("panel_figures").setLevel(log_level)
# Keep matplotlib quiet unless explicitly logging warnings/errors.
logging.getLogger("matplotlib").setLevel(matplotlib_log_level)

logging.getLogger(__name__).info(
	"rebuild pipeline started from_date=%s to_date=%s log_level=%s",
	from_date,
	to_date,
	log_level_name,
)
logging.getLogger(__name__).info("shared dataframe cache path=%s", cache_path)

generate_all_statistics(
	Path("liquidity_figures"),
	from_date=from_date,
	to_date=to_date,
	cache_path=cache_path,
)

write_regression_tables(Path("regressions"), from_date=from_date, to_date=to_date, cache_path=cache_path)
generate_all_figures(Path("liquidity_figures"), from_date=from_date, to_date=to_date, cache_path=cache_path)
PY

if [[ "$BUILD_PDF" == "1" ]]; then
	latexmk -C >/dev/null 2>&1 || true
	mkdir -p build
	latexmk \
	  -pdf \
	  -synctex=1 \
	  -interaction=nonstopmode \
	  -file-line-error \
	  -outdir=build/ \
	  liquidity-premium.tex
	cp build/liquidity-premium.pdf liquidity-premium.pdf
	cp build/liquidity-premium.synctex.gz liquidity-premium.synctex.gz
fi
