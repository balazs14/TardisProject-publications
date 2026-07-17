#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAPER_DIR="$ROOT_DIR/publications/AMU-liquidity-premium"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

FROM_DATE=${FROM_DATE:-2020-01-01}
TO_DATE=${TO_DATE:-2026-06-05}
BUILD_PDF=${BUILD_PDF:-1}
LOG_LEVEL=${LOG_LEVEL:-INFO}
AMU_FORCE_RECREATE_CACHE=${AMU_FORCE_RECREATE_CACHE:-0}
# ARTIFACTS_DIR controls where figures and tables are written.
# Use e.g. ARTIFACTS_DIR=artifacts_short for a 2024-only build and
# ARTIFACTS_DIR=artifacts_long for the full 2020-2026 build.
# The script always creates/updates an "artifacts" symlink pointing at the
# chosen directory so that liquidity-premium.tex sees the latest outputs.
ARTIFACTS_DIR=${ARTIFACTS_DIR:-artifacts_long}

export FROM_DATE
export TO_DATE
export BUILD_PDF
export LOG_LEVEL
export AMU_FORCE_RECREATE_CACHE
export ARTIFACTS_DIR

cd "$PAPER_DIR"

mkdir -p "$ARTIFACTS_DIR"

# If artifacts/ is still a plain directory from before the multi-build scheme
# was introduced, migrate it to artifacts_long/ on first run.
if [[ -d artifacts && ! -L artifacts ]]; then
    echo "Migrating plain artifacts/ to artifacts_long/ ..."
    mv artifacts artifacts_long
fi

# Point the symlink at the target directory (relative, so the tex file works
# even if the repo is moved).
ln -sfn "$ARTIFACTS_DIR" artifacts
echo "artifacts -> $ARTIFACTS_DIR"

"$VENV_PYTHON" - <<'PY'
import os
import logging
from pathlib import Path

from amu_statistics import generate_all_statistics, write_dynamic_tex_assumptions
from amu_cache import build_shared_cache_path
from panel_figures import generate_all_figures
from panel_regressions import write_regression_tables

from_date = os.environ["FROM_DATE"]
to_date = os.environ["TO_DATE"]
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
matplotlib_log_level = max(log_level, logging.INFO)
force_recreate_cache = os.environ.get("AMU_FORCE_RECREATE_CACHE", "0") == "1"
artifacts_dir = Path(os.environ["ARTIFACTS_DIR"])
cache_path = build_shared_cache_path(artifacts_dir, from_date, to_date)

logging.basicConfig(
	level=log_level,
	format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logging.getLogger("tardis").setLevel(log_level)
logging.getLogger("tardis.process_pcp").setLevel(log_level)
logging.getLogger("amu_statistics").setLevel(log_level)
logging.getLogger("amu_panel").setLevel(log_level)
logging.getLogger("panel_regressions").setLevel(log_level)
logging.getLogger("panel_figures").setLevel(log_level)
logging.getLogger("matplotlib").setLevel(matplotlib_log_level)

logging.getLogger(__name__).info(
	"rebuild pipeline started from_date=%s to_date=%s artifacts_dir=%s log_level=%s",
	from_date, to_date, artifacts_dir, log_level_name,
)
logging.getLogger(__name__).info("shared dataframe cache path=%s", cache_path)
logging.getLogger(__name__).info("force recreate cache=%s", force_recreate_cache)

write_dynamic_tex_assumptions(artifacts_dir)

generate_all_statistics(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	force_recreate_cache=force_recreate_cache,
	cache_path=cache_path,
)

write_regression_tables(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	cache_path=cache_path,
	force_recreate_cache=force_recreate_cache,
)

generate_all_figures(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	cache_path=cache_path,
	force_recreate_cache=force_recreate_cache,
)
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
