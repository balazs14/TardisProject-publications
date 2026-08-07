#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAPER_DIR="$ROOT_DIR/publications/AMU-liquidity-premium"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

FROM_DATE=${FROM_DATE:-2020-01-01}
TO_DATE=${TO_DATE:-2026-06-05}
BUILD_PDF=${BUILD_PDF:-0}
LOG_LEVEL=${LOG_LEVEL:-INFO}
# Recreate the two caches selectively (0 = reuse if present, 1 = rebuild from raw):
#   RECREATE_STAT_CACHE  -> the tick-level statistics frame (18 GB)
#   RECREATE_PANEL_CACHE -> the aggregated panel (drives the regressions)
RECREATE_STAT_CACHE=${RECREATE_STAT_CACHE:-0}
RECREATE_PANEL_CACHE=${RECREATE_PANEL_CACHE:-0}
# SUBSAMPLE_DAYS -> development speedup: process only one calendar day in every N when
# rebuilding the stat/panel caches (selected deterministically by date, so the two stay
# aligned). Within-day computation is unchanged. 1 = full run (default). Set e.g.
# SUBSAMPLE_DAYS=10 for a ~10x-faster experimental rebuild; drop it for the final run.
SUBSAMPLE_DAYS=${SUBSAMPLE_DAYS:-1}
# RECREATE_R_SENSITIVITY -> re-run the full r-sensitivity sweep via
# run_r_sensitivity.sh (recomputes AMU across discount rates and rewrites the
# r-dependence table + plot). VERY expensive: one full ~18 GB tick-frame rebuild
# per r value. OFF by default; set to 1 only when you explicitly want to refresh
# it. Optionally set R_SENSITIVITY_VALUES="0.00 0.025 0.05 0.075 0.10".
RECREATE_R_SENSITIVITY=${RECREATE_R_SENSITIVITY:-0}
R_SENSITIVITY_VALUES=${R_SENSITIVITY_VALUES:-}
# ARTIFACTS_DIR controls where figures and tables are written.
# Use e.g. ARTIFACTS_DIR=artifacts_short for a 2024-only build and
# ARTIFACTS_DIR=artifacts_long for the full 2020-2026 build.
# The script always creates/updates an "artifacts" symlink pointing at the
# chosen directory so that liquidity-premium.tex sees the latest outputs.
ARTIFACTS_DIR=${ARTIFACTS_DIR:-artifacts_long_nofilterstale}

export FROM_DATE
export TO_DATE
export BUILD_PDF
export LOG_LEVEL
export RECREATE_STAT_CACHE
export RECREATE_PANEL_CACHE
export SUBSAMPLE_DAYS
export RECREATE_R_SENSITIVITY
export ARTIFACTS_DIR

cd "$PAPER_DIR"

mkdir -p "$ARTIFACTS_DIR"

# If artifacts/ is still a plain directory from before the multi-build scheme
# was introduced, migrate it to artifacts_long/ on first run.
if [[ -d artifacts && ! -L artifacts ]]; then
    echo "Migrating plain artifacts/ to artifacts_bak/ ..."
    mv artifacts artifacts_bak
fi

# Point the symlink at the target directory (relative, so the tex file works
# even if the repo is moved).
ln -sfn "$ARTIFACTS_DIR" artifacts
echo "artifacts -> $ARTIFACTS_DIR"

if [[ "${SUBSAMPLE_DAYS:-1}" -gt 1 ]]; then
	echo "SUBSAMPLE_DAYS=$SUBSAMPLE_DAYS"
	echo "  -> DEVELOPMENT MODE: rebuilding caches from only 1 calendar day in every $SUBSAMPLE_DAYS."
	echo "     Within-day calculations are unchanged, but results are a day-subsample. Unset (or =1) for the full run."
fi

if [[ -n "${ONLY_RECREATE_ARTIFACTS:-}" ]]; then
	echo "ONLY_RECREATE_ARTIFACTS=$ONLY_RECREATE_ARTIFACTS"
	echo "  -> regenerating only those artifacts (tags = figures-tables-overview.tex \\label strings, e.g. fig:friction_timeseries_direct, or file stems). Everything else is left as-is."
fi

"$VENV_PYTHON" - <<'PY'
import os
import logging
from pathlib import Path

import artifact_filter as art_filter
from amu_statistics import generate_all_statistics, write_dynamic_tex_assumptions
from amu_cache import build_shared_cache_path
from panel_figures import generate_all_figures
from panel_regressions import write_regression_tables
from cross_exchange_quotes import plot_cross_exchange_quote_diff

from_date = os.environ["FROM_DATE"]
to_date = os.environ["TO_DATE"]
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
matplotlib_log_level = max(log_level, logging.INFO)
recreate_stat_cache = os.environ.get("RECREATE_STAT_CACHE", "0") == "1"
recreate_panel_cache = os.environ.get("RECREATE_PANEL_CACHE", "0") == "1"
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
logging.getLogger(__name__).info(
	"recreate stat cache=%s panel cache=%s", recreate_stat_cache, recreate_panel_cache,
)

write_dynamic_tex_assumptions(artifacts_dir)

# Statistics frame (tick-level, 18 GB): controlled by RECREATE_STAT_CACHE.
generate_all_statistics(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	force_recreate_cache=recreate_stat_cache,
	cache_path=cache_path,
)

# Panel (aggregated, drives the regressions): controlled by RECREATE_PANEL_CACHE.
write_regression_tables(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	cache_path=cache_path,
	force_recreate_cache=recreate_panel_cache,
)

generate_all_figures(
	artifacts_dir,
	from_date=from_date,
	to_date=to_date,
	cache_path=cache_path,
	force_recreate_cache=recreate_panel_cache,
)

# Cross-exchange quote-difference figure (reads the aligned parquets directly; independent
# of the stat/panel caches). Gated so a selective run only pays for it when requested.
if art_filter.wanted("fig:cross_exchange_quote_diff", "cross_exchange_quote_diff"):
	plot_cross_exchange_quote_diff(from_date, to_date, artifacts_dir)
PY

# Optional, very expensive: recompute AMU across discount rates and refresh the
# r-dependence table + plot. Gated behind RECREATE_R_SENSITIVITY so a normal
# rebuild never triggers it.
if [[ "$RECREATE_R_SENSITIVITY" == "1" ]]; then
	echo "RECREATE_R_SENSITIVITY=1: running the r-sensitivity sweep (expensive)"
	# shellcheck disable=SC2086
	FROM_DATE="$FROM_DATE" TO_DATE="$TO_DATE" VENV_PYTHON="$VENV_PYTHON" \
		bash ./run_r_sensitivity.sh $R_SENSITIVITY_VALUES
else
	echo "Skipping r-sensitivity sweep (set RECREATE_R_SENSITIVITY=1 to run it)."
fi

if [[ "$BUILD_PDF" == "1" ]]; then
    latexmk -C >/dev/null 2>&1 || true
        rm -fr build    
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

	# Figures/tables-only overview (same artifacts, packed several per page).
	latexmk \
	  -pdf \
	  -synctex=1 \
	  -interaction=nonstopmode \
	  -file-line-error \
	  -outdir=build/ \
	  figures-tables-overview.tex
	cp build/figures-tables-overview.pdf figures-tables-overview.pdf

	# Presentation deck (built and copied in the same way as figures/tables overview).
	latexmk \
	  -pdf \
	  -synctex=1 \
	  -interaction=nonstopmode \
	  -file-line-error \
	  -outdir=build/ \
	  presentation.tex
	cp build/presentation.pdf presentation.pdf
fi
