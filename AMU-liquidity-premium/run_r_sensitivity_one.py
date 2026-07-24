"""Recompute AMU for a single discount rate r into its own artifacts directory.

Driven by run_r_sensitivity.sh, one subprocess per r so the PCP_R_OVERRIDE is
picked up at config-import time. HEAVY: rebuilds the ~18 GB tick-level
statistics frame (r enters e^{-rT} before aggregation, so it cannot be swept
cheaply the way `cost` can). Expects env: FROM_DATE, TO_DATE, ARTIFACTS_DIR,
PCP_R_OVERRIDE. Run outside the sandbox.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from amu_statistics import generate_all_statistics
from amu_cache import build_shared_cache_path
from panel_regressions import write_regression_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from_date = os.environ["FROM_DATE"]
to_date = os.environ["TO_DATE"]
artifacts_dir = Path(os.environ["ARTIFACTS_DIR"])
artifacts_dir.mkdir(parents=True, exist_ok=True)
r_value = os.environ.get("PCP_R_OVERRIDE", "<config default>")

cache_path = build_shared_cache_path(artifacts_dir, from_date, to_date)
logging.getLogger(__name__).info("r-sensitivity run r=%s artifacts_dir=%s", r_value, artifacts_dir)

# Force a fresh tick-level rebuild so the new r flows through e^{-rT}.
generate_all_statistics(
    artifacts_dir,
    from_date=from_date,
    to_date=to_date,
    force_recreate_cache=True,
    cache_path=cache_path,
)
# Aggregate the panel and emit text_numbers.tex (AmuPreBp / AmuPostBp / AmuMeanBp).
write_regression_tables(
    artifacts_dir,
    from_date=from_date,
    to_date=to_date,
    cache_path=cache_path,
    force_recreate_cache=True,
)
logging.getLogger(__name__).info("r-sensitivity done r=%s", r_value)
