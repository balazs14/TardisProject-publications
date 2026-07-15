from __future__ import annotations

import os
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    # Walk up from any starting directory so scripts/notebooks can be run from
    # arbitrary subdirectories inside TardisProject and still resolve repo-root
    # resources (for example datasets/, publications/, and tardis/ imports).
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "tardis").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project root with pyproject.toml and tardis/")


def bootstrap_repo_root(start: Path | None = None) -> Path:
    root = find_project_root(Path(start).resolve() if start is not None else Path(__file__).resolve())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_config() -> dict[str, Any]:
    configured = os.getenv("AMU_CONFIG_PATH", "").strip()
    config_path = Path(configured) if configured else Path(__file__).with_suffix(".toml")
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


CONFIG = _load_config()

PARQUET_BATCH_ROWS = int(CONFIG["cache"]["parquet_batch_rows"])

REL_STRIKE_MIN = float(CONFIG["filters"]["rel_strike_min"])
REL_STRIKE_MAX = float(CONFIG["filters"]["rel_strike_max"])
SPREAD_BP_MAX = int(CONFIG["filters"]["spread_bp_max"])
MIN_QUOTE_SIZE_DOLLAR = float(CONFIG["filters"]["min_quote_size_dollar"])
MIN_MMA_BP = float(CONFIG["filters"]["min_mma_bp"])
MAX_MMA_BP = float(CONFIG["filters"]["max_mma_bp"])
MAX_AMU_BP = int(CONFIG["filters"]["max_amu_bp"])

PCP_COST_PER_NOTIONAL = float(CONFIG["pcp"]["cost_per_notional"])
PCP_FUT_MGN_RATE = float(CONFIG["pcp"]["fut_mgn_rate"])
PCP_SHORT_PUT_MGN_RATE = float(CONFIG["pcp"]["short_put_mgn_rate"])
PCP_SHORT_CALL_MGN_RATE = float(CONFIG["pcp"]["short_call_mgn_rate"])
PCP_R = float(CONFIG["pcp"]["r"])
PCP_CONTRACT_SIZE = float(CONFIG["pcp"]["contract_size"])
PCP_METRIC_KWARGS = {
    "cost_per_notional": PCP_COST_PER_NOTIONAL,
    "fut_mgn_rate": PCP_FUT_MGN_RATE,
    "short_put_mgn_rate": PCP_SHORT_PUT_MGN_RATE,
    "short_call_mgn_rate": PCP_SHORT_CALL_MGN_RATE,
    "r": PCP_R,
    "contract_size": PCP_CONTRACT_SIZE,
}

REL_STRIKE_BUCKETS = int(CONFIG["panel"]["rel_strike_buckets"])
TTE_BUCKETS = int(CONFIG["panel"]["tte_buckets"])
TTE_SQRT_MIN = float(CONFIG["panel"]["tte_sqrt_min"])
TTE_SQRT_MAX = float(CONFIG["panel"]["tte_sqrt_max"])
MEAN_PANEL_COLUMNS = list(CONFIG["panel"]["mean_panel_columns"])
SUM_PANEL_COLUMNS = list(CONFIG["panel"]["sum_panel_columns"])
STALE_PANEL_COLUMNS = list(CONFIG["panel"]["stale_panel_columns"])

PCPB_COLUMNS = list(CONFIG["statistics"]["pcpb_columns"])

POST_2024_START = date.fromisoformat(CONFIG["regression"]["post_2024_start"])
NONATM_DISTANCE = float(CONFIG["regression"]["nonatm_distance"])
SHORT_TTE_CUTOFF = float(CONFIG["regression"]["short_tte_cutoff"])

EVENT_DATES = {
    label: date.fromisoformat(value)
    for label, value in CONFIG["figures"]["event_dates"].items()
}
