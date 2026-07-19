from __future__ import annotations

from collections.abc import Iterable
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from amu_config import CONFIG, bootstrap_repo_root
from amu_cache import get_cached_frame, put_cached_frame
from amu_metrics import (
    amu_conditional_bp_expr,
    amu_unconditional_bp_expr,
    positive_count_expr,
    positive_part_sum_expr,
    tickpath_amu_agg_exprs,
)
from tardis import package_set_log_level
from tardis.process_pcp import compute_pcp_metrics
from tardis.utils import debug_runtime


PROJECT_ROOT = bootstrap_repo_root(Path(__file__).resolve())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
package_set_log_level(logging.DEBUG)

panel_cfg = CONFIG["panel"]
pcp_cfg = CONFIG["pcp"]
filters_cfg = CONFIG["filters"]

mean_panel_columns = list(panel_cfg["mean_panel_columns"])
sum_panel_columns = list(panel_cfg["sum_panel_columns"])
stale_panel_columns = list(panel_cfg["stale_panel_columns"])

rel_strike_buckets = int(panel_cfg["rel_strike_buckets"])
tte_buckets = int(panel_cfg["tte_buckets"])
tte_sqrt_min = float(panel_cfg["tte_sqrt_min"])
tte_sqrt_max = float(panel_cfg["tte_sqrt_max"])
rel_strike_min = float(filters_cfg["rel_strike_min"])
rel_strike_max = float(filters_cfg["rel_strike_max"])
max_amu_bp = float(filters_cfg["max_amu_bp"])
filter_stale = bool(filters_cfg.get("filter_stale", False))
# Absolute ARP cost grid (basis points): conditional/unconditional ARP precomputed
# per cell at each grid cost, so calculations can be redone at any cost without
# rebuilding from the tick frame. Grid costs are ABSOLUTE and independent of
# cost_per_notional. Because MMA = gross_wedge - c0_bp, evaluating ARP at cost c uses
# the offset delta = c - c0_bp, and (MMA - delta) = gross_wedge - c cancels c0_bp
# exactly -- so a grid cost of 20 always means ARP at 20 bp of the gross wedge,
# regardless of cost_per_notional. Grid is [start, end, step], end inclusive.
c0_bp = float(pcp_cfg["cost_per_notional"]) * 1.0e4
# cost_grid is in fractions of notional (like cost_per_notional); convert to integer
# basis points for the offset math and the per-cost column names.
_cost_grid = list(pcp_cfg.get("cost_grid", [0.0005, 0.005, 0.0005]))
_cg_bp = [int(round(float(x) * 1.0e4)) for x in _cost_grid]  # [start_bp, end_bp, step_bp]
COST_GRID_BP = list(range(_cg_bp[0], _cg_bp[1] + 1, _cg_bp[2]))
pcp_metric_kwargs = {
    "cost_per_notional": float(pcp_cfg["cost_per_notional"]),
    "fut_mgn_rate": float(pcp_cfg["fut_mgn_rate"]),
    "short_put_mgn_rate": float(pcp_cfg["short_put_mgn_rate"]),
    "short_call_mgn_rate": float(pcp_cfg["short_call_mgn_rate"]),
    "r": float(pcp_cfg["r"]),
    "contract_size": float(pcp_cfg["contract_size"]),
}


def _bucket_midpoint(values: pl.Series, lower: float, upper: float, bucket_count: int, name: str) -> pl.Series:
    edges = np.linspace(lower, upper, bucket_count + 1)
    midpoints_by_bucket = 0.5 * (edges[:-1] + edges[1:])
    midpoints: list[float | None] = []
    for value in values.to_list():
        if value is None:
            midpoints.append(None)
        else:
            midpoints.append(float(midpoints_by_bucket[int(value)]))
    return pl.Series(name, midpoints, dtype=pl.Float64)


def _bucket_upper(values: pl.Series, lower: float, upper: float, bucket_count: int, name: str) -> pl.Series:
    edges = np.linspace(lower, upper, bucket_count + 1)
    uppers_by_bucket = edges[1:]
    uppers: list[float | None] = []
    for value in values.to_list():
        if value is None:
            uppers.append(None)
        else:
            uppers.append(float(uppers_by_bucket[int(value)]))
    return pl.Series(name, uppers, dtype=pl.Float64)


def _parse_file_day(file_path: Path) -> date | None:
    parts = file_path.stem.split("_")
    if len(parts) < 5:
        return None
    try:
        return date.fromisoformat(parts[-2])
    except ValueError:
        return None


def _day_in_range(day: date, from_date: str | None, to_date: str | None) -> bool:
    if from_date is not None and day < date.fromisoformat(from_date):
        return False
    if to_date is not None and day > date.fromisoformat(to_date):
        return False
    return True


def _aligned_panel_files(
    exchange: str,
    sample_freq: str,
    raw_data_dir: str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[Path]:
    root = Path(raw_data_dir.format(exchange=exchange))
    pattern = f"{exchange}_aligned_put_call_quotes_trades_chain_*_{sample_freq}.parquet"
    files = []
    for file_path in sorted(root.glob(pattern)):
        file_day = _parse_file_day(file_path)
        if file_day is None or _day_in_range(file_day, from_date, to_date):
            files.append(file_path)
    return files


def _bucket_index(values: pl.Series, lower: float, upper: float, bucket_count: int, name: str) -> pl.Series:
    numbers = values.cast(pl.Float64, strict=False).to_numpy()
    buckets: list[int | None] = [None] * len(numbers)
    valid = np.isfinite(numbers)

    if valid.any():
        edges = np.linspace(lower, upper, bucket_count + 1)
        raw = np.searchsorted(edges, numbers[valid], side="right") - 1
        clipped = np.clip(raw, 0, bucket_count - 1)
        valid_indices = np.flatnonzero(valid)
        for idx, bucket in zip(valid_indices, clipped, strict=False):
            buckets[int(idx)] = int(bucket)

    return pl.Series(name, buckets, dtype=pl.Int64)


def _rel_strike_bucket(df: pl.DataFrame) -> pl.Series:
    index = _bucket_index(df.get_column("rel_strike"), rel_strike_min, rel_strike_max, rel_strike_buckets, "rel_strike_bucket_idx")
    return _bucket_midpoint(index, rel_strike_min, rel_strike_max, rel_strike_buckets, "rel_strike_bucket")


def _tte_bucket(df: pl.DataFrame) -> pl.Series:
    tte = df.get_column("tte").cast(pl.Float64, strict=False).to_numpy()
    sqrt_tte = np.sqrt(np.clip(tte, a_min=0.0, a_max=None))
    if not np.isfinite(sqrt_tte).any():
        return pl.Series("tte_bucket", [None] * len(tte), dtype=pl.Float64)
    index = _bucket_index(pl.Series("tte", sqrt_tte), tte_sqrt_min, tte_sqrt_max, tte_buckets, "tte_bucket_idx")
    tte_upper_sqrt = _bucket_upper(index, tte_sqrt_min, tte_sqrt_max, tte_buckets, "tte_bucket_sqrt_upper")
    tte_upper_actual = [None if value is None else float(value) ** 2 for value in tte_upper_sqrt.to_list()]
    return pl.Series("tte_bucket", tte_upper_actual, dtype=pl.Float64)


def _required_panel_columns() -> list[str]:
    return [
        "timestamp",
        "exchange",
        "ref_sym",
        "rel_strike",
        "tte",
        "index",
        "mma_fwd_bp",
        "mma_bck_bp",
        "pcpb_fwd_bp",
        "pcpb_bck_bp",
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
    ]


def _panel_metrics() -> list[pl.Expr]:
    metrics = [pl.len().alias("n_obs")]
    mean_output_names = {
        "call_opt_spread_bp": "mean_call_spread_bp",
        "put_opt_spread_bp": "mean_put_spread_bp",
    }
    metrics.extend(
        pl.mean(column).alias(mean_output_names.get(column, f"mean_{column}"))
        for column in mean_panel_columns
    )
    # ARP accumulators (shared definition in amu_metrics): the positive-part sum and
    # the count of positive tickpaths across the four paths. n_obs (the tick count)
    # is the denominator for the unconditional flavour; both mean_amu_conditional_bp
    # and mean_amu_unconditional_bp are formed as ratios after aggregation below.
    metrics.extend(
        tickpath_amu_agg_exprs(
            max_amu_bp, sum_alias="sum_amu_tickpath_bp", count_alias="num_amu_tickpath"
        )
    )
    # Cost-grid ARP accumulators: for each grid cost c, the positive-part sum and the
    # executable-path count at offset delta = c - c0_bp. The conditional/unconditional
    # ratios are formed from these (and n_obs) after aggregation.
    for c in COST_GRID_BP:
        delta = float(c) - c0_bp
        metrics.append(positive_part_sum_expr(max_amu_bp, delta).alias(f"amu_possum_bp_c{c:02d}"))
        metrics.append(positive_count_expr(delta).cast(pl.Float64).alias(f"amu_poscount_c{c:02d}"))
    metrics.extend(pl.sum(column).alias(f"sum_{column}") for column in sum_panel_columns)
    metrics.extend(
        (pl.col(column).cast(pl.Float64, strict=False).mean() * 1.0).alias(f"frac_{column}")
        for column in stale_panel_columns
    )
    return metrics


def _panel_block_from_file(file_path: Path) -> pl.DataFrame:
    raw = pl.read_parquet(file_path)
    if raw.is_empty():
        logger.debug("Skipping empty aligned file: %s", file_path)
        return pl.DataFrame()

    panel_ready = compute_pcp_metrics(raw, **pcp_metric_kwargs)
    panel_ready = panel_ready.drop_nulls(_required_panel_columns())
    if filter_stale:
        for column in stale_panel_columns:
            if column in panel_ready.columns:
                panel_ready = panel_ready.filter(~pl.col(column).fill_null(False).cast(pl.Boolean))
    if panel_ready.is_empty():
        return pl.DataFrame()
    panel_ready = panel_ready.with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        _rel_strike_bucket(panel_ready),
        _tte_bucket(panel_ready),
    )

    group_keys = ["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"]
    block = panel_ready.group_by(group_keys, maintain_order=True).agg(_panel_metrics())
    # Both explicit ARP flavours per cell at the baseline cost: conditional (over
    # positive tickpaths) and unconditional (over all tickpaths = frequency x size).
    ratio_columns = [
        amu_conditional_bp_expr("sum_amu_tickpath_bp", "num_amu_tickpath").alias("mean_amu_conditional_bp"),
        amu_unconditional_bp_expr("sum_amu_tickpath_bp", "n_obs").alias("mean_amu_unconditional_bp"),
    ]
    # Same two flavours at each grid cost, derived from the per-cost accumulators.
    for c in COST_GRID_BP:
        possum, poscount = f"amu_possum_bp_c{c:02d}", f"amu_poscount_c{c:02d}"
        ratio_columns.append(amu_conditional_bp_expr(possum, poscount).alias(f"mean_amu_cond_bp_c{c:02d}"))
        ratio_columns.append(amu_unconditional_bp_expr(possum, "n_obs").alias(f"mean_amu_uncond_bp_c{c:02d}"))
    return block.with_columns(ratio_columns)


def _append_block(panel: pl.DataFrame, block: pl.DataFrame) -> pl.DataFrame:
    if panel.is_empty():
        return block
    if block.is_empty():
        return panel
    return pl.concat([panel, block], how="vertical_relaxed", rechunk=False)


@debug_runtime("build_amu_panel")
def build_amu_panel(
    exchanges: Iterable[str] = ("okex", "deribit"),
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/{exchange}/",
    from_date: str | None = None,
    to_date: str | None = None,
    cache_path: str | Path | None = None,
    force_recreate_cache: bool = False,
) -> pl.DataFrame:
    """Build a compact panel from aligned put/call quote-trade-chain files.

    When from_date and to_date are provided, only files whose embedded day falls
    within the inclusive range are processed.
    """

    if cache_path is not None and not force_recreate_cache:
        cached_panel = get_cached_frame(cache_path, "amu_panel")
        if cached_panel is not None:
            logger.debug("Cache hit for build_amu_panel: %s", cache_path)
            return cached_panel
    elif cache_path is not None and force_recreate_cache:
        logger.info("Force cache recreation enabled for panel parquet: %s", cache_path)

    panel = pl.DataFrame()
    for exchange in exchanges:
        for file_path in _aligned_panel_files(exchange, sample_freq, raw_data_dir, from_date=from_date, to_date=to_date):
            block = _panel_block_from_file(file_path)
            panel = _append_block(panel, block)
    panel = panel.sort(["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"])

    if cache_path is not None:
        logger.debug("Cache write for build_amu_panel: %s", cache_path)
        put_cached_frame(cache_path, "amu_panel", panel, from_date=from_date, to_date=to_date)

    return panel

