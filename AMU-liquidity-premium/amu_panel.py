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
    tickpath_amu_agg_exprs_flavored,
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
# Absolute AMU cost grid (basis points): conditional/unconditional AMU precomputed
# per cell at each grid cost, so calculations can be redone at any cost without
# rebuilding from the tick frame. Grid costs are ABSOLUTE and independent of
# cost_per_notional. Because MMA = gross_wedge - c0_bp, evaluating AMU at cost c uses
# the offset delta = c - c0_bp, and (MMA - delta) = gross_wedge - c cancels c0_bp
# exactly -- so a grid cost of 20 always means AMU at 20 bp of the gross wedge,
# regardless of cost_per_notional. Grid is [start, end, step], end inclusive.
c0_bp = float(pcp_cfg["cost_per_notional"]) * 1.0e4
# cost_grid is in fractions of notional (like cost_per_notional); convert to integer
# basis points for the offset math and the per-cost column names.
_cost_grid = list(pcp_cfg.get("cost_grid", [0.0005, 0.005, 0.0005]))
_cg_bp = [int(round(float(x) * 1.0e4)) for x in _cost_grid]  # [start_bp, end_bp, step_bp]
COST_GRID_BP = list(range(_cg_bp[0], _cg_bp[1] + 1, _cg_bp[2]))
pcp_metric_kwargs = {
    "cost_per_notional": float(pcp_cfg["cost_per_notional"]),
    "cost_per_option_value": float(pcp_cfg.get("cost_per_option_value", 0.0)),
    "flat_dollar_amount": float(pcp_cfg.get("flat_dollar_amount", 0.0)),
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
        "mma_bck_bp",
        "mma_fwd_bp",
        "pcpb_bck_bp",
        "pcpb_fwd_bp",
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
    # AMU accumulators (shared definition in amu_metrics): the positive-part sum and
    # the count of positive tickpaths across the four paths. n_obs (the tick count)
    # is the denominator for the unconditional flavour; both mean_amu_conditional_bp
    # and mean_amu_unconditional_bp are formed as ratios after aggregation below.
    metrics.extend(
        tickpath_amu_agg_exprs_flavored(
            max_amu_bp,
            sum_alias="sum_amu_tickpath_bp",
            count_alias="num_amu_tickpath",
            sum_dollar_alias="sum_amu_tickpath_dollar",
            sum_capital_alias="sum_amu_tickpath_capital_bp",
        )
    )
    # Cost-grid AMU accumulators: for each grid cost c, the positive-part sum and the
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


# Cell-day liquidity measures, computed directly on the cell's intraday series
# (cell = exchange x underlying x strike bucket x maturity bucket, per day). Within a
# cell-day we aggregate ticks to one observation per 5-min timestamp -- the "cell-tick":
# mean call mid, mean put mid, mean quoted spread (over pairs and both sides), and the
# summed call and put traded size -- giving a ~288-long (12 per hour x 24 h) intraday
# series per cell-day, plus the mean index. From these we build four measures:
#   * amihud_shares / amihud_dollar -- |return| per unit traded, averaged over the pooled
#     call and put cell-tick series, with volume measured in traded contracts (shares) or
#     in dollars (contracts x index; the *index*, not the option price);
#   * spread_bp_recovery / spread_dollar_recovery -- the event-conditioned resiliency
#     (Kyle's third dimension of liquidity) of the bp spread and of the dollar spread
#     (bp spread x index / 1e4): the AR(1) decay half-life of the spread's deviation from
#     a rolling ~1-hour baseline, fit only on the windows after a large widening, so it
#     captures how fast the book heals rather than the near-unit-root drift of the level
#     (see _spread_resiliency; cf. Degryse et al. 2005, Large 2007).
# Robust to missing trade columns: without trades the two Amihud measures are null and
# only the recoveries are produced.
_CELL_DAY_LIQ_COLUMNS = ("amihud_shares", "amihud_dollar", "amihud_capital", "spread_bp_recovery", "spread_dollar_recovery")
_LN2 = 0.6931471805599453
_BIN_MINUTES = 5.0  # cell-tick spacing; spread recovery is reported in minutes
_RECOVERY_BASELINE_BINS = 12  # centered rolling-baseline window (~1 hour) for the deviation
_MAD_TO_STD = 1.4826  # MAD -> robust standard-deviation scaling
_EVENT_K = 2.0  # a widening = deviation above k robust-std of the local baseline
_EVENT_HORIZON_BINS = 12  # post-widening decay window (~1 hour) the AR(1) is fit on
_EVENT_MIN = 3  # min widening events per cell-day for a recovery estimate
_EVENT_MIN_PAIRS = 10  # min post-widening AR pairs for a recovery estimate


def _cell_day_liquidity(panel_ready: pl.DataFrame) -> pl.DataFrame:
    cell_keys = ["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"]
    cols = set(panel_ready.columns)
    need = {
        "call_bid_price_xS", "call_ask_price_xS", "put_bid_price_xS", "put_ask_price_xS",
        "call_opt_spread_bp", "put_opt_spread_bp", "timestamp",
    } | set(cell_keys)
    if not need <= cols:
        base = panel_ready.select(cell_keys).unique()
        return base.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in _CELL_DAY_LIQ_COLUMNS])

    have_trades = {"call_trade_amount", "put_trade_amount"} <= cols
    cvol = pl.col("call_trade_amount").cast(pl.Float64).fill_null(0.0) if have_trades else pl.lit(0.0)
    pvol = pl.col("put_trade_amount").cast(pl.Float64).fill_null(0.0) if have_trades else pl.lit(0.0)
    df = panel_ready.with_columns(
        (0.5 * (pl.col("call_bid_price_xS") + pl.col("call_ask_price_xS"))).alias("_cmid"),
        (0.5 * (pl.col("put_bid_price_xS") + pl.col("put_ask_price_xS"))).alias("_pmid"),
        (0.5 * (pl.col("call_opt_spread_bp").cast(pl.Float64) + pl.col("put_opt_spread_bp").cast(pl.Float64))).alias("_spr"),
        cvol.alias("_cvol"),
        pvol.alias("_pvol"),
    )
    # Cell-tick: one observation per (cell, 5-min timestamp).
    bins = df.group_by(cell_keys + ["timestamp"]).agg(
        pl.col("_cmid").mean().alias("cmid"),
        pl.col("_pmid").mean().alias("pmid"),
        pl.col("_spr").mean().alias("spr"),
        pl.col("index").mean().alias("index"),
        pl.col("_cvol").sum().alias("cvol"),
        pl.col("_pvol").sum().alias("pvol"),
    ).sort(cell_keys + ["timestamp"])
    # Reindex each cell-day onto a strict 5-min grid, so a one-step lag is exactly one
    # 5-min slot: intraday gaps become null rows and are dropped pairwise below (no
    # spurious persistence or multi-bin returns across a gap). Missing slots carry no
    # volume. This makes both the Amihud returns and the AR(1) spread lag true 5-min steps.
    bins = bins.upsample(time_column="timestamp", every=f"{int(_BIN_MINUTES)}m", group_by=cell_keys, maintain_order=True)
    bins = bins.with_columns(
        pl.col("cvol").fill_null(0.0),
        pl.col("pvol").fill_null(0.0),
    ).with_columns(
        (pl.col("cmid") / pl.col("cmid").shift(1).over(cell_keys) - 1.0).alias("cret"),
        (pl.col("pmid") / pl.col("pmid").shift(1).over(cell_keys) - 1.0).alias("pret"),
        # Dollar spread cell-tick series: the bp spread rescaled by the index level
        # (spread_bp = (ask-bid)/index * 1e4, so this recovers the quoted dollar width).
        (pl.col("spr") / 10000.0 * pl.col("index")).alias("spr_dollar"),
    )
    # Amihud, pooled over call and put, in two units: per traded contract (shares) and
    # per dollar of volume (contracts x index -- multiplied by the *index*, not the
    # option price). amihud_dollar = amihud_shares / index up to within-bin variation.
    call_ok = (pl.col("cvol") > 0) & pl.col("cret").is_not_null()
    put_ok = (pl.col("pvol") > 0) & pl.col("pret").is_not_null()
    # Option capital per cell-tick: the sum of the call and put mid prices (K), used as the
    # denominator scaler for the capital-normalized Amihud (volume in premium dollars,
    # contracts x (call+put mid), rather than notional dollars, contracts x index).
    _cap = (pl.col("cmid") + pl.col("pmid"))
    amihud = bins.group_by(cell_keys).agg(
        (pl.col("cret").abs() / pl.col("cvol")).filter(call_ok).sum().alias("_as_c"),
        (pl.col("pret").abs() / pl.col("pvol")).filter(put_ok).sum().alias("_as_p"),
        (pl.col("cret").abs() / (pl.col("cvol") * pl.col("index"))).filter(call_ok).sum().alias("_ad_c"),
        (pl.col("pret").abs() / (pl.col("pvol") * pl.col("index"))).filter(put_ok).sum().alias("_ad_p"),
        (pl.col("cret").abs() / (pl.col("cvol") * _cap)).filter(call_ok).sum().alias("_ac_c"),
        (pl.col("pret").abs() / (pl.col("pvol") * _cap)).filter(put_ok).sum().alias("_ac_p"),
        call_ok.sum().alias("_n_c"),
        put_ok.sum().alias("_n_p"),
    ).with_columns(
        pl.when((pl.col("_n_c") + pl.col("_n_p")) > 0)
        .then((pl.col("_as_c") + pl.col("_as_p")) / (pl.col("_n_c") + pl.col("_n_p")))
        .otherwise(None).alias("amihud_shares"),
        pl.when((pl.col("_n_c") + pl.col("_n_p")) > 0)
        .then((pl.col("_ad_c") + pl.col("_ad_p")) / (pl.col("_n_c") + pl.col("_n_p")))
        .otherwise(None).alias("amihud_dollar"),
        pl.when((pl.col("_n_c") + pl.col("_n_p")) > 0)
        .then((pl.col("_ac_c") + pl.col("_ac_p")) / (pl.col("_n_c") + pl.col("_n_p")))
        .otherwise(None).alias("amihud_capital"),
    ).select(cell_keys + ["amihud_shares", "amihud_dollar", "amihud_capital"])

    return (
        amihud
        .join(_spread_resiliency(bins, cell_keys, "spr", "spread_bp_recovery"), on=cell_keys, how="left")
        .join(_spread_resiliency(bins, cell_keys, "spr_dollar", "spread_dollar_recovery"), on=cell_keys, how="left")
    )


def _spread_resiliency(bins: pl.DataFrame, cell_keys: list[str], spread_col: str, out_name: str) -> pl.DataFrame:
    """Event-conditioned resiliency of ``spread_col``: the AR(1) decay half-life (minutes)
    of its deviation from a centered rolling ~1-hour baseline, fit only on the windows
    that follow a *large* widening -- a deviation up-crossing beyond ``_EVENT_K`` robust
    standard deviations (from the daily MAD) of the baseline. This keys on genuine shocks
    and ignores the small idiosyncratic wiggles that make an unconditional AR(1) near
    unit-root (cf. Degryse et al. 2005; Large 2007)."""
    b = bins.with_columns(
        (
            pl.col(spread_col)
            - pl.col(spread_col).rolling_median(window_size=_RECOVERY_BASELINE_BINS, min_samples=3, center=True).over(cell_keys)
        ).alias("_dev")
    ).with_columns(
        pl.col("_dev").shift(1).over(cell_keys).alias("_devlag"),
    )
    sigma = b.group_by(cell_keys).agg(
        (_MAD_TO_STD * (pl.col("_dev") - pl.col("_dev").median()).abs().median()).alias("_sigma")
    )
    b = b.join(sigma, on=cell_keys, how="left").with_columns(
        (pl.col("_dev") > _EVENT_K * pl.col("_sigma")).fill_null(False).alias("_above"),
    ).with_columns(
        # A widening *onset*: the deviation crosses up through the +k*sigma band.
        (pl.col("_above") & ~pl.col("_above").shift(1).over(cell_keys).fill_null(False)).alias("_event"),
    ).with_columns(
        # In a post-widening decay window if a widening began within the last H slots.
        (
            pl.col("_event").cast(pl.Int32)
            .rolling_sum(window_size=_EVENT_HORIZON_BINS + 1, min_samples=1).over(cell_keys) >= 1
        ).alias("_active"),
    )
    m = pl.col("_active") & pl.col("_dev").is_not_null() & pl.col("_devlag").is_not_null()
    res = b.group_by(cell_keys).agg(
        pl.col("_event").sum().alias("_n_ev"),
        m.sum().alias("_n"),
        pl.col("_devlag").filter(m).sum().alias("_sx"),
        pl.col("_dev").filter(m).sum().alias("_sy"),
        (pl.col("_devlag") ** 2).filter(m).sum().alias("_sxx"),
        (pl.col("_dev") * pl.col("_devlag")).filter(m).sum().alias("_sxy"),
    ).with_columns(
        # OLS AR(1) decay slope rho on the post-widening bins (cov/var of the deviation).
        (
            (pl.col("_sxy") - pl.col("_sx") * pl.col("_sy") / pl.col("_n"))
            / (pl.col("_sxx") - pl.col("_sx") ** 2 / pl.col("_n"))
        ).alias("_rho")
    ).with_columns(
        # Half-life = -ln2/ln(rho) in 5-min steps -> minutes; rho clipped into (0, 1).
        # rho<=0 (immediate reversion) maps to a near-zero recovery. Null unless the cell
        # has enough widenings and post-widening pairs to fit the decay.
        pl.when((pl.col("_n_ev") >= _EVENT_MIN) & (pl.col("_n") >= _EVENT_MIN_PAIRS))
        .then((-_LN2 * _BIN_MINUTES) / pl.col("_rho").clip(1e-6, 0.999999).log())
        .otherwise(None)
        .alias(out_name)
    )
    return res.select(cell_keys + [out_name])


def _panel_block_from_file(file_path: Path) -> pl.DataFrame:
    raw = pl.read_parquet(file_path)
    if raw.is_empty():
        logger.debug("Skipping empty aligned file: %s", file_path)
        return pl.DataFrame()

    panel_ready = compute_pcp_metrics(raw, **pcp_metric_kwargs)
    panel_ready = panel_ready.drop_nulls(_required_panel_columns())
    # Apply the tick-level filters up front, so the aggregated panel (and therefore
    # the regressions, the cost/r tables, and every panel figure) is built from the
    # exact same sample as the descriptive tick-frame figures. filter_stale stays at
    # its config default (off) so the stale-quote fraction remains a usable regressor
    # and the robustness grid can still toggle it. Deferred import avoids a cycle.
    from amu_statistics import filter_ticks
    panel_ready = filter_ticks(panel_ready)
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
    # Both explicit AMU flavours per cell at the baseline cost: conditional (over
    # positive tickpaths) and unconditional (over all tickpaths = frequency x size).
    ratio_columns = [
        amu_conditional_bp_expr("sum_amu_tickpath_bp", "num_amu_tickpath").alias("mean_amu_conditional_bp"),
        amu_unconditional_bp_expr("sum_amu_tickpath_bp", "n_obs").alias("mean_amu_unconditional_bp"),
        # Dollar (per one notional) and capital-bp flavours of the same wedges.
        amu_conditional_bp_expr("sum_amu_tickpath_dollar", "num_amu_tickpath").alias("mean_amu_conditional_dollar"),
        amu_unconditional_bp_expr("sum_amu_tickpath_dollar", "n_obs").alias("mean_amu_unconditional_dollar"),
        amu_conditional_bp_expr("sum_amu_tickpath_capital_bp", "num_amu_tickpath").alias(
            "mean_amu_conditional_capital_bp"
        ),
        amu_unconditional_bp_expr("sum_amu_tickpath_capital_bp", "n_obs").alias("mean_amu_unconditional_capital_bp"),
    ]
    # Same two flavours at each grid cost, derived from the per-cost accumulators.
    for c in COST_GRID_BP:
        possum, poscount = f"amu_possum_bp_c{c:02d}", f"amu_poscount_c{c:02d}"
        ratio_columns.append(amu_conditional_bp_expr(possum, poscount).alias(f"mean_amu_cond_bp_c{c:02d}"))
        ratio_columns.append(amu_unconditional_bp_expr(possum, "n_obs").alias(f"mean_amu_uncond_bp_c{c:02d}"))
    block = block.with_columns(ratio_columns)
    # Attach the cell-day liquidity measures (Amihud, spread recovery).
    liquidity = _cell_day_liquidity(panel_ready)
    return block.join(
        liquidity, on=["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"], how="left"
    )


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

