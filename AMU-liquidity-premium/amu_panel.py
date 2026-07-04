from __future__ import annotations

from collections.abc import Iterable
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from amu_config import CONFIG, bootstrap_repo_root
from amu_cache import get_cached_frame, put_cached_frame
from tardis import package_set_log_level, test_utils as tu
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
pcp_metric_kwargs = {
    "cost_per_notional": float(pcp_cfg["cost_per_notional"]),
    "fut_mgn_rate": float(pcp_cfg["fut_mgn_rate"]),
    "short_put_mgn_rate": float(pcp_cfg["short_put_mgn_rate"]),
    "short_call_mgn_rate": float(pcp_cfg["short_call_mgn_rate"]),
    "r": float(pcp_cfg["r"]),
    "contract_size": float(pcp_cfg["contract_size"]),
}


def _bucket_midpoint(values: pl.Series, lower: float, upper: float, bucket_count: int, name: str) -> pl.Series:
    step = (upper - lower) / bucket_count
    midpoints: list[float | None] = []
    for value in values.to_list():
        if value is None:
            midpoints.append(None)
        else:
            midpoints.append(lower + (int(value) + 0.5) * step)
    return pl.Series(name, midpoints, dtype=pl.Float64)


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
        step = (upper - lower) / bucket_count
        raw = np.floor((numbers[valid] - lower) / step).astype(int)
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
    return _bucket_midpoint(index, tte_sqrt_min, tte_sqrt_max, tte_buckets, "tte_bucket")


def _required_panel_columns() -> list[str]:
    return [
        "timestamp",
        "exchange",
        "ref_sym",
        "rel_strike",
        "tte",
        "index",
        "amu_fwd_bp",
        "amu_bck_bp",
        "pcpb_fwd_bp",
        "pcpb_bck_bp",
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
    ]


def _panel_metrics() -> list[pl.Expr]:
    metrics = [pl.len().alias("n_obs")]
    metrics.extend(pl.mean(column).alias(f"mean_{column}") for column in mean_panel_columns)
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
    if panel_ready.is_empty():
        return pl.DataFrame()

    panel_ready = panel_ready.with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        _rel_strike_bucket(panel_ready),
        _tte_bucket(panel_ready),
    )

    group_keys = ["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"]
    return panel_ready.group_by(group_keys, maintain_order=True).agg(_panel_metrics())


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
) -> pl.DataFrame:
    """Build a compact panel from aligned put/call quote-trade-chain files.

    When from_date and to_date are provided, only files whose embedded day falls
    within the inclusive range are processed.
    """

    if cache_path is not None:
        cached_panel = get_cached_frame(cache_path, "amu_panel")
        if cached_panel is not None:
            logger.debug("Cache hit for build_amu_panel: %s", cache_path)
            return cached_panel

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

def test_build_amu_panel():
    df = build_amu_panel(from_date="2026-01-01", to_date="2026-01-05")
    tu.assert_df_equal(df.to_pandas().head(2).T, """
                                                0                    1
day                           2026-01-01 00:00:00  2026-01-01 00:00:00
exchange                                  deribit              deribit
ref_sym                                    BTCUSD               BTCUSD
rel_strike_bucket                        0.809524             0.809524
tte_bucket                                   0.85                 0.95
n_obs                                           7                  255
mean_strike                               75000.0              75000.0
mean_index                           88732.214286          87951.57451
mean_rel_strike                          0.817763             0.815763
mean_tte                                 0.728815              0.97938
mean_fut_mid_price                   91713.622857         91939.336039
mean_call_bid_price                      0.255786              0.25558
mean_call_ask_price                      0.258357              0.31191
mean_call_bid_amount                     6.971429            18.211765
mean_call_ask_amount                    11.371429            10.480392
mean_put_bid_price                       0.074214             0.095008
mean_put_ask_price                       0.075857             0.097482
mean_put_bid_amount                      5.157143            24.375686
mean_put_ask_amount                     26.242857              7.18549
mean_fut_bid_price                   91713.622857         91939.336039
mean_fut_ask_price                   91713.622857         91939.336039
mean_fut_bid_amount                           NaN                  NaN
mean_fut_ask_amount                           NaN                  NaN
mean_spot_bid_price                  88706.428571         87931.988235
mean_spot_ask_price                       88758.0         87971.160784
mean_spot_bid_amount                     0.019986             0.026654
mean_spot_ask_amount                     0.008071             0.020742
mean_call_open_interest                      76.9                  0.0
mean_put_open_interest                     2061.7            15.247059
mean_call_ask_price_xS               22931.278429         27437.494373
mean_call_bid_price_xS                  22689.844         22475.761704
mean_put_ask_price_xS                 6732.916929          8575.337359
mean_put_bid_price_xS                 6583.277429          8353.998067
mean_cost                                2.661966             2.638547
mean_capital_fwd                       780.694771           754.665265
mean_capital_bck                       388.984419           360.068186
mean_pcpb_forward                       -1.586028           -22.293997
mean_pcpb_backward                      -2.324711           -29.536722
mean_pcpb_fwd_real                      -4.247994           -24.932544
mean_pcpb_bck_real                      -4.986678            -32.17527
mean_pcpb_fwd_bp                       -20.314537          -303.260161
mean_pcpb_bck_bp                       -59.768447          -876.580484
mean_pcpb_fwd_real_bp                  -54.411955          -338.251272
mean_pcpb_bck_real_bp                 -128.202363           -950.23161
mean_pcpb_fwd_ann_bp                   -27.873398          -309.661961
mean_pcpb_bck_ann_bp                   -82.007842          -895.093107
mean_pcpb_fwd_real_ann_bp              -74.658154          -345.389792
mean_pcpb_bck_real_ann_bp             -175.905399          -970.295704
mean_call_opt_spread_bp                 27.208469           564.559606
mean_put_opt_spread_bp                  16.864415            25.172334
mean_bigger_opt_spread_bp               27.208469           564.559606
mean_smaller_opt_spread_bp              16.864415            25.172334
mean_amu_fwd_bp                        -47.873722          -283.666609
mean_amu_bck_bp                        -56.199161          -366.065331
mean_min_quote_size_dollar          286516.071429        431561.617059
mean_contract_size                           0.01                 0.01
sum_call_trade_amount                         0.0                  0.0
sum_call_trade_signed_amount                  0.0                  0.0
sum_call_trade_price_amount                   0.0                  0.0
sum_put_trade_amount                          0.0                  0.4
sum_put_trade_signed_amount                   0.0                 -0.2
sum_put_trade_price_amount                    0.0              0.03878
frac_call_stale                               0.0             0.227451
frac_put_stale                                0.0             0.027451
frac_spot_stale                               0.0                  0.0
""")

