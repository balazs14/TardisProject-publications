from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns

from amu_config import CONFIG, bootstrap_repo_root

PROJECT_ROOT = bootstrap_repo_root(Path(__file__).resolve())

from tardis.utils import debug_runtime
from tardis.process_pcp import compute_pcp_metrics
from amu_cache import get_cached_frame_parquet_path, put_cached_frame_parquet_path


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

PUBLICATION_DIR = Path(__file__).resolve().parent

filters_cfg = CONFIG["filters"]
statistics_cfg = CONFIG["statistics"]
pcp_cfg = CONFIG["pcp"]

rel_strike_min_default = float(filters_cfg["rel_strike_min"])
rel_strike_max_default = float(filters_cfg["rel_strike_max"])
spread_bp_max_default = int(filters_cfg["spread_bp_max"])
min_quote_size_dollar_default = float(filters_cfg["min_quote_size_dollar"])
max_amu_bp_default = int(filters_cfg["max_amu_bp"])

pcpb_columns = list(statistics_cfg["pcpb_columns"])
pcp_metric_kwargs = {
    "cost_per_notional": float(pcp_cfg["cost_per_notional"]),
    "fut_mgn_rate": float(pcp_cfg["fut_mgn_rate"]),
    "short_put_mgn_rate": float(pcp_cfg["short_put_mgn_rate"]),
    "short_call_mgn_rate": float(pcp_cfg["short_call_mgn_rate"]),
    "r": float(pcp_cfg["r"]),
    "contract_size": float(pcp_cfg["contract_size"]),
}


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


def _aligned_chain_files(
    exchange: str,
    sample_freq: str,
    raw_data_dir: str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[Path]:
    root = PROJECT_ROOT / raw_data_dir.format(exchange=exchange)
    pattern = f"{exchange}_aligned_put_call_quotes_trades_chain_*_{sample_freq}.parquet"
    files = []
    for file_path in sorted(root.glob(pattern)):
        file_day = _parse_file_day(file_path)
        if file_day is None or _day_in_range(file_day, from_date, to_date):
            files.append(file_path)
    return files

def _ticks_filter_expr(
    *,
    rel_strike_min: float,
    rel_strike_max: float,
    spread_bp_max: int,
    min_quote_size_dollar: float,
    apply_rel_strike: bool,
) -> pl.Expr:
    expr = (
        pl.col("call_opt_spread_bp").is_between(0, spread_bp_max)
        & pl.col("put_opt_spread_bp").is_between(0, spread_bp_max)
        & pl.col("min_quote_size_dollar").is_not_null()
        & (pl.col("min_quote_size_dollar") >= min_quote_size_dollar)
    )
    if apply_rel_strike:
        expr = expr & pl.col("rel_strike").is_between(rel_strike_min, rel_strike_max)
    return expr


def filter_ticks(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    rel_strike_min: float = rel_strike_min_default,
    rel_strike_max: float = rel_strike_max_default,
    spread_bp_max: int = spread_bp_max_default,
    min_quote_size_dollar: float = min_quote_size_dollar_default,
    *,
    apply_rel_strike: bool = True,
) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
    required = {
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
    }
    if apply_rel_strike:
        required.add("rel_strike")

    if isinstance(df, pl.LazyFrame):
        column_names = set(df.collect_schema().names())
    else:
        column_names = set(df.columns)

    missing = required - column_names
    assert not missing, f"Missing filter columns: {sorted(missing)}"

    if isinstance(df, pd.DataFrame):
        mask = (
            pd.to_numeric(df["call_opt_spread_bp"], errors="coerce").between(0, spread_bp_max)
            & pd.to_numeric(df["put_opt_spread_bp"], errors="coerce").between(0, spread_bp_max)
            & pd.to_numeric(df["min_quote_size_dollar"], errors="coerce").ge(min_quote_size_dollar)
        )
        if apply_rel_strike:
            mask = mask & pd.to_numeric(df["rel_strike"], errors="coerce").between(rel_strike_min, rel_strike_max)
        return df.loc[mask].copy()

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df.filter(
            _ticks_filter_expr(
                rel_strike_min=rel_strike_min,
                rel_strike_max=rel_strike_max,
                spread_bp_max=spread_bp_max,
                min_quote_size_dollar=min_quote_size_dollar,
                apply_rel_strike=apply_rel_strike,
            )
        )

    raise TypeError(f"Unsupported frame type: {type(df)}")


def _mark_up_pcpb(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("amu_fwd_bp") + pl.col("call_opt_spread_bp")).alias("fwd_call"),
        (pl.col("amu_fwd_bp") + pl.col("put_opt_spread_bp")).alias("fwd_put"),
        (pl.col("amu_bck_bp") + pl.col("call_opt_spread_bp")).alias("bck_call"),
        (pl.col("amu_bck_bp") + pl.col("put_opt_spread_bp")).alias("bck_put"),
    )


def _pcpb_block_from_file(file_path: Path) -> pl.DataFrame:
    raw = pl.read_parquet(file_path)
    if raw.is_empty():
        return pl.DataFrame()

    pcpb = compute_pcp_metrics(raw, **pcp_metric_kwargs)
    if pcpb.is_empty():
        return pl.DataFrame()

    return _mark_up_pcpb(pcpb).select(pcpb_columns)


def _cache_frame_path(cache_path: str | Path, key: str) -> Path:
    cache_meta = Path(cache_path)
    base_name = cache_meta.name.removesuffix("_meta.pkl")
    return cache_meta.parent / f"{base_name}__{key}.parquet"


def _build_amu_statistics_frame_to_cache(
    *,
    cache_path: str | Path,
    key: str,
    exchanges: Iterable[str],
    sample_freq: str,
    raw_data_dir: str,
    from_date: str | None,
    to_date: str | None,
) -> Path:
    frame_path = _cache_frame_path(cache_path, key)
    tmp_path = frame_path.with_suffix(frame_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_cols = 0
    try:
        for exchange in exchanges:
            for file_path in _aligned_chain_files(
                exchange,
                sample_freq,
                raw_data_dir,
                from_date=from_date,
                to_date=to_date,
            ):
                logger.debug(f"reading file {file_path}")
                block = _pcpb_block_from_file(file_path)
                if block.is_empty():
                    continue
                table = block.to_arrow()
                if writer is None:
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(str(tmp_path), table.schema, compression="zstd")
                writer.write_table(table)
                total_rows += block.height
                total_cols = block.width

        if writer is None:
            empty = pl.DataFrame(schema={column: pl.Null for column in pcpb_columns})
            empty.write_parquet(tmp_path)
            total_rows = 0
            total_cols = len(pcpb_columns)
        else:
            writer.close()
            writer = None

        tmp_path.replace(frame_path)
        put_cached_frame_parquet_path(
            cache_path,
            key,
            frame_path=frame_path,
            n_rows=total_rows,
            n_cols=total_cols,
            from_date=from_date,
            to_date=to_date,
        )
        return frame_path
    finally:
        if writer is not None:
            writer.close()


@debug_runtime("build_amu_statistics_frame_cached_path")
def build_amu_statistics_frame_cached_path(
    *,
    cache_path: str | Path,
    exchanges: Iterable[str] = ("okex", "deribit"),
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/{exchange}/",
    from_date: str | None = None,
    to_date: str | None = None,
) -> Path:
    key = "amu_statistics_frame"
    cached = get_cached_frame_parquet_path(cache_path, key)
    if cached is not None:
        logger.debug("Cache hit for build_amu_statistics_frame_cached_path: %s", cache_path)
        return cached

    logger.debug("Cache miss for build_amu_statistics_frame_cached_path: %s", cache_path)
    return _build_amu_statistics_frame_to_cache(
        cache_path=cache_path,
        key=key,
        exchanges=exchanges,
        sample_freq=sample_freq,
        raw_data_dir=raw_data_dir,
        from_date=from_date,
        to_date=to_date,
    )


def inspect_pcpb_input(df: pl.DataFrame) -> None:
    logger.info("AMU columns: %s", ", ".join(df.columns))
    logger.info("AMU ref_syms: %s", df.select("ref_sym").unique().sort("ref_sym").to_series().to_list())
    logger.info("AMU exchanges: %s", df.select("exchange").unique().sort("exchange").to_series().to_list())
    min_ts, max_ts = df.select(
        pl.col("timestamp").min().alias("min_ts"),
        pl.col("timestamp").max().alias("max_ts"),
    ).row(0)
    logger.info("AMU timestamp range: %s -> %s", min_ts, max_ts)


def calculate_amu_bps(pcpb: pd.DataFrame, *, max_bp_cutoff: int = max_amu_bp_default) -> tuple[float, int, int]:
    if pcpb.empty:
        return np.nan, 0, 0

    fwd_call_bp = float(pcpb["fwd_call"].clip(lower=0, upper=max_bp_cutoff).mean())
    bck_call_bp = float(pcpb["bck_call"].clip(lower=0, upper=max_bp_cutoff).mean())
    bck_put_bp = float(pcpb["bck_put"].clip(lower=0, upper=max_bp_cutoff).mean())
    fwd_put_bp = float(pcpb["fwd_put"].clip(lower=0, upper=max_bp_cutoff).mean())

    fwd_call_num = int((pcpb["fwd_call"] > 0).sum())
    bck_call_num = int((pcpb["bck_call"] > 0).sum())
    bck_put_num = int((pcpb["bck_put"] > 0).sum())
    fwd_put_num = int((pcpb["fwd_put"] > 0).sum())
    amu_num = int(((pcpb["fwd_put"] > 0) | (pcpb["bck_put"] > 0) | (pcpb["fwd_call"] > 0) | (pcpb["bck_call"] > 0)).sum())

    weighted_sum = (
        fwd_call_bp * fwd_call_num
        + bck_call_bp * bck_call_num
        + bck_put_bp * bck_put_num
        + fwd_put_bp * fwd_put_num
    )
    weighted_avg = weighted_sum / amu_num if amu_num > 0 else np.nan
    return weighted_avg, amu_num, len(pcpb)


@debug_runtime("summarize_timeslice_option_coverage")
def summarize_timeslice_option_coverage(df: pl.DataFrame) -> pd.DataFrame:
    filtered = filter_ticks(df)
    if filtered.is_empty():
        return pd.DataFrame()

    strikes_frame = filter_ticks(df, apply_rel_strike=False)

    group_keys = ["ref_sym", "exchange", "timestamp"]
    per_ts = filtered.group_by(group_keys).agg(
        pl.len().alias("num_contracts_ts"),
        pl.col("exp").n_unique().alias("num_expirations_ts"),
        pl.col("call_opt_spread_bp").mean().alias("call_spread_bp_ts"),
        pl.col("put_opt_spread_bp").mean().alias("put_spread_bp_ts"),
    )
    strikes_per_ts = strikes_frame.group_by(group_keys).agg(
        pl.col("strike").n_unique().alias("num_strikes_ts"),
    )
    per_ts = per_ts.join(strikes_per_ts, on=group_keys, how="left")
    n_days = filtered.group_by(["ref_sym", "exchange"]).agg(pl.col("mdy").n_unique().alias("Num days in sample"))
    summary = (
        per_ts.group_by(["ref_sym", "exchange"])
        .agg(
            pl.col("num_contracts_ts").mean().alias("Avg num contracts"),
            pl.col("num_expirations_ts").mean().alias("Avg num expirations"),
            pl.col("num_strikes_ts").mean().alias("Avg num strikes"),
            pl.col("call_spread_bp_ts").mean().alias("Avg call spread (bp)"),
            pl.col("put_spread_bp_ts").mean().alias("Avg put spread (bp)"),
        )
        .join(n_days, on=["ref_sym", "exchange"], how="left")
        .sort(["ref_sym", "exchange"])
    )
    return summary.to_pandas()


def dataframe_to_tabular_tex(df: pd.DataFrame, path: Path) -> None:
    latex = df.to_latex(
        index=True,
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        na_rep="",
        bold_rows=False,
    )
    path.write_text(latex, encoding="utf-8")


@debug_runtime("write_summary_daily_table")
def write_summary_daily_table(df: pl.DataFrame, *, output_dir: Path) -> Path:
    summary = summarize_timeslice_option_coverage(df)
    summary_rounded = summary.copy()
    for column in summary_rounded.columns:
        if column in {
            "Avg num contracts",
            "Avg num expirations",
            "Avg num strikes",
            "Num days in sample",
            "Avg call spread (bp)",
            "Avg put spread (bp)",
        }:
            summary_rounded[column] = pd.to_numeric(summary_rounded[column], errors="coerce").round(0).astype("Int64")
    summary_t = summary_rounded.set_index(["ref_sym", "exchange"]).T
    summary_t.columns = summary_t.columns.set_names([None, None])
    output_path = output_dir / "summary_daily_option_coverage_table.tex"
    dataframe_to_tabular_tex(summary_t, output_path)
    return output_path


@debug_runtime("write_amu_summary_table")
def write_amu_summary_table(pcpb: pd.DataFrame, *, output_dir: Path) -> Path:
    filtered = filter_ticks(pcpb)
    rows = []
    for (ref_sym, exchange), group in filtered.groupby(["ref_sym", "exchange"], sort=True):
        amu_bps, amu_num, num_pairs = calculate_amu_bps(group)
        rows.append(
            {
                "ref_sym": ref_sym,
                "exchange": exchange,
                "AMU(bps)": amu_bps,
                "MU occurs": amu_num,
                "Num Observations": num_pairs,
            }
        )

    table = pd.DataFrame(rows).set_index(["ref_sym", "exchange"]).rename_axis(index=["underlying", "exchange"])
    table["AMU(bps)"] = table["AMU(bps)"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    for column in ["MU occurs", "Num Observations"]:
        table[column] = table[column].map(lambda value: f"{int(value):,}" if pd.notna(value) else "")

    output_path = output_dir / "amu_summary_table.tex"
    dataframe_to_tabular_tex(table, output_path)
    return output_path


def _safe_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace("/", "_")


def _gaussian_kernel_smooth(x: np.ndarray, y: np.ndarray, *, bandwidth: float = 0.02) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.size == 0:
        return out
    for index, x0 in enumerate(x):
        if not np.isfinite(x0):
            continue
        weights = np.exp(-0.5 * ((x_valid - x0) / bandwidth) ** 2)
        weight_sum = weights.sum()
        if weight_sum > 0:
            out[index] = np.dot(weights, y_valid) / weight_sum
    return out


@debug_runtime("plot_4_spreads")
def plot_4_spreads(pcpb: pd.DataFrame, *, output_dir: Path, rng: int = 100) -> list[Path]:
    filtered = filter_ticks(pcpb)
    output_paths: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")
    for (exchange, ref_sym), subset in filtered.groupby(["exchange", "ref_sym"], sort=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        visible_max = 0.0
        for column, color in zip(["fwd_call", "fwd_put", "bck_call", "bck_put"], sns.color_palette("deep", n_colors=4), strict=False):
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            counts, edges = np.histogram(values, bins=100, range=(-rng, rng), density=True)
            visible_max = max(visible_max, float(np.nanmax(counts)) if counts.size else 0.0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            x_fine = np.linspace(-rng, rng, 800)
            y_fine = np.interp(x_fine, centers, counts, left=0.0, right=0.0)
            ax.plot(x_fine[x_fine < 0], y_fine[x_fine < 0], color=color, linestyle="--", linewidth=1.6, alpha=0.9)
            ax.plot(x_fine[x_fine >= 0], y_fine[x_fine >= 0], color=color, linestyle="-", linewidth=2.6, alpha=1.0, label=column)

        amu_text, _, _ = calculate_amu_bps(subset)
        ax.axvline(0, color="black", linestyle="-", linewidth=3.0, alpha=0.8)
        ax.set_xlim(-rng, rng)
        ax.set_ylim(0, visible_max * 1.05 if visible_max > 0 else 1.0)
        ax.set_xlabel("AMU bp")
        ax.set_ylabel("Density")
        ax.set_title(f"{exchange} {ref_sym}")
        ax.legend(title="Spread")
        ax.text(0.75, 0.5, f"mean AMU = {amu_text:.1f} bp", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")
        ax.text(0.25, 0.75, "no trading opportunities", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")

        output_path = output_dir / f"{_safe_name(exchange)}_{_safe_name(ref_sym)}_4_spreads.pdf"
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.show()
        print(pd.Series(values).quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
        plt.close(fig)
        output_paths.append(output_path)
        breakpoint()
    return output_paths


@debug_runtime("plot_amu_bps_by_date")
def plot_amu_bps_by_date(pcpb: pd.DataFrame, *, output_dir: Path) -> Path:
    filtered = filter_ticks(pcpb)
    daily_amu = (
        filtered.assign(market=lambda frame: frame["exchange"].astype(str) + " | " + frame["ref_sym"].astype(str))
        .groupby(["mdy", "exchange", "ref_sym", "market"], sort=True)
        .apply(lambda group: pd.Series(calculate_amu_bps(group), index=["amu_bps", "num_amu", "num_pairs"]))
        .reset_index()
        .sort_values(["market", "mdy"])
    )
    btc_index = (
        filtered.loc[filtered["ref_sym"] == "BTCUSD", ["mdy", "index"]]
        .groupby("mdy", sort=True)["index"]
        .mean()
        .reset_index()
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=daily_amu, x="mdy", y="amu_bps", hue="market", linewidth=2.0, marker=None, ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(data=btc_index, x="mdy", y="index", color="black", linewidth=2.6, linestyle="--", ax=ax2, label="BTCUSD index")
    ax2.grid(False)
    ax2.set_ylabel("BTCUSD index price")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", title="")
    if ax2.legend_ is not None:
        ax2.legend_.remove()

    ax.set_xlabel("Date")
    ax.set_ylabel("AMU bps")
    ax.set_title("AMU by Date with BTCUSD Index")
    ax.tick_params(axis="x", labelrotation=90)

    output_path = output_dir / "multi_exchange_amu_bps_by_date.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("plot_amu_bps_by_rel_strike")
def plot_amu_bps_by_rel_strike(pcpb: pd.DataFrame, *, output_dir: Path) -> Path:
    filtered = filter_ticks(pcpb)
    plot_df = (
        filtered.dropna(subset=["exchange", "ref_sym", "rel_strike"])
        .assign(
            rel_strike=lambda frame: pd.to_numeric(frame["rel_strike"], errors="coerce"),
            market=lambda frame: frame["exchange"].astype(str) + " | " + frame["ref_sym"].astype(str),
        )
        .dropna(subset=["rel_strike"])
        .loc[lambda frame: frame["rel_strike"].between(0.3, 3.0)]
        .assign(rel_strike_bin=lambda frame: (frame["rel_strike"] / 0.01).round() * 0.01)
    )
    curve_df = (
        plot_df.groupby(["market", "rel_strike_bin"], sort=True)
        .apply(lambda group: pd.Series(calculate_amu_bps(group), index=["amu_bps", "num_amu", "num_pairs"]))
        .reset_index()
        .sort_values(["market", "rel_strike_bin"])
    )
    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["rel_strike_bin"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "rel_strike_bin", "num_amu", "num_pairs"]].melt(
        id_vars=["market", "rel_strike_bin"],
        value_vars=["num_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_amu": "Num MU ticks in bin", "num_pairs": "Num ticks in bin"})
    counts_long["count_smooth"] = np.nan
    for (market, metric), group in counts_long.groupby(["market", "metric"], sort=False):
        counts_long.loc[group.index, "count_smooth"] = _gaussian_kernel_smooth(
            group["rel_strike_bin"].to_numpy(),
            group["count"].to_numpy(),
        )

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    sns.lineplot(data=curve_df, x="rel_strike_bin", y="amu_bps_smooth", hue="market", linewidth=2.0, palette="deep", ax=ax_top)
    ax_top.set_ylabel("AMU bps (smoothed)")
    ax_top.set_title("AMU vs Relative Strike (all exchange/ref_sym)")
    ax_top.legend(title="", loc="best")
    sns.lineplot(
        data=counts_long,
        x="rel_strike_bin",
        y="count_smooth",
        hue="market",
        style="metric",
        linewidth=1.8,
        palette="deep",
        dashes={"Num MU ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Relative strike (1% bins)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num MU ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_rel_strike.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("plot_amu_bps_by_tte")
def plot_amu_bps_by_tte(pcpb: pd.DataFrame, *, output_dir: Path) -> Path:
    filtered = filter_ticks(pcpb)
    tmp = (
        filtered[["exchange", "ref_sym", "tte", "fwd_call", "bck_call", "bck_put", "fwd_put"]]
        .assign(
            tte=lambda frame: pd.to_numeric(frame["tte"], errors="coerce"),
            market=lambda frame: frame["exchange"].astype(str) + " | " + frame["ref_sym"].astype(str),
        )
        .dropna(subset=["tte", "market"])
        .loc[lambda frame: frame["tte"].between(0.0, 0.7)]
        .copy()
    )
    tmp["tte_bin"] = pd.cut(tmp["tte"], bins=np.linspace(0.0, 0.7, 101), include_lowest=True)
    for column in ["fwd_call", "bck_call", "bck_put", "fwd_put"]:
        tmp[f"{column}_clip"] = tmp[column].clip(lower=0, upper=max_amu_bp_default)
        tmp[f"{column}_pos"] = (tmp[column] > 0).astype(int)

    grouped = (
        tmp.groupby(["market", "tte_bin"], sort=True)
        .agg(
            fwd_call_bp=("fwd_call_clip", "mean"),
            bck_call_bp=("bck_call_clip", "mean"),
            bck_put_bp=("bck_put_clip", "mean"),
            fwd_put_bp=("fwd_put_clip", "mean"),
            fwd_call_num=("fwd_call_pos", "sum"),
            bck_call_num=("bck_call_pos", "sum"),
            bck_put_num=("bck_put_pos", "sum"),
            fwd_put_num=("fwd_put_pos", "sum"),
            num_pairs=("tte", "size"),
        )
        .reset_index()
    )
    grouped["num_amu"] = grouped["fwd_call_num"] + grouped["bck_call_num"] + grouped["bck_put_num"] + grouped["fwd_put_num"]
    grouped["weighted_sum"] = (
        grouped["fwd_call_bp"] * grouped["fwd_call_num"]
        + grouped["bck_call_bp"] * grouped["bck_call_num"]
        + grouped["bck_put_bp"] * grouped["bck_put_num"]
        + grouped["fwd_put_bp"] * grouped["fwd_put_num"]
    )
    grouped["amu_bps"] = grouped["weighted_sum"] / grouped["num_amu"].replace(0, np.nan)

    curve_df = grouped[["market", "tte_bin", "amu_bps", "num_amu", "num_pairs"]].copy()
    curve_df["tte_bin"] = curve_df["tte_bin"].astype("object")
    curve_df["tte_bin_center"] = curve_df["tte_bin"].map(lambda value: value.mid if isinstance(value, pd.Interval) else np.nan)
    curve_df = curve_df.sort_values(["market", "tte_bin_center"])
    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["tte_bin_center"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "tte_bin_center", "num_amu", "num_pairs"]].melt(
        id_vars=["market", "tte_bin_center"],
        value_vars=["num_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_amu": "Num MU ticks in bin", "num_pairs": "Num ticks in bin"})
    counts_long["count_smooth"] = np.nan
    for (market, metric), group in counts_long.groupby(["market", "metric"], sort=False):
        counts_long.loc[group.index, "count_smooth"] = _gaussian_kernel_smooth(
            group["tte_bin_center"].to_numpy(),
            group["count"].to_numpy(),
        )

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    sns.lineplot(data=curve_df, x="tte_bin_center", y="amu_bps_smooth", hue="market", linewidth=2.0, palette="deep", ax=ax_top)
    ax_top.set_ylabel("AMU bps (smoothed)")
    ax_top.set_title("AMU vs TTE (all exchange/ref_sym)")
    ax_top.legend(title="", loc="best")
    sns.lineplot(
        data=counts_long,
        x="tte_bin_center",
        y="count_smooth",
        hue="market",
        style="metric",
        linewidth=1.8,
        palette="deep",
        dashes={"Num MU ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Time to Expiration (years, 100 linear bins from 0.0 to 0.7)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num MU ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_tte.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _lazy_pcpb(parquet_path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(parquet_path))


def _parquet_num_rows(parquet_path: str | Path) -> int:
    return int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)


def _amu_agg_exprs() -> list[pl.Expr]:
    return [
        pl.col("fwd_call").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("fwd_call_bp"),
        pl.col("bck_call").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("bck_call_bp"),
        pl.col("bck_put").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("bck_put_bp"),
        pl.col("fwd_put").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("fwd_put_bp"),
        (pl.col("fwd_call") > 0).sum().alias("fwd_call_num"),
        (pl.col("bck_call") > 0).sum().alias("bck_call_num"),
        (pl.col("bck_put") > 0).sum().alias("bck_put_num"),
        (pl.col("fwd_put") > 0).sum().alias("fwd_put_num"),
        pl.len().alias("num_pairs"),
    ]


def _add_amu_bps_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        (
            pl.col("fwd_call_num")
            + pl.col("bck_call_num")
            + pl.col("bck_put_num")
            + pl.col("fwd_put_num")
        ).alias("num_amu")
    ).with_columns(
        (
            pl.col("fwd_call_bp") * pl.col("fwd_call_num")
            + pl.col("bck_call_bp") * pl.col("bck_call_num")
            + pl.col("bck_put_bp") * pl.col("bck_put_num")
            + pl.col("fwd_put_bp") * pl.col("fwd_put_num")
        ).alias("weighted_sum")
    ).with_columns(
        pl.when(pl.col("num_amu") > 0)
        .then(pl.col("weighted_sum") / pl.col("num_amu"))
        .otherwise(None)
        .alias("amu_bps")
    )


@debug_runtime("inspect_pcpb_input_from_parquet")
def inspect_pcpb_input_from_parquet(parquet_path: str | Path) -> None:
    parquet_file = pq.ParquetFile(str(parquet_path))
    logger.info("AMU columns: %s", ", ".join(parquet_file.schema.names))

    lf = _lazy_pcpb(parquet_path)
    ref_syms = lf.select(pl.col("ref_sym").unique().sort()).collect().to_series().to_list()
    exchanges = lf.select(pl.col("exchange").unique().sort()).collect().to_series().to_list()
    min_ts, max_ts = lf.select(
        pl.col("timestamp").min().alias("min_ts"),
        pl.col("timestamp").max().alias("max_ts"),
    ).collect().row(0)

    logger.info("AMU ref_syms: %s", ref_syms)
    logger.info("AMU exchanges: %s", exchanges)
    logger.info("AMU timestamp range: %s -> %s", min_ts, max_ts)


@debug_runtime("write_summary_daily_table_from_parquet")
def write_summary_daily_table_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    strikes_lf = filter_ticks(_lazy_pcpb(parquet_path), apply_rel_strike=False)
    group_keys = ["ref_sym", "exchange", "timestamp"]
    per_ts = filtered_lf.group_by(group_keys).agg(
        pl.len().alias("num_contracts_ts"),
        pl.col("exp").n_unique().alias("num_expirations_ts"),
        pl.col("call_opt_spread_bp").mean().alias("call_spread_bp_ts"),
        pl.col("put_opt_spread_bp").mean().alias("put_spread_bp_ts"),
    )
    strikes_per_ts = strikes_lf.group_by(group_keys).agg(
        pl.col("strike").n_unique().alias("num_strikes_ts"),
    )
    per_ts = per_ts.join(strikes_per_ts, on=group_keys, how="left")
    n_days = filtered_lf.group_by(["ref_sym", "exchange"]).agg(pl.col("mdy").n_unique().alias("Num days in sample"))
    summary = (
        per_ts.group_by(["ref_sym", "exchange"])
        .agg(
            pl.col("num_contracts_ts").mean().alias("Avg num contracts"),
            pl.col("num_expirations_ts").mean().alias("Avg num expirations"),
            pl.col("num_strikes_ts").mean().alias("Avg num strikes"),
            pl.col("call_spread_bp_ts").mean().alias("Avg call spread (bp)"),
            pl.col("put_spread_bp_ts").mean().alias("Avg put spread (bp)"),
        )
        .join(n_days, on=["ref_sym", "exchange"], how="left")
        .sort(["ref_sym", "exchange"])
        .collect()
    )

    summary_pd = summary.to_pandas()
    for column in [
        "Avg num contracts",
        "Avg num expirations",
        "Avg num strikes",
        "Num days in sample",
        "Avg call spread (bp)",
        "Avg put spread (bp)",
    ]:
        summary_pd[column] = pd.to_numeric(summary_pd[column], errors="coerce").round(0).astype("Int64")

    summary_t = summary_pd.set_index(["ref_sym", "exchange"]).T
    summary_t.columns = summary_t.columns.set_names([None, None])
    output_path = output_dir / "summary_daily_option_coverage_table.tex"
    dataframe_to_tabular_tex(summary_t, output_path)
    return output_path


@debug_runtime("write_amu_summary_table_from_parquet")
def write_amu_summary_table_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    lf = _add_amu_bps_columns(filtered_lf.group_by(["ref_sym", "exchange"]).agg(_amu_agg_exprs()))
    table = (
        lf.select(
            pl.col("ref_sym").alias("underlying"),
            "exchange",
            pl.col("amu_bps").alias("AMU(bps)"),
            pl.col("num_amu").alias("MU occurs"),
            pl.col("num_pairs").alias("Num Observations"),
        )
        .sort(["underlying", "exchange"])
        .collect()
        .to_pandas()
    )
    table = table.set_index(["underlying", "exchange"])
    table["AMU(bps)"] = table["AMU(bps)"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    for column in ["MU occurs", "Num Observations"]:
        table[column] = table[column].map(lambda value: f"{int(value):,}" if pd.notna(value) else "")

    output_path = output_dir / "amu_summary_table.tex"
    dataframe_to_tabular_tex(table, output_path)
    return output_path


@debug_runtime("plot_4_spreads_from_parquet")
def plot_4_spreads_from_parquet(parquet_path: str | Path, *, output_dir: Path, rng: int = 100) -> list[Path]:
    columns = [
        "exchange",
        "ref_sym",
        "rel_strike",
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
        "fwd_call",
        "fwd_put",
        "bck_call",
        "bck_put",
    ]
    parquet = pq.ParquetFile(str(parquet_path))
    bin_edges = np.linspace(-rng, rng, 101)
    bin_width = float(bin_edges[1] - bin_edges[0])
    hist_counts: dict[tuple[str, str, str], np.ndarray] = {}
    sample_sizes: dict[tuple[str, str, str], int] = {}

    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        chunk = pl.from_arrow(batch)
        chunk = filter_ticks(chunk)
        if chunk.is_empty():
            continue
        for (exchange, ref_sym), subset in chunk.group_by(["exchange", "ref_sym"], maintain_order=False):
            market = (str(exchange), str(ref_sym))
            for column in ["fwd_call", "fwd_put", "bck_call", "bck_put"]:
                values = subset.get_column(column).cast(pl.Float64, strict=False).to_numpy()
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                hist, _ = np.histogram(values, bins=bin_edges, density=False)
                key = (market[0], market[1], column)
                if key not in hist_counts:
                    hist_counts[key] = hist.astype(float)
                    sample_sizes[key] = int(values.size)
                else:
                    hist_counts[key] += hist
                    sample_sizes[key] += int(values.size)

    markets = sorted({(exchange, ref_sym) for (exchange, ref_sym, _) in hist_counts.keys()})
    output_paths: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")
    for exchange, ref_sym in markets:
        fig, ax = plt.subplots(figsize=(10, 6))
        visible_max = 0.0
        amu_parts: list[tuple[float, int]] = []
        for column, color in zip(["fwd_call", "fwd_put", "bck_call", "bck_put"], sns.color_palette("deep", n_colors=4), strict=False):
            key = (exchange, ref_sym, column)
            if key not in hist_counts or sample_sizes[key] == 0:
                continue
            counts = hist_counts[key] / (sample_sizes[key] * bin_width)
            visible_max = max(visible_max, float(np.nanmax(counts)) if counts.size else 0.0)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            x_fine = np.linspace(-rng, rng, 800)
            y_fine = np.interp(x_fine, centers, counts, left=0.0, right=0.0)
            ax.plot(x_fine[x_fine < 0], y_fine[x_fine < 0], color=color, linestyle="--", linewidth=1.6, alpha=0.9)
            ax.plot(x_fine[x_fine >= 0], y_fine[x_fine >= 0], color=color, linestyle="-", linewidth=2.6, alpha=1.0, label=column)

            clipped_centers = np.clip(centers, 0.0, max_amu_bp_default)
            mean_clip = float(np.sum(clipped_centers * counts) * bin_width)
            n_pos = int(np.sum(hist_counts[key][centers > 0]))
            amu_parts.append((mean_clip, n_pos))

        amu_num = sum(count for _, count in amu_parts)
        amu_text = float(np.sum([mean * count for mean, count in amu_parts]) / amu_num) if amu_num > 0 else np.nan
        ax.axvline(0, color="black", linestyle="-", linewidth=3.0, alpha=0.8)
        ax.set_xlim(-rng, rng)
        ax.set_ylim(0, visible_max * 1.05 if visible_max > 0 else 1.0)
        ax.set_xlabel("AMU bp")
        ax.set_ylabel("Density")
        ax.set_title(f"{exchange} {ref_sym}")
        ax.legend(title="Spread")
        ax.text(0.75, 0.5, f"mean AMU = {amu_text:.1f} bp", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")
        ax.text(0.25, 0.75, "no trading opportunities", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")

        output_path = output_dir / f"{_safe_name(exchange)}_{_safe_name(ref_sym)}_4_spreads.pdf"
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


@debug_runtime("plot_amu_bps_by_date_from_parquet")
def plot_amu_bps_by_date_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    lf = filtered_lf.with_columns(
        (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market")
    )
    daily_amu = _add_amu_bps_columns(
        lf.group_by(["mdy", "exchange", "ref_sym", "market"]).agg(_amu_agg_exprs())
    ).select(["mdy", "market", "amu_bps"]).sort(["market", "mdy"]).collect().to_pandas()

    btc_index = (
        filtered_lf
        .filter(pl.col("ref_sym") == "BTCUSD")
        .group_by("mdy")
        .agg(pl.col("index").mean().alias("index"))
        .sort("mdy")
        .collect()
        .to_pandas()
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=daily_amu, x="mdy", y="amu_bps", hue="market", linewidth=2.0, marker=None, ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(data=btc_index, x="mdy", y="index", color="black", linewidth=2.6, linestyle="--", ax=ax2, label="BTCUSD index")
    ax2.grid(False)
    ax2.set_ylabel("BTCUSD index price")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", title="")
    if ax2.legend_ is not None:
        ax2.legend_.remove()

    ax.set_xlabel("Date")
    ax.set_ylabel("AMU bps")
    ax.set_title("AMU by Date with BTCUSD Index")
    ax.tick_params(axis="x", labelrotation=90)

    output_path = output_dir / "multi_exchange_amu_bps_by_date.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("plot_amu_bps_by_rel_strike_from_parquet")
def plot_amu_bps_by_rel_strike_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    curve_df = _add_amu_bps_columns(
        filtered_lf
        .filter(
            pl.col("exchange").is_not_null()
            & pl.col("ref_sym").is_not_null()
            & pl.col("rel_strike").is_not_null()
            & pl.col("rel_strike").is_between(0.3, 3.0)
        )
        .with_columns(
            (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market"),
            ((pl.col("rel_strike") / 0.01).round() * 0.01).alias("rel_strike_bin"),
        )
        .group_by(["market", "rel_strike_bin"])
        .agg(_amu_agg_exprs())
    ).select(["market", "rel_strike_bin", "amu_bps", "num_amu", "num_pairs"]).sort(["market", "rel_strike_bin"]).collect().to_pandas()

    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["rel_strike_bin"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "rel_strike_bin", "num_amu", "num_pairs"]].melt(
        id_vars=["market", "rel_strike_bin"],
        value_vars=["num_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_amu": "Num MU ticks in bin", "num_pairs": "Num ticks in bin"})
    counts_long["count_smooth"] = np.nan
    for (market, metric), group in counts_long.groupby(["market", "metric"], sort=False):
        counts_long.loc[group.index, "count_smooth"] = _gaussian_kernel_smooth(
            group["rel_strike_bin"].to_numpy(),
            group["count"].to_numpy(),
        )

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    sns.lineplot(data=curve_df, x="rel_strike_bin", y="amu_bps_smooth", hue="market", linewidth=2.0, palette="deep", ax=ax_top)
    ax_top.set_ylabel("AMU bps (smoothed)")
    ax_top.set_title("AMU vs Relative Strike (all exchange/ref_sym)")
    ax_top.legend(title="", loc="best")
    sns.lineplot(
        data=counts_long,
        x="rel_strike_bin",
        y="count_smooth",
        hue="market",
        style="metric",
        linewidth=1.8,
        palette="deep",
        dashes={"Num MU ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Relative strike (1% bins)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num MU ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_rel_strike.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("plot_amu_bps_by_tte_from_parquet")
def plot_amu_bps_by_tte_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    bin_width = 0.7 / 100.0
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    curve_df = _add_amu_bps_columns(
        filtered_lf
        .filter(
            pl.col("exchange").is_not_null()
            & pl.col("ref_sym").is_not_null()
            & pl.col("tte").is_not_null()
            & pl.col("tte").is_between(0.0, 0.7)
        )
        .with_columns(
            (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market"),
            (
                (
                    (pl.col("tte") / bin_width)
                    .floor()
                    .clip(lower_bound=0, upper_bound=99)
                )
                * bin_width
                + (0.5 * bin_width)
            ).alias("tte_bin_center"),
        )
        .group_by(["market", "tte_bin_center"])
        .agg(_amu_agg_exprs())
    ).select(["market", "tte_bin_center", "amu_bps", "num_amu", "num_pairs"]).sort(["market", "tte_bin_center"]).collect().to_pandas()

    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["tte_bin_center"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "tte_bin_center", "num_amu", "num_pairs"]].melt(
        id_vars=["market", "tte_bin_center"],
        value_vars=["num_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_amu": "Num MU ticks in bin", "num_pairs": "Num ticks in bin"})
    counts_long["count_smooth"] = np.nan
    for (market, metric), group in counts_long.groupby(["market", "metric"], sort=False):
        counts_long.loc[group.index, "count_smooth"] = _gaussian_kernel_smooth(
            group["tte_bin_center"].to_numpy(),
            group["count"].to_numpy(),
        )

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    sns.lineplot(data=curve_df, x="tte_bin_center", y="amu_bps_smooth", hue="market", linewidth=2.0, palette="deep", ax=ax_top)
    ax_top.set_ylabel("AMU bps (smoothed)")
    ax_top.set_title("AMU vs TTE (all exchange/ref_sym)")
    ax_top.legend(title="", loc="best")
    sns.lineplot(
        data=counts_long,
        x="tte_bin_center",
        y="count_smooth",
        hue="market",
        style="metric",
        linewidth=1.8,
        palette="deep",
        dashes={"Num MU ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Time to Expiration (years, 100 linear bins from 0.0 to 0.7)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num MU ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_tte.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("generate_all_statistics")
def generate_all_statistics(
    output_dir: str | Path,
    *,
    from_date: str,
    to_date: str,
    recreate_pcpb: bool = False,
    cache_path: str | Path | None = None,
) -> dict[str, Path | list[Path]]:
    if recreate_pcpb:
        logger.info("AMU_RECREATE_PCPB is ignored; AMU statistics now rebuild directly from dataset parquet files.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    active_cache_path = cache_path or (output_root / f"amu_frames_{from_date}_to_{to_date}_meta.pkl")
    pcpb_parquet_path = build_amu_statistics_frame_cached_path(
        cache_path=active_cache_path,
        from_date=from_date,
        to_date=to_date,
    )
    assert _parquet_num_rows(pcpb_parquet_path) > 0, f"No AMU statistics data found for range {from_date}..{to_date}"
    inspect_pcpb_input_from_parquet(pcpb_parquet_path)

    return {
        "summary_daily_option_coverage_table": write_summary_daily_table_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "amu_summary_table": write_amu_summary_table_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "spread_figures": plot_4_spreads_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_date": plot_amu_bps_by_date_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_rel_strike": plot_amu_bps_by_rel_strike_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_tte": plot_amu_bps_by_tte_from_parquet(pcpb_parquet_path, output_dir=output_root),
    }


def main() -> None:
    generate_all_statistics(
        PUBLICATION_DIR,
        from_date=os.environ.get("FROM_DATE", "2020-01-01"),
        to_date=os.environ.get("TO_DATE", "2026-06-05"),
        recreate_pcpb=os.environ.get("AMU_RECREATE_PCPB", "0") == "1",
    )


#if __name__ == "__main__":
#	main()
def testme():
    pass