from __future__ import annotations

import logging
import os
import inspect
import pickle
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
panel_cfg = CONFIG["panel"]
regression_cfg = CONFIG["regression"]

rel_strike_min_default = float(filters_cfg["rel_strike_min"])
rel_strike_max_default = float(filters_cfg["rel_strike_max"])
spread_bp_max_default = int(filters_cfg["spread_bp_max"])
min_quote_size_dollar_default = float(filters_cfg["min_quote_size_dollar"])
min_mma_bp_default = float(filters_cfg["min_mma_bp"])
max_mma_bp_default = float(filters_cfg["max_mma_bp"])
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

event_dates = {
    label: pd.Timestamp(value)
    for label, value in CONFIG["figures"]["event_dates"].items()
}


def write_dynamic_tex_assumptions(publication_dir: str | Path = PUBLICATION_DIR) -> dict[str, Path]:
    """Write dynamic TeX inputs used by the paper from live config/code defaults."""
    output_dir = Path(publication_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nonatm_distance = float(regression_cfg["nonatm_distance"])
    short_tte_cutoff = float(regression_cfg["short_tte_cutoff"])
    cost_per_notional = float(pcp_cfg["cost_per_notional"])
    max_amu_bp = int(filters_cfg["max_amu_bp"])
    max_mma_bp = int(filters_cfg["max_mma_bp"])
    min_quote_size_dollar = float(filters_cfg["min_quote_size_dollar"])
    spread_bp_max = int(filters_cfg["spread_bp_max"])
    rel_strike_min = float(filters_cfg["rel_strike_min"])
    rel_strike_max = float(filters_cfg["rel_strike_max"])

    # Import lazily to avoid coupling module import order.
    from amu_panel import build_amu_panel

    sample_freq_default = inspect.signature(build_amu_panel).parameters["sample_freq"].default

    regression_cutoffs_path = output_dir / "regression_cutoffs.tex"
    cutoff_macros = {
        "RegNonAtmCutoff": f"{nonatm_distance:g}",
        "RegShortTteCutoff": f"{short_tte_cutoff:g}",
        "CostPerNotional": f"{cost_per_notional:g}",
        "MaxAmuBps": f"{max_amu_bp:d}",
        "MaxMmaBps": f"{max_mma_bp:d}",
        "SampleFreqDefault": f"{sample_freq_default}",
        "MinQuoteSizeDollar": f"{min_quote_size_dollar:g}",
        "SpreadBpMax": f"{spread_bp_max:d}",
        "RelStrikeMin": f"{rel_strike_min:g}",
        "RelStrikeMax": f"{rel_strike_max:g}",
    }
    required_cutoff_macros = {
        "RegNonAtmCutoff",
        "RegShortTteCutoff",
        "CostPerNotional",
        "MaxAmuBps",
        "MaxMmaBps",
        "SampleFreqDefault",
        "MinQuoteSizeDollar",
        "SpreadBpMax",
        "RelStrikeMin",
        "RelStrikeMax",
    }
    missing_cutoff_macros = required_cutoff_macros - set(cutoff_macros)
    assert not missing_cutoff_macros, f"Missing regression cutoff macros: {sorted(missing_cutoff_macros)}"

    regression_cutoffs_path.write_text(
        "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n"
            for name, value in cutoff_macros.items()
        ),
        encoding="utf-8",
    )

    return {
        "regression_cutoffs": regression_cutoffs_path,
    }

def filter_ticks(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    rel_strike_min: float = rel_strike_min_default,
    rel_strike_max: float = rel_strike_max_default,
    spread_bp_max: int = spread_bp_max_default,
    min_quote_size_dollar: float = min_quote_size_dollar_default,
    min_mma_bp: float = min_mma_bp_default,
    max_mma_bp: float = max_mma_bp_default,
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
        mma_columns = [
            column
            for column in ("fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp", "mma_fwd_bp", "mma_bck_bp")
            if column in df.columns
        ]
        for column in mma_columns:
            mask = mask & pd.to_numeric(df[column], errors="coerce").between(min_mma_bp, max_mma_bp)
        return df.loc[mask].copy()

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        expr = (
            pl.col("call_opt_spread_bp").is_between(0, spread_bp_max)
            & pl.col("put_opt_spread_bp").is_between(0, spread_bp_max)
            & pl.col("min_quote_size_dollar").is_not_null()
            & (pl.col("min_quote_size_dollar") >= min_quote_size_dollar)
        )
        if apply_rel_strike:
            expr = expr & pl.col("rel_strike").is_between(rel_strike_min, rel_strike_max)
        mma_columns = [
            column
            for column in ("fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp", "mma_fwd_bp", "mma_bck_bp")
            if column in column_names
        ]
        for column in mma_columns:
            expr = expr & pl.col(column).is_between(min_mma_bp, max_mma_bp)
        return df.filter(expr)

    raise TypeError(f"Unsupported frame type: {type(df)}")


@debug_runtime("build_amu_statistics_frame_cached_path")
def build_amu_statistics_frame_cached_path(
    *,
    cache_path: str | Path,
    exchanges: Iterable[str] = ("okex", "deribit"),
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/{exchange}/",
    from_date: str | None = None,
    to_date: str | None = None,
    force_recreate_cache: bool = False,
) -> Path:
    key = "amu_statistics_frame"
    if not force_recreate_cache:
        cached = get_cached_frame_parquet_path(cache_path, key)
        if cached is not None:
            logger.debug("Cache hit for build_amu_statistics_frame_cached_path: %s", cache_path)
            return cached
    else:
        logger.info("Force cache recreation enabled for statistics parquet: %s", cache_path)

    logger.debug("Cache miss for build_amu_statistics_frame_cached_path: %s", cache_path)
    cache_meta = Path(cache_path)
    base_name = cache_meta.name.removesuffix("_meta.pkl")
    frame_path = cache_meta.parent / f"{base_name}__{key}.parquet"
    tmp_path = frame_path.with_suffix(frame_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    from_day = date.fromisoformat(from_date) if from_date is not None else None
    to_day = date.fromisoformat(to_date) if to_date is not None else None
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_cols = 0

    try:
        for exchange in exchanges:
            root = PROJECT_ROOT / raw_data_dir.format(exchange=exchange)
            pattern = f"{exchange}_aligned_put_call_quotes_trades_chain_*_{sample_freq}.parquet"
            for file_path in sorted(root.glob(pattern)):
                parts = file_path.stem.split("_")
                file_day = None
                if len(parts) >= 5:
                    try:
                        file_day = date.fromisoformat(parts[-2])
                    except ValueError:
                        file_day = None
                if file_day is not None:
                    if from_day is not None and file_day < from_day:
                        continue
                    if to_day is not None and file_day > to_day:
                        continue

                logger.debug(f"reading file {file_path}")
                raw = pl.read_parquet(file_path)
                if raw.is_empty():
                    continue
                block = compute_pcp_metrics(raw, **pcp_metric_kwargs)
                if block.is_empty():
                    continue
                block = block.select(pcpb_columns)

                table = block.to_arrow()
                if writer is None:
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(str(tmp_path), table.schema, compression="zstd")
                writer.write_table(table)
                total_rows += block.height
                total_cols = block.width

        if writer is None:
            pl.DataFrame(schema={column: pl.Null for column in pcpb_columns}).write_parquet(tmp_path)
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


def dataframe_to_tabular_tex(df: pd.DataFrame, path: Path) -> None:
    print(df)
    logger.debug("Writing table to %s with shape=%s", path, df.shape)
    latex = df.to_latex(
        index=True,
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        na_rep="",
        bold_rows=False,
    )
    path.write_text(latex, encoding="utf-8")


# Backward-compatible alias for temporary rename used during debugging.
F_tabular_tex = dataframe_to_tabular_tex


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


def _lazy_pcpb(parquet_path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(parquet_path))


def _parquet_num_rows(parquet_path: str | Path) -> int:
    return int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)


def _amu_agg_exprs() -> list[pl.Expr]:
    return [
        pl.col("fwd_joincall_bp").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("fwd_joincall_bp_clipped"),
        pl.col("bck_joincall_bp").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("bck_joincall_bp_clipped"),
        pl.col("bck_joinput_bp").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("bck_joinput_bp_clipped"),
        pl.col("fwd_joinput_bp").clip(lower_bound=0, upper_bound=max_amu_bp_default).mean().alias("fwd_joinput_bp_clipped"),
        (pl.col("fwd_joincall_bp") > 0).sum().alias("fwd_joincall_num"),
        (pl.col("bck_joincall_bp") > 0).sum().alias("bck_joincall_num"),
        (pl.col("bck_joinput_bp") > 0).sum().alias("bck_joinput_num"),
        (pl.col("fwd_joinput_bp") > 0).sum().alias("fwd_joinput_num"),
        pl.len().alias("num_pairs"),
        (
            (
                (pl.col("fwd_joincall_bp") > 0)
                | (pl.col("bck_joincall_bp") > 0)
                | (pl.col("bck_joinput_bp") > 0)
                | (pl.col("fwd_joinput_bp") > 0)
            )
            .cast(pl.Int64)
            .sum()
            .alias("num_has_amu")
        ),
    ]


def _add_amu_bps_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        (
            pl.col("fwd_joincall_num")
            + pl.col("bck_joincall_num")
            + pl.col("bck_joinput_num")
            + pl.col("fwd_joinput_num")
        ).alias("num_amu")
    ).with_columns(
        (
            pl.col("fwd_joincall_bp_clipped") * pl.col("fwd_joincall_num")
            + pl.col("bck_joincall_bp_clipped") * pl.col("bck_joincall_num")
            + pl.col("bck_joinput_bp_clipped") * pl.col("bck_joinput_num")
            + pl.col("fwd_joinput_bp_clipped") * pl.col("fwd_joinput_num")
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
    cache_path = output_dir / "descriptive_summary_daily_numeric_cache.pkl"
    if not _regression_force_recreate_cache_enabled() and cache_path.exists():
        with cache_path.open("rb") as handle:
            summary_t = pickle.load(handle)
        logger.debug("write_summary_daily_table_from_parquet using cached numerical output cache_path=%s", cache_path)
    else:
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
        with cache_path.open("wb") as handle:
            pickle.dump(summary_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("write_summary_daily_table_from_parquet refreshed numerical cache cache_path=%s", cache_path)

    output_path = output_dir / "summary_daily_option_coverage_table.tex"
    dataframe_to_tabular_tex(summary_t, output_path)
    return output_path


@debug_runtime("write_amu_summary_table_from_parquet")
def write_amu_summary_table_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    cache_path = output_dir / "descriptive_amu_summary_numeric_cache.pkl"
    if not _regression_force_recreate_cache_enabled() and cache_path.exists():
        with cache_path.open("rb") as handle:
            table = pickle.load(handle)
        logger.debug("write_amu_summary_table_from_parquet using cached numerical output cache_path=%s", cache_path)
    else:
        filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
        lf = _add_amu_bps_columns(filtered_lf.group_by(["ref_sym", "exchange"]).agg(_amu_agg_exprs()))
        table = (
            lf.select(
                pl.col("ref_sym").alias("underlying"),
                "exchange",
                pl.col("amu_bps").alias("mean AMU(bps)"),
                pl.col("num_has_amu").alias("positive MMA occurs"),
                pl.col("num_pairs").alias("Num Observations"),
            )
            .sort(["underlying", "exchange"])
            .collect()
            .to_pandas()
        )
        table = table.set_index(["underlying", "exchange"])
        table["mean AMU(bps)"] = table["mean AMU(bps)"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        for column in ["positive MMA occurs", "Num Observations"]:
            table[column] = table[column].map(lambda value: f"{int(value):,}" if pd.notna(value) else "")
        with cache_path.open("wb") as handle:
            pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("write_amu_summary_table_from_parquet refreshed numerical cache cache_path=%s", cache_path)

    output_path = output_dir / "amu_summary_table.tex"
    dataframe_to_tabular_tex(table, output_path)
    return output_path


def _regression_force_recreate_cache_enabled() -> bool:
    value = os.environ.get("REGRESSION_FORCE_RECREATE_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@debug_runtime("plot_4_spreads_from_parquet")
def plot_4_spreads_from_parquet(parquet_path: str | Path, *, output_dir: Path, rng: int = 100) -> list[Path]:
    parquet = pq.ParquetFile(str(parquet_path))
    columns = [
        "exchange",
        "ref_sym",
        "rel_strike",
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
        "fwd_joincall_bp",
        "fwd_joinput_bp",
        "bck_joincall_bp",
        "bck_joinput_bp",
    ]
    bin_edges = np.linspace(-rng, rng, 101)
    bin_width = float(bin_edges[1] - bin_edges[0])
    cost_per_notional_bp = -float(pcp_cfg["cost_per_notional"]) * 10_000.0
    hist_counts: dict[tuple[str, str, str], np.ndarray] = {}
    sample_sizes: dict[tuple[str, str, str], int] = {}
    clipped_sums: dict[tuple[str, str, str], float] = {}
    raw_counts: dict[tuple[str, str, str], int] = {}
    positive_counts: dict[tuple[str, str, str], int] = {}

    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        chunk = pl.from_arrow(batch)
        chunk = filter_ticks(chunk)
        if chunk.is_empty():
            continue
        for (exchange, ref_sym), subset in chunk.group_by(["exchange", "ref_sym"], maintain_order=False):
            market = (str(exchange), str(ref_sym))
            for column in ["fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp"]:
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

                clipped_values = np.clip(values, 0.0, float(max_amu_bp_default))
                clipped_sums[key] = clipped_sums.get(key, 0.0) + float(clipped_values.sum())
                raw_counts[key] = raw_counts.get(key, 0) + int(values.size)
                positive_counts[key] = positive_counts.get(key, 0) + int(np.sum(values > 0))

    markets = sorted({(exchange, ref_sym) for (exchange, ref_sym, _) in hist_counts.keys()})
    output_paths: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")
    for exchange, ref_sym in markets:
        fig, ax = plt.subplots(figsize=(10, 6))
        visible_max = 0.0
        amu_parts: list[tuple[float, int]] = []
        for column, color in zip(["fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp"], sns.color_palette("deep", n_colors=4), strict=False):
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

            count = raw_counts.get(key, 0)
            mean_clip = (clipped_sums[key] / count) if count > 0 else np.nan
            n_pos = positive_counts.get(key, 0)
            amu_parts.append((mean_clip, n_pos))

        amu_num = sum(count for _, count in amu_parts)
        amu_text = float(np.sum([mean * count for mean, count in amu_parts]) / amu_num) if amu_num > 0 else np.nan
        ax.axvline(0, color="black", linestyle="-", linewidth=3.0, alpha=0.8)
        ax.set_xlim(-rng, rng)
        ax.set_ylim(0, visible_max * 1.05 if visible_max > 0 else 1.0)
        ax.set_xlabel("MMA bp")
        ax.set_ylabel("Density")
        ax.set_title(f"{exchange} {ref_sym}")
        ax.legend(title="Spread")
        ax.text(0.75, 0.5, f"mean AMU = {amu_text:.1f} bp", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")
        ax.text(0.25, 0.75, "no trading opportunities", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), va="top", ha="center")

        output_path = output_dir / (
            f"{str(exchange).strip().lower().replace('-', '_').replace('/', '_')}_"
            f"{str(ref_sym).strip().lower().replace('-', '_').replace('/', '_')}_4_spreads.pdf"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
        logger.debug(
            "Saved spread figure %s | market=%s/%s curves=%d samples=%d mean_amu_bp=%.2f y_max=%.4f",
            output_path,
            exchange,
            ref_sym,
            len(amu_parts),
            amu_num,
            amu_text,
            visible_max,
        )
    logger.debug(
        "Spread figure summary: markets=%d figures=%d histogram_series=%d",
        len(markets),
        len(output_paths),
        len(hist_counts),
    )

    target_markets = [
        ("deribit", "BTCUSD"),
        ("okex", "BTCUSD"),
        ("deribit", "ETHUSD"),
        ("okex", "ETHUSD"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    visible_max = 0.0
    spread_columns = ["fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp"]
    for column, color in zip(["fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp"], sns.color_palette("deep", n_colors=4), strict=False):
        curves: list[np.ndarray] = []
        for exchange, ref_sym in target_markets:
            key = (exchange, ref_sym, column)
            if key not in hist_counts or sample_sizes.get(key, 0) == 0:
                continue
            curves.append(hist_counts[key] / (sample_sizes[key] * bin_width))
        if not curves:
            continue
        avg_counts = np.mean(np.vstack(curves), axis=0)
        visible_max = max(visible_max, float(np.nanmax(avg_counts)) if avg_counts.size else 0.0)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        x_fine = np.linspace(-rng, rng, 800)
        y_fine = np.interp(x_fine, centers, avg_counts, left=0.0, right=0.0)
        ax.plot(x_fine[x_fine < 0], y_fine[x_fine < 0], color=color, linestyle="--", linewidth=1.6, alpha=0.9)
        ax.plot(x_fine[x_fine >= 0], y_fine[x_fine >= 0], color=color, linestyle="-", linewidth=2.6, alpha=1.0, label=column)

    overall_clipped_sum = 0.0
    overall_positive_count = 0
    for exchange, ref_sym in target_markets:
        for column in spread_columns:
            key = (exchange, ref_sym, column)
            overall_clipped_sum += clipped_sums.get(key, 0.0)
            overall_positive_count += positive_counts.get(key, 0)
    overall_amu_bp = (overall_clipped_sum / overall_positive_count) if overall_positive_count > 0 else np.nan

    ax.axvline(0, color="black", linestyle="-", linewidth=3.0, alpha=0.8)
    ax.axvline(cost_per_notional_bp, color="black", linestyle="--", linewidth=2.0, alpha=0.8)
    ax.set_xlim(-rng, rng)
    ax.set_ylim(0, visible_max * 1.05 if visible_max > 0 else 1.0)
    ax.set_xlabel("MMA bp")
    ax.set_ylabel("Density")
    ax.set_title("MMA distribution")
    ax.legend(title="Spread")
    ax.text(
        0.16,
        0.90,
        "no MM arbitrage",
        color="#b2182b",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b2182b", alpha=0.85),
        va="top",
        ha="left",
    )
    ax.text(
        0.98,
        0.90,
        f"AMU = {overall_amu_bp:.1f} bp",
        color="#1b9e77",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1b9e77", alpha=0.85),
        va="top",
        ha="right",
    )
    ax.text(
        0.98,
        0.82,
        f"Cost = {cost_per_notional_bp:.1f} bp",
        color="black",
        transform=ax.transAxes,
        va="top",
        ha="right",
    )
    output_path_avg = output_dir / "all_markets_avg_4_spreads.pdf"
    fig.tight_layout()
    fig.savefig(output_path_avg, dpi=220, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(output_path_avg)
    logger.debug("Saved aggregate spread figure %s", output_path_avg)

    return output_paths


@debug_runtime("plot_total_mma_hist_pre_post_btc_etp_from_parquet")
def plot_total_mma_hist_pre_post_btc_etp_from_parquet(parquet_path: str | Path, *, output_dir: Path, rng: int = 100) -> Path:
    parquet = pq.ParquetFile(str(parquet_path))
    columns = [
        "mdy",
        "rel_strike",
        "call_opt_spread_bp",
        "put_opt_spread_bp",
        "min_quote_size_dollar",
        "fwd_joincall_bp",
        "fwd_joinput_bp",
        "bck_joincall_bp",
        "bck_joinput_bp",
    ]
    spread_columns = ["fwd_joincall_bp", "fwd_joinput_bp", "bck_joincall_bp", "bck_joinput_bp"]
    bin_edges = np.linspace(-rng, rng, 101)
    bin_width = float(bin_edges[1] - bin_edges[0])
    cost_per_notional_bp = -float(pcp_cfg["cost_per_notional"]) * 10_000.0

    event_day = pd.Timestamp(str(regression_cfg["post_2024_start"])).date()

    pre_hist = np.zeros(len(bin_edges) - 1, dtype=float)
    post_hist = np.zeros(len(bin_edges) - 1, dtype=float)
    pre_count = 0
    post_count = 0
    pre_clipped_sum = 0.0
    post_clipped_sum = 0.0
    pre_positive_count = 0
    post_positive_count = 0

    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        chunk = pl.from_arrow(batch)
        chunk = filter_ticks(chunk)
        if chunk.is_empty():
            continue
        mdy = pd.to_datetime(chunk.get_column("mdy").to_numpy(), errors="coerce")
        pre_mask = np.asarray(mdy < pd.Timestamp(event_day))
        post_mask = np.asarray(mdy >= pd.Timestamp(event_day))

        for column in spread_columns:
            values = chunk.get_column(column).cast(pl.Float64, strict=False).to_numpy()
            valid = np.isfinite(values)
            if np.any(pre_mask & valid):
                v = values[pre_mask & valid]
                hist, _ = np.histogram(v, bins=bin_edges, density=False)
                pre_hist += hist
                pre_count += int(v.size)
                pre_clipped_sum += float(np.clip(v, 0.0, float(max_amu_bp_default)).sum())
                pre_positive_count += int(np.sum(v > 0))
            if np.any(post_mask & valid):
                v = values[post_mask & valid]
                hist, _ = np.histogram(v, bins=bin_edges, density=False)
                post_hist += hist
                post_count += int(v.size)
                post_clipped_sum += float(np.clip(v, 0.0, float(max_amu_bp_default)).sum())
                post_positive_count += int(np.sum(v > 0))

    pre_density = pre_hist / (pre_count * bin_width) if pre_count > 0 else np.zeros_like(pre_hist)
    post_density = post_hist / (post_count * bin_width) if post_count > 0 else np.zeros_like(post_hist)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x_fine = np.linspace(-rng, rng, 800)
    pre_fine = np.interp(x_fine, centers, pre_density, left=0.0, right=0.0)
    post_fine = np.interp(x_fine, centers, post_density, left=0.0, right=0.0)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_fine, pre_fine, color="#1f77b4", linewidth=2.4, label="Pre 2024")
    ax.plot(x_fine, post_fine, color="#d62728", linewidth=2.4, label="Post 2024")
    ax.axvline(0, color="black", linestyle="-", linewidth=2.6, alpha=0.85)
    ax.axvline(cost_per_notional_bp, color="black", linestyle="--", linewidth=2.0, alpha=0.8)
    ax.set_xlim(-rng, rng)
    visible_max = max(float(np.nanmax(pre_fine)) if pre_fine.size else 0.0, float(np.nanmax(post_fine)) if post_fine.size else 0.0)
    ax.set_ylim(0, visible_max * 1.05 if visible_max > 0 else 1.0)
    ax.set_xlabel("MMA bp")
    ax.set_ylabel("Density")
    ax.set_title("Total MMA histogram: Pre/Post 2024")
    ax.legend(title="")

    pre_amu_bp = (pre_clipped_sum / pre_positive_count) if pre_positive_count > 0 else np.nan
    post_amu_bp = (post_clipped_sum / post_positive_count) if post_positive_count > 0 else np.nan
    ax.text(
        0.16,
        0.90,
        "no MM arbitrage",
        color="#b2182b",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b2182b", alpha=0.85),
        va="top",
        ha="left",
    )
    ax.text(
        0.98,
        0.90,
        f"Pre AMU = {pre_amu_bp:.1f} bp",
        color="#1f77b4",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1f77b4", alpha=0.85),
        va="top",
        ha="right",
    )
    ax.text(
        0.98,
        0.80,
        f"Post AMU = {post_amu_bp:.1f} bp",
        color="#d62728",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d62728", alpha=0.85),
        va="top",
        ha="right",
    )

    output_path = output_dir / "total_mma_hist_pre_post_btc_etp.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.debug(
        "Saved pre/post MMA histogram %s pre_count=%d post_count=%d",
        output_path,
        pre_count,
        post_count,
    )
    return output_path


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
    ax.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left",
        bbox_to_anchor=(-0.15, 1.0),
        borderaxespad=0.0,
        title="",
    )
    if ax2.legend_ is not None:
        ax2.legend_.remove()

    ax.set_xlabel("Date")
    ax.set_ylabel("AMU bps")
    ax.set_title("AMU by Date with BTCUSD Index")
    ax.tick_params(axis="x", labelrotation=90)

    output_path = output_dir / "multi_exchange_amu_bps_by_date.pdf"
    amu_min = float(daily_amu["amu_bps"].min()) if not daily_amu.empty else np.nan
    amu_max = float(daily_amu["amu_bps"].max()) if not daily_amu.empty else np.nan
    btc_min = float(btc_index["index"].min()) if not btc_index.empty else np.nan
    btc_max = float(btc_index["index"].max()) if not btc_index.empty else np.nan
    logger.debug(
        "Date plot summary: rows=%d markets=%d amu_bp_range=[%.2f, %.2f] btc_index_range=[%.2f, %.2f] output=%s",
        len(daily_amu),
        daily_amu["market"].nunique() if "market" in daily_amu else 0,
        amu_min,
        amu_max,
        btc_min,
        btc_max,
        output_path,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


@debug_runtime("plot_amu_bps_avg_2023_2024_with_events_from_parquet")
def plot_amu_bps_avg_2023_2024_with_events_from_parquet(parquet_path: str | Path, *, output_dir: Path) -> Path:
    filtered_lf = filter_ticks(_lazy_pcpb(parquet_path))
    lf = filtered_lf.with_columns(
        (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market")
    )
    daily_amu = _add_amu_bps_columns(
        lf.group_by(["mdy", "exchange", "ref_sym", "market"]).agg(_amu_agg_exprs())
    ).select(["mdy", "exchange", "ref_sym", "market", "amu_bps"]).sort(["market", "mdy"]).collect().to_pandas()

    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-12-31")
    target_markets = {
        ("deribit", "BTCUSD"),
        ("okex", "BTCUSD"),
        ("deribit", "ETHUSD"),
        ("okex", "ETHUSD"),
    }
    plot_frame = daily_amu.loc[
        daily_amu.apply(lambda row: (str(row["exchange"]), str(row["ref_sym"])) in target_markets, axis=1)
    ].copy()
    plot_frame["mdy"] = pd.to_datetime(plot_frame["mdy"])
    plot_frame = plot_frame.loc[plot_frame["mdy"].between(start, end)]

    avg_daily = (
        plot_frame.groupby("mdy", as_index=False)["amu_bps"]
        .mean()
        .sort_values("mdy")
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))
    if avg_daily.empty:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No AMU observations for 2023-01-01 to 2024-12-31 in the target markets",
            ha="center",
            va="center",
            wrap=True,
        )
    else:
        sns.lineplot(data=avg_daily, x="mdy", y="amu_bps", color="#1f77b4", linewidth=2.6, ax=ax)

    from matplotlib.lines import Line2D

    event_handles: list[Line2D] = []
    event_palette = sns.color_palette("tab10", n_colors=max(len(event_dates), 1))
    for idx, (label, event_day) in enumerate(event_dates.items()):
        event_ts = pd.Timestamp(event_day)
        if start <= event_ts <= end:
            event_color = event_palette[idx % len(event_palette)]
            ax.axvline(event_ts, color=event_color, linestyle="-", linewidth=2.2, alpha=0.95)
            event_handles.append(Line2D([0], [0], color=event_color, linestyle="-", linewidth=2.2, label=label))

    avg_handle = Line2D([0], [0], color="#1f77b4", linestyle="-", linewidth=2.6, label="Avg AMU (all markets)")
    handles = [avg_handle, *event_handles]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            title="Vertical lines: main events",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("AMU bps")
    ax.set_title("AMU by Date (2023-2024) with Main Events")
    ax.tick_params(axis="x", labelrotation=90)

    output_path = output_dir / "multi_exchange_amu_bps_by_date_2023_2024_avg_events.pdf"
    amu_min = float(avg_daily["amu_bps"].min()) if not avg_daily.empty else np.nan
    amu_max = float(avg_daily["amu_bps"].max()) if not avg_daily.empty else np.nan
    logger.debug(
        "Date avg-event plot summary: rows=%d amu_bp_range=[%.2f, %.2f] output=%s",
        len(avg_daily),
        amu_min,
        amu_max,
        output_path,
    )
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
    ).select(["market", "rel_strike_bin", "amu_bps", "num_has_amu", "num_pairs"]).sort(["market", "rel_strike_bin"]).collect().to_pandas()

    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["rel_strike_bin"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "rel_strike_bin", "num_has_amu", "num_pairs"]].melt(
        id_vars=["market", "rel_strike_bin"],
        value_vars=["num_has_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_has_amu": "Num positive MMA ticks in bin", "num_pairs": "Num ticks in bin"})
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
        dashes={"Num positive MMA ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Relative strike (1% bins)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num positive MMA ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_rel_strike.pdf"
    strike_min = float(curve_df["rel_strike_bin"].min()) if not curve_df.empty else np.nan
    strike_max = float(curve_df["rel_strike_bin"].max()) if not curve_df.empty else np.nan
    logger.debug(
        "Rel-strike plot summary: rows=%d markets=%d rel_strike_range=[%.2f, %.2f] mean_num_has_amu=%.2f output=%s",
        len(curve_df),
        curve_df["market"].nunique() if "market" in curve_df else 0,
        strike_min,
        strike_max,
        float(curve_df["num_has_amu"].mean()) if not curve_df.empty else np.nan,
        output_path,
    )
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
    ).select(["market", "tte_bin_center", "amu_bps", "num_has_amu", "num_pairs"]).sort(["market", "tte_bin_center"]).collect().to_pandas()

    curve_df["amu_bps_smooth"] = np.nan
    for market, group in curve_df.groupby("market", sort=False):
        curve_df.loc[group.index, "amu_bps_smooth"] = _gaussian_kernel_smooth(
            group["tte_bin_center"].to_numpy(),
            group["amu_bps"].to_numpy(),
        )

    counts_long = curve_df[["market", "tte_bin_center", "num_has_amu", "num_pairs"]].melt(
        id_vars=["market", "tte_bin_center"],
        value_vars=["num_has_amu", "num_pairs"],
        var_name="metric",
        value_name="count",
    )
    counts_long["metric"] = counts_long["metric"].map({"num_has_amu": "Num positive MMA ticks in bin", "num_pairs": "Num ticks in bin"})
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
        dashes={"Num positive MMA ticks in bin": "", "Num ticks in bin": (4, 2)},
        legend=False,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Time to Expiration (years, 100 linear bins from 0.0 to 0.7)")
    ax_bottom.set_ylabel("Count")

    from matplotlib.lines import Line2D

    ax_bottom.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Num positive MMA ticks in bin"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Num ticks in bin"),
        ],
        title="",
    )

    output_path = output_dir / "multi_exchange_amu_bps_by_tte.pdf"
    tte_min = float(curve_df["tte_bin_center"].min()) if not curve_df.empty else np.nan
    tte_max = float(curve_df["tte_bin_center"].max()) if not curve_df.empty else np.nan
    logger.debug(
        "TTE plot summary: rows=%d markets=%d tte_range=[%.4f, %.4f] mean_num_has_amu=%.2f output=%s",
        len(curve_df),
        curve_df["market"].nunique() if "market" in curve_df else 0,
        tte_min,
        tte_max,
        float(curve_df["num_has_amu"].mean()) if not curve_df.empty else np.nan,
        output_path,
    )
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
    force_recreate_cache: bool = False,
    cache_path: str | Path | None = None,
) -> dict[str, Path | list[Path]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    active_cache_path = cache_path or (output_root / f"amu_frames_{from_date}_to_{to_date}_meta.pkl")
    pcpb_parquet_path = build_amu_statistics_frame_cached_path(
        cache_path=active_cache_path,
        from_date=from_date,
        to_date=to_date,
        force_recreate_cache=force_recreate_cache,
    )
    assert _parquet_num_rows(pcpb_parquet_path) > 0, f"No AMU statistics data found for range {from_date}..{to_date}"
    inspect_pcpb_input_from_parquet(pcpb_parquet_path)

    return {
        "summary_daily_option_coverage_table": write_summary_daily_table_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "amu_summary_table": write_amu_summary_table_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "spread_figures": plot_4_spreads_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "total_mma_hist_pre_post_btc_etp": plot_total_mma_hist_pre_post_btc_etp_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_date": plot_amu_bps_by_date_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_date_2023_2024_avg_events": plot_amu_bps_avg_2023_2024_with_events_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_rel_strike": plot_amu_bps_by_rel_strike_from_parquet(pcpb_parquet_path, output_dir=output_root),
        "multi_exchange_amu_bps_by_tte": plot_amu_bps_by_tte_from_parquet(pcpb_parquet_path, output_dir=output_root),
    }


def main() -> None:
    generate_all_statistics(
        PUBLICATION_DIR,
        from_date=os.environ.get("FROM_DATE", "2020-01-01"),
        to_date=os.environ.get("TO_DATE", "2026-06-05"),
        force_recreate_cache=os.environ.get("AMU_FORCE_RECREATE_CACHE", "0") == "1",
    )
