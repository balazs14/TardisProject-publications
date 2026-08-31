"""Exploratory (throwaway): per-market MMA distribution, pre- vs post-2024.

A by-market facet of the paper's pooled pre/post MMA histogram. One panel per
market; pre and post density curves overlaid. Reads a statistics frame, filters
with the paper's filter_ticks, splits days at regression.post_2024_start.

Usage: python explore_mma_prepost.py <statistics_frame.parquet> <out_dir> [rg_step]
  rg_step subsamples every rg_step-th parquet row group (memory/time control on a
  small box); use 1 for full resolution.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import amu_statistics as S

PATHS = ["bck_joincall_bp", "bck_joinput_bp", "fwd_joincall_bp", "fwd_joinput_bp"]
FILTER_COLS = [
    "mdy", "exchange", "ref_sym", "rel_strike",
    "call_opt_spread_bp", "put_opt_spread_bp", "min_quote_size_dollar",
] + PATHS
RNG = 100          # plot/bin MMA over [-RNG, RNG] bp
BIN_W = 1.0        # 1 bp bins


def _collapse(frames):
    return pl.concat(frames, how="vertical").group_by(["market", "period", "bin"]).agg(
        pl.col("len").sum().alias("len")
    )


def build_hist(frame_path: str, post_date: dt.date, rg_step: int = 4):
    import gc

    pf = pq.ParquetFile(frame_path)
    running, accum, day_accum = None, [], []
    for i in range(0, pf.num_row_groups, rg_step):
        chunk = S.filter_ticks(pl.from_arrow(pf.read_row_group(i, columns=FILTER_COLS)))
        if chunk.is_empty():
            continue
        chunk = chunk.with_columns(
            (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market"),
            pl.when(pl.col("mdy").cast(pl.Date) < pl.lit(post_date)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
        )
        # distinct (market, period, day) so we can normalise by days available
        day_accum.append(chunk.select(["market", "period", pl.col("mdy").cast(pl.Date)]).unique())
        parts = [chunk.select(["market", "period", pl.col(p).alias("mma")]) for p in PATHS]
        long = pl.concat(parts, how="vertical").filter(
            pl.col("mma").is_finite() & pl.col("mma").is_between(-RNG, RNG)
        ).with_columns(pl.col("mma").floor().cast(pl.Int32).alias("bin"))
        accum.append(long.group_by(["market", "period", "bin"]).len())
        if len(accum) >= 40:
            running = _collapse(accum if running is None else [running, *accum])
            accum = []
            gc.collect()
    if accum:
        running = _collapse(accum if running is None else [running, *accum])
    days = pl.concat(day_accum, how="vertical").unique().group_by(["market", "period"]).len()
    ndays = {(r["market"], r["period"]): int(r["len"]) for r in days.to_dicts()}
    return running.to_pandas(), ndays


def main():
    frame_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    rg_step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    metric = sys.argv[4] if len(sys.argv) > 4 else "density"   # "density" or "count"
    out_dir.mkdir(parents=True, exist_ok=True)

    post_start = S.regression_cfg["post_2024_start"]
    post_date = dt.date.fromisoformat(str(post_start))
    cost_bp = float(S.pcp_cfg["cost_per_notional"]) * 1e4  # MMA reference at -cost

    hist, ndays = build_hist(frame_path, post_date, rg_step=rg_step)
    markets = sorted(hist["market"].unique())
    colors = {"pre": "#1f77b4", "post": "#d62728"}
    YLAB = {"count": "number of tickpaths", "per_day": "tickpaths per available day", "density": "density"}

    sns.set_theme(style="whitegrid", context="talk")
    ncol = 2
    nrow = int(np.ceil(len(markets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 5 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    print(f"{'market':<18} {'period':<5} {'n_ticks':>12} {'mean_mma':>9}")
    for ax, market in zip(axes, markets):
        for period in ("pre", "post"):
            sub = hist[(hist["market"] == market) & (hist["period"] == period)]
            if sub.empty:
                continue
            centers = sub["bin"].to_numpy() + 0.5
            counts = sub["len"].to_numpy(dtype=float)
            order = np.argsort(centers)
            centers, counts = centers[order], counts[order]
            total = counts.sum()
            if total <= 0:
                continue
            nd = ndays.get((market, period), 0)
            if metric == "per_day":
                yval = counts / nd if nd > 0 else np.full_like(counts, np.nan)
            elif metric == "count":
                yval = counts
            else:
                yval = counts / (total * BIN_W)
            ax.plot(centers, yval, color=colors[period], linewidth=2.2, label=f"{period} (n={int(total):,}, {nd}d)")
            mean_mma = float((centers * counts).sum() / total)
            print(f"{market:<18} {period:<5} {int(total):>12,} days={nd:>4} {mean_mma:>9.2f}")
        ax.axvline(0, color="black", linewidth=2.0, alpha=0.8)
        ax.axvline(-cost_bp, color="black", linestyle="--", linewidth=1.6, alpha=0.7)
        ax.set_xlim(-RNG, RNG)
        ax.set_ylim(bottom=0.0)
        ax.set_title(market)
        ax.set_xlabel("MMA (bp)")
        ax.set_ylabel(YLAB.get(metric, "density"))
        ax.legend(title="", fontsize=11)
    for ax in axes[len(markets):]:
        ax.set_visible(False)

    fig.suptitle(f"MMA {YLAB.get(metric, 'density')} by market, pre/post {post_date}  (dashed = -cost {cost_bp:.0f}bp)")
    fig.tight_layout()
    out = out_dir / f"mma_prepost_by_market_{metric}.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
