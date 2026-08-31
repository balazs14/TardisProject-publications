#!/usr/bin/env python3
"""Diagnostic: intraday 5-min spread series for one exchange/underlying/day.

Shows whether the spread is piecewise-constant (quotes held across many 5-min slots),
which is what makes the AR(1) spread-recovery half-life near unit-root. It prints the
key number -- the fraction of consecutive 5-min steps on which the spread is UNCHANGED
-- plus the lag-1 autocorrelation of the busiest cell's deviation from its 1h baseline,
and saves a two-panel plot (market-wide average spread, and the busiest cell's spread
with its rolling-median baseline).

Usage:
    python plot_spread_series.py [DATE] [EXCHANGE] [REF_SYM] [DATA_DIR_TEMPLATE]
    e.g.  python plot_spread_series.py 2021-01-01 okex BTCUSD
          python plot_spread_series.py 2021-01-01 okex BTCUSD /abs/path/{exchange}/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from amu_config import bootstrap_repo_root

REPO_ROOT = bootstrap_repo_root(Path(__file__).resolve())

from tardis.process_pcp import compute_pcp_metrics  # noqa: E402
import amu_panel as ap  # noqa: E402
from amu_statistics import filter_ticks  # noqa: E402

DATE = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
EXCH = sys.argv[2] if len(sys.argv) > 2 else "okex"
REF = sys.argv[3] if len(sys.argv) > 3 else "BTCUSD"
DATA_DIR = sys.argv[4] if len(sys.argv) > 4 else "datasets/{exchange}/"

files = list(ap._aligned_panel_files(EXCH, "5min", DATA_DIR, from_date=DATE, to_date=DATE))
if not files:
    raise SystemExit(
        f"No aligned put/call parquet for {EXCH} {DATE} under {DATA_DIR.format(exchange=EXCH)}. "
        "Pass the correct DATA_DIR_TEMPLATE as the 4th argument."
    )

raw = pl.read_parquet(files[0])
pr = compute_pcp_metrics(raw, **ap.pcp_metric_kwargs)
pr = pr.drop_nulls(ap._required_panel_columns())
pr = filter_ticks(pr)
pr = pr.with_columns(
    pl.col("timestamp").dt.date().alias("day"),
    ap._rel_strike_bucket(pr),
    ap._tte_bucket(pr),
).filter(pl.col("ref_sym") == REF)
pr = pr.with_columns(
    (0.5 * (pl.col("call_opt_spread_bp") + pl.col("put_opt_spread_bp"))).alias("spr")
)
if pr.is_empty():
    raise SystemExit(f"No {REF} rows after filtering for {EXCH} {DATE}.")

cell_keys = ["exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"]
bins = (
    pr.group_by(cell_keys + ["timestamp"]).agg(pl.col("spr").mean().alias("spr"))
    .sort(cell_keys + ["timestamp"])
)

# Busiest cell (most 5-min observations).
counts = bins.group_by(cell_keys).agg(pl.len().alias("nt")).sort("nt", descending=True)
top = counts.row(0, named=True)
sel = (
    bins.filter(pl.all_horizontal([pl.col(k) == top[k] for k in cell_keys]))
    .select(["timestamp", "spr"]).sort("timestamp")
    .upsample("timestamp", every="5m", maintain_order=True)
)
sel = sel.with_columns(
    pl.col("spr").rolling_median(window_size=ap._RECOVERY_BASELINE_BINS, min_samples=3, center=True).alias("base")
).with_columns((pl.col("spr") - pl.col("base")).alias("dev"))

s = sel["spr"]
unchanged = (s == s.shift(1)).drop_nulls()
flat_frac = float(unchanged.mean()) if len(unchanged) else float("nan")
phi = sel.select(pl.corr("dev", pl.col("dev").shift(1))).item()
print(f"busiest cell: rel_strike={top['rel_strike_bucket']:.2f} tte={top['tte_bucket']:.3f} "
      f"obs={top['nt']}")
print(f"  UNCHANGED-step fraction = {flat_frac:.2%}   (high => piecewise-constant / held quotes)")
print(f"  deviation lag-1 autocorr phi = {phi:.4f}  (near 1 => AR(1) half-life blows up)")

mkt = pr.group_by("timestamp").agg(pl.col("spr").mean().alias("spr")).sort("timestamp")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(mkt["timestamp"].to_list(), mkt["spr"].to_list(), lw=1.2)
axes[0].set_title(f"{EXCH} {REF} {DATE}: market-wide average 5-min spread")
axes[0].set_ylabel("spread (bp)")
axes[1].plot(sel["timestamp"].to_list(), sel["spr"].to_list(), marker=".", ms=3, lw=0.8, label="cell spread")
axes[1].plot(sel["timestamp"].to_list(), sel["base"].to_list(), lw=2, label="1h rolling median")
axes[1].set_title(f"busiest cell (unchanged {flat_frac:.0%} of steps, dev phi={phi:.3f})")
axes[1].set_ylabel("spread (bp)")
axes[1].legend()
fig.tight_layout()
out = Path("artifacts") / f"spread_series_{EXCH}_{REF}_{DATE}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
