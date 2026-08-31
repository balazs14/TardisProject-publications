"""Exploratory: estimate c* by minimizing the across-day variance of the
conditional AMU, as a function of the cost threshold c.

Idea (see discussion): the join-path *_bp columns are the per-tickpath MMA
(gross wedge net of the calibrated cost c0). Sweeping a cost c is equivalent to
thresholding MMA at delta = c - c0. For each market and each day we form the
conditional AMU  E[MMA - delta | MMA > delta]  (positive part, upper-clipped at
max_amu_bp, exactly as the paper's AMU). c* is the cost that makes this daily
series flattest over time (min Var_t), i.e. loads all time variation onto the
frequency/extensive margin. Everything above c* is then the non-cost wedge;
unconditional AMU @ c* is the per-quote risk-compensation measure.

One streaming pass builds a per-(market, day) histogram of MMA; the sweep over c
is then cheap numpy on the histogram.

Usage: python explore_cost_star.py <statistics_frame.parquet> <out_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import amu_statistics as S  # for filter_ticks (same filters as the paper)

PATHS = ["bck_joincall_bp", "bck_joinput_bp", "fwd_joincall_bp", "fwd_joinput_bp"]
C0 = 30.0          # bp; = cost_per_notional (0.003) * 1e4. MMA = gross wedge - C0.
MAXCLIP = 100.0    # matches filters.max_amu_bp
BIN_LO, BIN_HI = -300, 300           # 1 bp histogram bins for MMA
DELTAS = np.arange(-25.0, 60.0 + 1e-9, 1.0)   # cost c = C0 + delta  ->  c in [5, 90] bp
COSTS = C0 + DELTAS
MIN_DAY_POS = 30   # require >=30 positive-part obs on a day to use it in Var_t


FILTER_COLS = [
    "mdy", "exchange", "ref_sym", "rel_strike",
    "call_opt_spread_bp", "put_opt_spread_bp", "min_quote_size_dollar",
] + PATHS


def _collapse(frames):
    return pl.concat(frames, how="vertical").group_by(["market", "mdy", "bin"]).agg(
        pl.col("len").sum().alias("len")
    )


def build_histogram(frame_path: str, rg_step: int = 4) -> "pd.DataFrame":
    """Single-process, memory-bounded histogram build. Subsamples every rg_step-th
    parquet row group (each day spans many row groups, so all ~250 days are still
    covered) and collapses periodically. rg_step keeps the Arrow-read retention on
    this small box under budget while finishing within the shell time limit."""
    import gc

    pf = pq.ParquetFile(frame_path)
    nrg = pf.num_row_groups
    running, accum = None, []
    for i in range(0, nrg, rg_step):
        chunk = S.filter_ticks(pl.from_arrow(pf.read_row_group(i, columns=FILTER_COLS)))
        if chunk.is_empty():
            continue
        chunk = chunk.with_columns(
            (pl.col("exchange").cast(pl.Utf8) + pl.lit(" | ") + pl.col("ref_sym").cast(pl.Utf8)).alias("market")
        )
        parts = [chunk.select(["market", "mdy", pl.col(p).alias("mma")]) for p in PATHS]
        long = pl.concat(parts, how="vertical").filter(pl.col("mma").is_finite()).with_columns(
            pl.col("mma").clip(float(BIN_LO), float(BIN_HI)).floor().cast(pl.Int32).alias("bin")
        )
        accum.append(long.group_by(["market", "mdy", "bin"]).len())
        if len(accum) >= 40:
            running = _collapse(accum if running is None else [running, *accum])
            accum = []
            gc.collect()
    if accum:
        running = _collapse(accum if running is None else [running, *accum])
    return running.to_pandas()


def _sweep_days(pivot) -> dict:
    """Given a (days x bin) count matrix for one group of days, compute the sweep
    curves over the cost grid: Var_t, CV_t, corr_t(cond,freq), and the mean levels."""
    centers = pivot.columns.to_numpy() + 0.5
    C = pivot.to_numpy(dtype=float)                 # (days, bins)
    total_per_day = C.sum(axis=1)                   # (days,)

    var_t = np.full(DELTAS.size, np.nan)
    cv_t = np.full(DELTAS.size, np.nan)
    corr_t = np.full(DELTAS.size, np.nan)   # corr over days of (conditional, frequency)
    mean_cond = np.full(DELTAS.size, np.nan)
    mean_freq = np.full(DELTAS.size, np.nan)
    n_days = np.zeros(DELTAS.size, dtype=int)
    for k, d in enumerate(DELTAS):
        mask = centers > d
        if not mask.any():
            continue
        w = np.clip(centers[mask] - d, 0.0, MAXCLIP)
        Cm = C[:, mask]
        den = Cm.sum(axis=1)                         # positive-part obs per day
        num = Cm @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            cond_day = np.where(den > 0, num / den, np.nan)
            freq_day = np.where(total_per_day > 0, den / total_per_day, np.nan)
        use = (den >= MIN_DAY_POS) & np.isfinite(cond_day) & np.isfinite(freq_day)
        vals = cond_day[use]
        n_days[k] = int(vals.size)
        if vals.size >= 5:
            var_t[k] = float(np.var(vals))
            m = float(np.mean(vals))
            cv_t[k] = float(np.std(vals) / m) if m != 0 else np.nan
            mean_cond[k] = m
            fv = freq_day[use]
            if np.std(vals) > 0 and np.std(fv) > 0:
                corr_t[k] = float(np.corrcoef(vals, fv)[0, 1])
        mean_freq[k] = float(np.nanmean(freq_day))
    return dict(
        var_t=var_t, cv_t=cv_t, corr_t=corr_t, mean_cond=mean_cond, mean_freq=mean_freq,
        centers=centers, C=C, days=pivot.index.to_numpy(), n_days=n_days,
    )


def _pivot(sub):
    return sub.pivot_table(index="mdy", columns="bin", values="len", aggfunc="sum", fill_value=0)


def sweep(hist_pd) -> dict:
    return {
        market: _sweep_days(_pivot(hist_pd[hist_pd["market"] == market]))
        for market in sorted(hist_pd["market"].unique())
    }


def sweep_by_period(hist_pd, post_start) -> dict:
    """Same sweep, but split each market's days into pre- vs post-2024 by post_start."""
    hp = hist_pd.copy()
    hp["_mdy"] = pd.to_datetime(hp["mdy"], errors="coerce")
    hp["period"] = np.where(hp["_mdy"] < pd.Timestamp(post_start), "pre", "post")
    results = {}
    for market in sorted(hp["market"].unique()):
        for period in ("pre", "post"):
            sub = hp[(hp["market"] == market) & (hp["period"] == period)]
            if not sub.empty:
                results[(market, period)] = _sweep_days(_pivot(sub))
    return results


def plot_prepost(res_pp: dict, out_dir: Path, post_start) -> Path:
    """Var_t and CV_t vs cost, one line per (market x pre/post-2024)."""
    from matplotlib.lines import Line2D

    markets = sorted({m for (m, _) in res_pp})
    palette = dict(zip(markets, sns.color_palette("deep", n_colors=len(markets))))
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    specs = [
        ("var_t", axes[0], r"$\mathrm{Var}_t[\ \mathrm{AMU}_{cond}(c)\ ]$",
         "(A) Across-day variance of conditional AMU"),
        ("cv_t", axes[1], r"$\mathrm{CV}_t[\ \mathrm{AMU}_{cond}(c)\ ]$",
         "(B) Coefficient of variation over days"),
    ]
    for metric, ax, ylab, title in specs:
        for (m, period), r in res_pp.items():
            ax.plot(
                COSTS, r[metric], color=palette[m],
                linestyle=("--" if period == "pre" else "-"),
                linewidth=2.0, alpha=0.9,
            )
        ax.axvline(C0, color="black", linestyle=":", linewidth=1.3, alpha=0.6)
        ax.set_xlabel("cost c (bp)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}\nby market x pre/post {pd.Timestamp(post_start).date()}")
    color_handles = [Line2D([0], [0], color=palette[m], lw=2.5, label=m) for m in markets]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="post"),
        Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="pre"),
    ]
    axes[0].legend(handles=color_handles, title="market", fontsize=10, loc="best")
    axes[1].legend(handles=style_handles, title="period", fontsize=11, loc="best")
    fig.tight_layout()
    out = out_dir / "cost_star_sweep_prepost.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def cstar(costs, curve):
    finite = np.isfinite(curve)
    if not finite.any():
        return np.nan
    idx = np.nanargmin(np.where(finite, curve, np.inf))
    return float(costs[idx])


def main():
    frame_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    rg_step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_pd = build_histogram(frame_path, rg_step=rg_step)
    res = sweep(hist_pd)
    markets = list(res.keys())

    print(f"{'market':<18} {'c*(Var)':>8} {'c*(CV)':>8} {'cond@c*':>9} {'freq@c*':>9} {'uncond@c*':>10}")
    for m in markets:
        r = res[m]
        cs_var = cstar(COSTS, r["var_t"])
        cs_cv = cstar(COSTS, r["cv_t"])
        j = int(np.nanargmin(np.where(np.isfinite(r["var_t"]), r["var_t"], np.inf)))
        cond = r["mean_cond"][j]
        freq = r["mean_freq"][j]
        print(f"{m:<18} {cs_var:8.1f} {cs_cv:8.1f} {cond:9.2f} {freq:9.4f} {cond*freq:10.3f}")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    palette = dict(zip(markets, sns.color_palette("deep", n_colors=len(markets))))

    ax = axes[0, 0]
    for m in markets:
        ax.plot(COSTS, res[m]["var_t"], label=m, color=palette[m], linewidth=2.0)
    ax.axvline(C0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("cost c (bp)")
    ax.set_ylabel(r"$\mathrm{Var}_t\,[\ \mathrm{AMU}_{cond}(c)\ ]$")
    ax.set_title("(A) Across-day variance of conditional AMU\n(user's criterion; dashed = current c0=30)")
    ax.legend(title="", fontsize=10)

    ax = axes[0, 1]
    for m in markets:
        ax.plot(COSTS, res[m]["cv_t"], label=m, color=palette[m], linewidth=2.0)
    ax.axvline(C0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("cost c (bp)")
    ax.set_ylabel(r"$\mathrm{CV}_t\,[\ \mathrm{AMU}_{cond}(c)\ ]$")
    ax.set_title("(B) Coefficient of variation over days\n(scale-free version)")
    ax.legend(title="", fontsize=10)

    ax = axes[1, 0]
    for m in markets:
        ax.plot(COSTS, res[m]["corr_t"], label=m, color=palette[m], linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(C0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("cost c (bp)")
    ax.set_ylabel(r"$\mathrm{corr}_t(\ \mathrm{AMU}_{cond},\ \mathrm{freq}\ )$")
    ax.set_title("(C) Day-wise corr of conditional & frequency\n(decouples where risk stops loading on size)")
    ax.legend(title="", fontsize=10)

    # (D) daily conditional-AMU series at low / mid / high cost, for one market.
    ax = axes[1, 1]
    m0 = markets[0]
    r = res[m0]
    centers = r["centers"]; C = r["C"]; days = r["days"]
    for c_val in (15.0, 40.0, 70.0):
        d = c_val - C0
        mask = centers > d
        w = np.clip(centers[mask] - d, 0.0, MAXCLIP)
        den = C[:, mask].sum(axis=1)
        num = C[:, mask] @ w
        cond_day = np.where(den >= MIN_DAY_POS, num / np.where(den > 0, den, np.nan), np.nan)
        ax.plot(days, cond_day, "-", linewidth=1.6, label=f"c={c_val:.0f} bp", alpha=0.9)
    ax.set_xlabel("day")
    ax.set_ylabel("conditional AMU (bp)")
    ax.set_title(f"(D) Daily conditional AMU at 3 costs\n({m0})")
    ax.legend(title="", fontsize=10)
    ax.tick_params(axis="x", labelrotation=90)

    fig.tight_layout()
    out = out_dir / "cost_star_sweep.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # Second figure: Var_t and CV_t split by market x pre/post-2024.
    post_start = S.regression_cfg["post_2024_start"]
    res_pp = sweep_by_period(hist_pd, post_start)
    print(f"\npre/post split at {pd.Timestamp(post_start).date()}  (num days per group):")
    for (m, period), r in res_pp.items():
        print(f"  {m:<18} {period:<4} days={len(r['days'])}")
    plot_prepost(res_pp, out_dir, post_start)


if __name__ == "__main__":
    main()
