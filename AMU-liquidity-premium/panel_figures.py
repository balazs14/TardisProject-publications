from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import artifact_filter as art_filter
from amu_config import CONFIG
from amu_metrics import PATHS_PER_TICK
from figure_arbitrage_paths import write_arbitrage_paths_figures

logger = logging.getLogger(__name__)
from figure_backward_joincall import write_backward_joincall_figure
from panel_regressions import (
    amu_spec1_regression_spec,
    amu_spec2_regression_spec,
    amu_spec3_regression_spec,
    build_liquidity_analysis_panel,
    filter_analysis_panel,
    _panel_pooled_amu,
    run_amu_spec1_regression,
    run_amu_spec2_regression,
    run_amu_spec3_regression,
)

event_dates = {
    label: pd.Timestamp(value)
    for label, value in CONFIG["figures"]["event_dates"].items()
}

# AMU flavour these figures plot. The panel carries mean_amu_conditional_bp,
# mean_amu_unconditional_bp and (derived) mean_amu_rate. PANEL_AMU_COLUMN is the
# "primary" series (heatmap / daily timeseries / compression); PANEL_AMU_UNCOND_COLUMN
# is the unconditional per-quote series (event study / friction gradient). Call
# set_amu_metric("rate") to repoint both at the rate so the whole figure set can be
# regenerated on the extensive margin (see build_R_overview.py).
PANEL_AMU_COLUMN = "mean_amu_conditional_bp"
PANEL_AMU_UNCOND_COLUMN = "mean_amu_unconditional_bp"


def set_amu_metric(kind: str = "default") -> None:
    """Repoint the module-level metric columns. kind in {default, unconditional, rate}."""
    global PANEL_AMU_COLUMN, PANEL_AMU_UNCOND_COLUMN
    if kind == "rate":
        PANEL_AMU_COLUMN = PANEL_AMU_UNCOND_COLUMN = "mean_amu_rate"
    elif kind == "unconditional":
        PANEL_AMU_COLUMN = PANEL_AMU_UNCOND_COLUMN = "mean_amu_unconditional_bp"
    else:
        PANEL_AMU_COLUMN, PANEL_AMU_UNCOND_COLUMN = "mean_amu_conditional_bp", "mean_amu_unconditional_bp"


# Each panel figure paired with the tags that select it under
# ONLY_RECREATE_ARTIFACTS: (paths-key, filename, plotting fn, *tags). The first tag
# is the LaTeX label in figures-tables-overview.tex; the stem is also accepted.
_PANEL_FIGURES = (
    ("amu_bps_by_cost_pre_post", "amu_bps_by_cost_pre_post.pdf", "plot_amu_by_cost_pre_post", "fig:amu_by_cost_pre_post"),
    ("daily_amu", "liquidity_daily_amu.png", "plot_daily_amu_timeseries", "fig:liquidity_daily_amu"),
    ("heatmap", "liquidity_pre_post_heatmap.png", "plot_pre_post_heatmaps", "fig:liquidity_pre_post_heatmap"),
    ("event_study", "liquidity_event_study.png", "plot_event_study", "fig:liquidity_event_study"),
    ("friction_gradient", "liquidity_friction_gradient.png", "plot_friction_gradient", "fig:liquidity_friction_gradient"),
    ("friction_timeseries_direct", "friction_timeseries_direct.png", "plot_friction_timeseries_direct", "fig:friction_timeseries_direct"),
    ("friction_timeseries_dynamic", "friction_timeseries_dynamic.png", "plot_friction_timeseries_dynamic", "fig:friction_timeseries_dynamic"),
    ("trade_volume", "trade_volume_timeseries.png", "plot_trade_volume_timeseries", "fig:trade_volume_timeseries"),
    ("premium_volume", "premium_volume_timeseries.png", "plot_premium_volume_timeseries", "fig:premium_volume_timeseries"),
    ("amu_units_timeseries", "amu_units_timeseries.png", "plot_amu_units_timeseries", "fig:amu_units_timeseries"),
    ("compression", "liquidity_compression_decomposition.png", "plot_compression_decomposition", "fig:liquidity_compression_decomposition"),
    ("regression_coefficients", "regression_coefficients.png", "plot_regression_coefficients", "fig:regression_coefficients"),
)


def generate_all_figures(output_dir: str | Path, *, frame: pd.DataFrame | None = None, **build_kwargs) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {key: output_root / fname for key, fname, _, *_ in _PANEL_FIGURES}
    schematic_tags = ("fig:arb_paths_crossed_noncrossed", "arbitrage_paths_crossed_noncrossed",
                      "fig:backward_joincall_schematic", "backward_joincall_schematic")
    # If a selective run requests nothing this generator owns, skip the (10-30 s)
    # panel build entirely.
    all_tags = [t for _, fname, _, *tags in _PANEL_FIGURES for t in (*tags, Path(fname).stem)]
    if not (art_filter.any_wanted(*all_tags) or art_filter.any_wanted(*schematic_tags)):
        return paths
    frame = frame if frame is not None else build_liquidity_analysis_panel(**build_kwargs)
    for key, fname, fn_name, *tags in _PANEL_FIGURES:
        if art_filter.wanted(*tags, Path(fname).stem):
            globals()[fn_name](frame, paths[key])
    # Hand-built schematic figures (data-independent) so a fresh clone reproduces
    # every figure in the paper, not just the data-driven ones.
    if art_filter.wanted("fig:arb_paths_crossed_noncrossed", "arbitrage_paths_crossed_noncrossed"):
        for created in write_arbitrage_paths_figures(output_root):
            paths[created.stem] = created
    if art_filter.wanted("fig:backward_joincall_schematic", "backward_joincall_schematic"):
        joincall = write_backward_joincall_figure(output_root)
        paths[joincall.stem] = joincall
    return paths


def plot_amu_by_cost_pre_post(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Conditional and unconditional AMU versus assumed round-trip cost, split
    pre/post-2024, computed on the same filtered, n_obs-weighted panel as the
    regressions (so the baseline-cost point equals the headline pre/post AMU).
    Replaces the old pooled-tick-histogram figure to keep one estimand paper-wide."""
    filtered = filter_analysis_panel(frame)
    pre = filtered[filtered["post_2024"] == 0.0]
    post = filtered[filtered["post_2024"] == 1.0]
    costs = list(range(5, 51, 5))

    def series(sub: pd.DataFrame, *, conditional: bool) -> list[float]:
        return [_panel_pooled_amu(sub, c, conditional=conditional) for c in costs]

    c0_bp = float(CONFIG["pcp"]["cost_per_notional"]) * 10_000.0
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(costs, series(pre, conditional=True), color="#1f77b4", linestyle="-", linewidth=2.2, label="Pre 2024, conditional")
    ax.plot(costs, series(post, conditional=True), color="#d62728", linestyle="-", linewidth=2.2, label="Post 2024, conditional")
    ax.plot(costs, series(pre, conditional=False), color="#1f77b4", linestyle="--", linewidth=2.2, label="Pre 2024, unconditional")
    ax.plot(costs, series(post, conditional=False), color="#d62728", linestyle="--", linewidth=2.2, label="Post 2024, unconditional")
    ax.axvline(c0_bp, color="black", linestyle=":", linewidth=1.4, alpha=0.7)
    ax.text(c0_bp, ax.get_ylim()[1], "  primary cost", color="black", fontsize=9, va="top", ha="left")
    ax.set_xlabel("Round-trip cost (bp)")
    ax.set_ylabel("$U_u$ (bp)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("$U_u$ versus assumed cost, pre/post 2024")
    ax.legend(title="", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return Path(output_path)


def plot_daily_amu_timeseries(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    plot_frame = (
        filtered.groupby(["day", "exchange", "ref_sym"], as_index=False)[PANEL_AMU_COLUMN]
        .mean()
        .assign(series=lambda df: df["exchange"] + " | " + df["ref_sym"])
    )
    # Trailing 1-month rolling mean per market for clarity.
    plot_frame["day"] = pd.to_datetime(plot_frame["day"])
    plot_frame = plot_frame.sort_values(["series", "day"]).set_index("day")
    plot_frame[PANEL_AMU_COLUMN] = plot_frame.groupby("series")[PANEL_AMU_COLUMN].transform(
        lambda s: s.rolling("30D", min_periods=1).mean()
    )
    plot_frame = plot_frame.reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=plot_frame, x="day", y=PANEL_AMU_COLUMN, hue="series", ax=ax)
    _add_event_markers(ax)
    ax.set_ylabel("$U_u$ (bp)")
    ax.set_xlabel("Day")
    ax.set_title("$U_u$ by date")
    return _finalize_figure(fig, output_path)


# (column, y-axis label, log-scale?) for the friction-channel time series. The two
# Amihud measures and the two recovery measures are present only once the panel is
# rebuilt with the trade columns; the plot silently drops any channel that is absent.
# Friction channels split into two pages (at most seven panels each), grouped by kind.
# Each entry: (column, y-axis label, log-y?). DIRECT = static properties of the standing
# quotes (spread in three units, staleness, depth in two units); DYNAMIC = the intraday
# illiquidity/resiliency measures (Amihud in three units, spread recovery in two).
_FRICTION_TS_DIRECT = (
    ("average_put_call_spread_dollar", r"Spread (\$)", False, True),
    ("stale_proxy", r"Stale $\mathrm{Stale}_{g,t}$ (fraction)", False),
    # Depth in PREMIUM dollars: resting size x contract_size x option price (= min quote
    # size valued at the call+put mid), not the notional (size x index).
    ("mean_min_quote_size_capital", r"Depth (premium \$)", False, True),
)
_FRICTION_TS_DYNAMIC = (
    ("amihud_capital", r"Amihud (premium volume)", False, "exchange"),
    ("spread_dollar_recovery", r"Spread-\$ recovery (min)", False),
)
# Channels aggregated across the surface by SUM (a daily total) rather than by mean/median.
_FRICTION_SUM_COLS = {"total_option_volume"}
_TRADE_VOLUME_PANEL = (
    ("total_option_volume", r"Option volume (contracts, call$+$put)", True),
)


def _plot_friction_panels(
    frame: pd.DataFrame, spec: tuple, title: str, output_path: str | Path | None, window: str = "30D",
    show_event_labels: bool = True,
) -> plt.Figure:
    """Stacked time-series panels (one per channel in ``spec``), one line per exchange/
    underlying group. Aggregated over strike and maturity so each (day, exchange,
    underlying) is a single point, then smoothed with a trailing 1-month rolling mean."""
    filtered = filter_analysis_panel(frame)
    panels = [p for p in spec if p[0] in filtered.columns]
    missing = [p[0] for p in spec if p[0] not in filtered.columns]
    if missing:
        logger.warning(
            "%s: dropping %d friction channel(s) absent from the panel -- rebuild the panel "
            "cache (RECREATE_PANEL_CACHE=1) if these are expected: %s",
            title.split(" over time")[0], len(missing), missing,
        )
    cols = [p[0] for p in panels]
    if not cols:
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.set_axis_off()
        return _finalize_figure(fig, output_path)
    # Spread recovery is a heavy-tailed half-life, so take the median across the surface
    # (robust to the thin-cell tail); volume is a total, so summed; the rest use the mean.
    aggfun = {
        col: ("sum" if col in _FRICTION_SUM_COLS else "median" if "recovery" in col else "mean")
        for col in cols
    }
    agg = (
        filtered.groupby(["day", "exchange", "ref_sym"], as_index=False)
        .agg(aggfun)
        .assign(series=lambda df: df["exchange"] + " | " + df["ref_sym"])
    )
    agg["day"] = pd.to_datetime(agg["day"])
    agg = agg.sort_values(["series", "day"]).set_index("day")
    agg[cols] = agg.groupby("series")[cols].transform(lambda s: s.rolling(window, min_periods=1).mean())
    agg = agg.reset_index()
    # One fixed colour per exchange/underlying series, shared across every panel.
    markets = sorted(agg["series"].unique())
    palette = dict(zip(markets, sns.color_palette(n_colors=max(1, len(markets)))))
    is_btc = agg["ref_sym"].str.contains("BTC", na=False)
    is_deribit = agg["exchange"].str.contains("deribit", case=False, na=False)
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.3 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    # A panel tuple is (column, label, log_y[, dual_axis]); dual_axis (True/"underlying"
    # splits BTC left / ETH right; "exchange" splits deribit left / okex right) puts the two
    # groups on two linear axes both starting at zero so one scale does not swamp the other.
    for ax, panel in zip(axes, panels, strict=False):
        col, label, logy = panel[0], panel[1], panel[2]
        dual = len(panel) > 3 and panel[3]
        if dual:
            if dual == "exchange":
                left_mask, left_lab, right_lab = is_deribit, "deribit", "okex"
            else:
                left_mask, left_lab, right_lab = is_btc, "BTC", "ETH"
            sns.lineplot(data=agg[left_mask], x="day", y=col, hue="series", palette=palette, ax=ax, legend=False)
            ax2 = ax.twinx()
            sns.lineplot(data=agg[~left_mask], x="day", y=col, hue="series", palette=palette, ax=ax2, legend=False)
            ax.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            ax2.grid(False)
            ax.set_ylabel(f"{label}\n({left_lab}, left axis)")
            ax2.set_ylabel(f"{label}\n({right_lab}, right axis)")
        else:
            sns.lineplot(data=agg, x="day", y=col, hue="series", palette=palette, ax=ax, legend=False)
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(label)
        _add_event_markers(ax, labels=show_event_labels)
        ax.set_xlabel("")
    axes[-1].set_xlabel("Day")
    handles = [Line2D([0], [0], color=palette[m], lw=5.0) for m in markets]
    axes[0].legend(handles, markets, title="", fontsize=17, ncol=2, loc="upper left",
                   handlelength=2.4, borderpad=0.7, labelspacing=0.5, columnspacing=1.4)
    fig.suptitle(title)
    return _finalize_figure(fig, output_path)


def plot_friction_timeseries_direct(frame: pd.DataFrame, output_path: str | Path | None = None, window: str = "30D") -> plt.Figure:
    """Direct quote frictions over time: the put--call spread in dollars; staleness; and
    depth in dollars. Event lines are drawn without on-plot text (described in the caption)."""
    return _plot_friction_panels(
        frame, _FRICTION_TS_DIRECT,
        "Direct quote frictions over time by exchange and underlying (1-month rolling mean, over strike and maturity)",
        output_path, window, show_event_labels=False,
    )


def plot_friction_timeseries_dynamic(frame: pd.DataFrame, output_path: str | Path | None = None, window: str = "30D") -> plt.Figure:
    """Dynamic frictions over time: the Amihud illiquidity ratio with volume in shares,
    in index dollars, and in option-capital dollars; and the spread recovery half-life
    for the bp and the dollar spread."""
    return _plot_friction_panels(
        frame, _FRICTION_TS_DYNAMIC,
        "Dynamic frictions over time by exchange and underlying (1-month rolling mean, over strike and maturity)",
        output_path, window,
    )


def plot_trade_volume_timeseries(frame: pd.DataFrame, output_path: str | Path | None = None, window: str = "30D") -> plt.Figure:
    """Daily option trading volume (call + put traded size, summed over the surface) by
    exchange and underlying, smoothed with a trailing 1-month rolling mean."""
    return _plot_friction_panels(
        frame, _TRADE_VOLUME_PANEL,
        "Daily option trading volume by exchange and underlying (1-month rolling mean, summed over strike and maturity)",
        output_path, window,
    )


def plot_premium_volume_timeseries(frame: pd.DataFrame, output_path: str | Path | None = None, window: str = "30D") -> plt.Figure:
    """Daily premium volume -- (call + put) traded premium in dollars, summed over the
    surface -- by exchange and underlying, with the BTCUSD index overlaid on the right
    axis. Colours/style follow the daily unconditional-unfairness figure."""
    filtered = filter_analysis_panel(frame)
    if "premium_volume" not in filtered.columns:
        logger.warning(
            "plot_premium_volume_timeseries: panel missing premium_volume -- "
            "rebuild the panel cache (RECREATE_PANEL_CACHE=1)"
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_axis_off()
        return _finalize_figure(fig, output_path)
    agg = (
        filtered.groupby(["day", "exchange", "ref_sym"], as_index=False)["premium_volume"].sum()
        .assign(market=lambda d: d["exchange"] + " | " + d["ref_sym"])
    )
    agg["day"] = pd.to_datetime(agg["day"])
    agg = agg.sort_values(["market", "day"]).set_index("day")
    agg["premium_volume"] = agg.groupby("market")["premium_volume"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    agg = agg.reset_index()
    btc = None
    if "btcusd_ref" in filtered.columns:
        btc = filtered.groupby("day", as_index=False)["btcusd_ref"].mean()
        btc["day"] = pd.to_datetime(btc["day"])
        btc = btc.sort_values("day")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=agg, x="day", y="premium_volume", hue="market", linewidth=2.0, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Premium volume (\\$, call$+$put)")
    ax.set_xlabel("Day")
    _add_event_markers(ax)
    lines, labels = ax.get_legend_handles_labels()
    if btc is not None:
        ax2 = ax.twinx()
        sns.lineplot(data=btc, x="day", y="btcusd_ref", color="black", linewidth=2.6, linestyle="--", ax=ax2, label="BTCUSD index")
        ax2.grid(False)
        ax2.set_ylabel("BTCUSD index price")
        lines2, labels2 = ax2.get_legend_handles_labels()
        if ax2.legend_ is not None:
            ax2.legend_.remove()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9, ncol=2)
    else:
        ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.suptitle("Daily premium volume by exchange and underlying (1-month rolling mean)")
    return _finalize_figure(fig, output_path)


# Unconditional market unfairness U_u in the three units, over time.
_AMU_TS_PANELS = (
    ("mean_amu_unconditional_bp", r"$U_u^{\mathrm{bp}}$ (bp of notional)", False),
    ("mean_amu_unconditional_capital_bp", r"$U_u^{\mathrm{cap}}$ (bp of capital)", False),
    ("mean_amu_unconditional_dollar", r"$U_u^{\$}$ (\$ per contract)", False),
)


def plot_amu_units_timeseries(frame: pd.DataFrame, output_path: str | Path | None = None, window: str = "30D") -> plt.Figure:
    """Unconditional market unfairness $U_u$ over time in three units -- basis points of
    notional, basis points of option capital, and dollars per contract -- one line per
    exchange/underlying group. Aggregated (mean) over strike and maturity, then smoothed
    with a trailing 1-month rolling mean."""
    return _plot_friction_panels(
        frame, _AMU_TS_PANELS,
        r"Unconditional unfairness $U_u$ over time by exchange and underlying (1-month rolling mean, over strike and maturity)",
        output_path, window,
    )


def plot_pre_post_heatmaps(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    heatmaps: list[pd.DataFrame] = []
    for ax, post_flag, title in zip(axes, (0.0, 1.0), ("Pre-2024", "Post-2024"), strict=False):
        heatmap = (
            filtered.loc[filtered["post_2024"] == post_flag]
            .groupby(["rel_strike_bucket", "tte_bucket"], as_index=False)[PANEL_AMU_COLUMN]
            .mean()
            .pivot(index="tte_bucket", columns="rel_strike_bucket", values=PANEL_AMU_COLUMN)
        )
        heatmaps.append(heatmap)

    # Compute 99th percentile of finite AMU values across both periods for adaptive color range.
    values = [hm.to_numpy(dtype=float).ravel() for hm in heatmaps if not hm.empty]
    finite_values = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.array([], dtype=float)
    shared_vmin = 0.0
    shared_vmax = float(np.percentile(finite_values, 99)) if finite_values.size > 0 else 10.0

    for ax, title, heatmap in zip(axes, ("Pre-2024", "Post-2024"), heatmaps, strict=False):
        if heatmap.empty:
            _annotate_empty_panel(ax, f"No observations for {title.lower()} sample")
            ax.set_xlabel("Relative strike bucket")
            ax.set_title(title)
            continue
        heatmap = heatmap.sort_index().sort_index(axis=1)
        heatmap.index = [f"{float(value):.2f}" for value in heatmap.index]
        heatmap.columns = [f"{float(value):.2f}" for value in heatmap.columns]
        sns.heatmap(
            heatmap,
            cmap="RdBu_r",
            vmin=shared_vmin,
            vmax=shared_vmax,
            ax=ax,
            cbar_kws={"label": "$U_c$ (bp)"},
        )
        ax.set_title(title)
        ax.set_xlabel("Relative strike bucket")
    axes[0].set_ylabel("TTE bucket")
    fig.suptitle("$U_c$ across strike and maturity buckets")
    return _finalize_figure(fig, output_path)


def plot_event_study(frame: pd.DataFrame, output_path: str | Path | None = None, window: int = 60) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    event_rows = []
    for label, event_day in event_dates.items():
        tmp = filtered.copy()
        tmp["event"] = label
        tmp["event_date"] = event_day.strftime("%Y-%m-%d")
        tmp["rel_day"] = (tmp["day"] - event_day).dt.days
        event_rows.append(tmp.loc[tmp["rel_day"].between(-window, window)])
    plot_frame = pd.concat(event_rows, ignore_index=True)
    # Aggregate across all markets (no exchange separation)
    event_amu_col = PANEL_AMU_UNCOND_COLUMN  # fig 9 plots unconditional AMU (or rate in R mode)
    plot_frame = plot_frame.groupby(["event", "event_date", "rel_day"], as_index=False)[event_amu_col].mean()
    fig, ax = plt.subplots(figsize=(11, 6))
    
    if plot_frame.empty:
        _annotate_empty_panel(ax, "No event windows fall inside the selected sample")
        ax.set_xlabel("Days relative to event")
        fig.suptitle("Event-study windows around 2024 regime markers")
        return _finalize_figure(fig, output_path)
    
    sns.lineplot(data=plot_frame, x="rel_day", y=event_amu_col, hue="event", ax=ax, linewidth=2.0, marker=None)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("$U_u$ (bp)")
    ax.set_xlabel("Days relative to event")
    
    # Add event dates to legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        event_date = plot_frame.loc[plot_frame["event"] == label, "event_date"].iloc[0] if label in plot_frame["event"].values else ""
        new_labels.append(f"{label}\n({event_date})")
    
    ax.legend(
        handles,
        new_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="",
    )
    
    fig.suptitle("Event-study windows around 2024 regime markers")
    return _finalize_figure(fig, output_path)


def plot_friction_gradient(frame: pd.DataFrame, output_path: str | Path | None = None, bins: int = 20) -> plt.Figure:
    y_col = PANEL_AMU_UNCOND_COLUMN  # headline AMU (or rate in R mode)
    filtered = filter_analysis_panel(frame)
    filtered = filtered.loc[filtered[y_col].notna()].copy()
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), sharey=True)
    if filtered.empty:
        for ax in axes:
            _annotate_empty_panel(ax, "No observations")
        fig.suptitle("Liquidity-friction gradients")
        return _finalize_figure(fig, output_path)

    for ax, column, title, xlabel in zip(
        axes[:3],
        ("average_put_call_spread_bp", "log_mean_min_quote_size_dollar", "stale_proxy"),
        ("$U_u$ versus option spread", "$U_u$ versus log quote depth", "$U_u$ versus staleness"),
        ("Average put-call spread (bp)", "Log mean minimum quote size (USD)", "Mean stale-quote fraction"),
        strict=False,
    ):
        binned = _binned_means(filtered, column, bins, y_col=y_col)
        sns.scatterplot(data=binned, x=column, y=y_col, ax=ax)
        sns.lineplot(data=binned, x=column, y=y_col, ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    combo = (
        filtered.groupby(["ref_sym", "exchange"], as_index=False)[y_col]
        .mean()
        .sort_values(["ref_sym", "exchange"])
    )
    sns.barplot(
        data=combo,
        x="ref_sym",
        y=y_col,
        hue="exchange",
        ax=axes[3],
    )
    axes[3].set_title("$U_u$ by underlying and exchange")
    axes[3].set_xlabel("Underlying")
    axes[3].set_ylabel("")
    axes[3].legend(title="Exchange")

    axes[0].set_ylabel("$U_u$ (bp)")
    fig.suptitle("Liquidity-friction gradients")
    return _finalize_figure(fig, output_path)


def plot_compression_decomposition(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    segment_configs = [
        (
            "Underlying",
            _segment_delta(filtered, "underlying", {"BTC": filtered["eth"] == 0.0, "ETH": filtered["eth"] == 1.0}),
            "Blues",
        ),
        (
            "Exchange",
            _segment_delta(filtered, "exchange", {"Deribit": filtered["exchange"] == "deribit", "OKX": filtered["exchange"] == "okex"}),
            "Greens",
        ),
        (
            "Moneyness",
            _segment_delta(filtered, "moneyness", {"ATM": filtered["nonatm"] == 0.0, "Non-ATM": filtered["nonatm"] == 1.0}),
            "Oranges",
        ),
        (
            "Maturity",
            _segment_delta(filtered, "maturity", {"NearExp": filtered["short_tte"] == 1.0, "FarExp": filtered["short_tte"] == 0.0}),
            "Purples",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (title, segment_frame, palette_name) in zip(axes_flat, segment_configs, strict=False):
        if segment_frame.empty:
            _annotate_empty_panel(ax, f"No observations for {title.lower()} split")
            continue
        palette = sns.color_palette(palette_name, n_colors=len(segment_frame))
        sns.barplot(data=segment_frame, x="group", y="delta_amu_bp", hue="group", palette=palette, legend=False, ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

    axes_flat[0].set_ylabel("Post-Pre 2024 $U_c$ (bp)")
    for ax in axes_flat[1:]:
        ax.set_ylabel("")
    fig.suptitle("$U_c$ compression by segment")
    return _finalize_figure(fig, output_path)


def plot_regression_coefficients(
    frame: pd.DataFrame,
    output_path: str | Path | None = None,
    **build_kwargs,
) -> plt.Figure:
    results = pd.concat(
        [
            run_amu_spec1_regression(panel=frame, **build_kwargs),
            run_amu_spec2_regression(panel=frame, **build_kwargs),
            run_amu_spec3_regression(panel=frame, **build_kwargs),
        ],
        ignore_index=True,
    )

    specs = [amu_spec1_regression_spec(), amu_spec2_regression_spec(), amu_spec3_regression_spec()]
    spec_order = [spec.name for spec in specs]
    pretty_titles = {
        specs[0].name: "(1) Post",
        specs[1].name: "(2) + segments",
        specs[2].name: "(3) + frictions, cell FE",
    }
    term_order = {spec.name: list(spec.regressors) for spec in specs}
    term_labels = {
        "post_2024": "Post-2024",
        "okx": "OKX",
        "eth": "ETH",
        "atm": "ATM",
        "short_tte": "NearExp",
        "log_mean_min_quote_size_dollar": "Depth",
        "average_put_call_spread_bp": "Spread",
        "stale_proxy": "Stale",
    }

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, spec_name in zip(np.atleast_1d(axes).ravel(), spec_order, strict=False):
        spec_results = results.loc[results["specification"] == spec_name].copy()
        spec_results["term"] = pd.Categorical(spec_results["term"], categories=term_order[spec_name], ordered=True)
        spec_results = spec_results.sort_values("term")
        y = np.arange(len(spec_results))
        coefficients = spec_results["coefficient"].to_numpy(dtype=float)
        errors = 1.96 * spec_results["std_error"].to_numpy(dtype=float)
        ax.barh(y, coefficients, color="#2a9d8f", alpha=0.9)
        ax.errorbar(coefficients, y, xerr=errors, fmt="none", ecolor="#264653", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([term_labels.get(term, term) for term in spec_results["term"]], fontsize=8)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(pretty_titles[spec_name], fontsize=11)
        ax.set_xlabel("Coefficient with 95% CI")

    fig.suptitle("Nested $U_u$ regressions: coefficient estimates with 95\\% CI")
    return _finalize_figure(fig, output_path)


def _add_event_markers(ax: plt.Axes, labels: bool = True) -> None:
    for label, event_day in event_dates.items():
        ax.axvline(event_day, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        if labels:
            ax.text(event_day, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)


def _binned_means(frame: pd.DataFrame, column: str, bins: int, y_col: str = "mean_mma_bp") -> pd.DataFrame:
    work = frame[[column, y_col]].dropna().copy()
    work["bin"] = pd.qcut(work[column], q=min(bins, work[column].nunique()), duplicates="drop")
    return work.groupby("bin", as_index=False).agg({column: "mean", y_col: "mean"})


def _segment_delta(frame: pd.DataFrame, segment: str, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, mask in groups.items():
        segment_frame = frame.loc[mask]
        pre = segment_frame.loc[segment_frame["post_2024"] == 0.0, "mean_mma_bp"].mean()
        post = segment_frame.loc[segment_frame["post_2024"] == 1.0, "mean_mma_bp"].mean()
        pre_amu = segment_frame.loc[segment_frame["post_2024"] == 0.0, PANEL_AMU_COLUMN].mean()
        post_amu = segment_frame.loc[segment_frame["post_2024"] == 1.0, PANEL_AMU_COLUMN].mean()
        rows.append({"segment": segment, "group": label, "delta_amu_bp": post_amu - pre_amu})
    return pd.DataFrame(rows)


def _finalize_figure(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    return fig


def _annotate_empty_panel(ax: plt.Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)