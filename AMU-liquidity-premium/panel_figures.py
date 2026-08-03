from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from amu_config import CONFIG
from amu_metrics import PATHS_PER_TICK
from figure_arbitrage_paths import write_arbitrage_paths_figures
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

# AMU flavour these figures plot. The panel carries both mean_amu_conditional_bp
# and mean_amu_unconditional_bp; switch to "mean_amu_unconditional_bp" for the
# per-quote (frequency x conditional) wedge instead of the conditional size.
PANEL_AMU_COLUMN = "mean_amu_conditional_bp"


def generate_all_figures(output_dir: str | Path, **build_kwargs) -> dict[str, Path]:
    frame = build_liquidity_analysis_panel(**build_kwargs)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily_amu": output_root / "liquidity_daily_amu.png",
        "heatmap": output_root / "liquidity_pre_post_heatmap.png",
        "event_study": output_root / "liquidity_event_study.png",
        "friction_gradient": output_root / "liquidity_friction_gradient.png",
        "compression": output_root / "liquidity_compression_decomposition.png",
        "regression_coefficients": output_root / "regression_coefficients.png",
        "amu_bps_by_cost_pre_post": output_root / "amu_bps_by_cost_pre_post.pdf",
    }
    plot_amu_by_cost_pre_post(frame, paths["amu_bps_by_cost_pre_post"])
    plot_daily_amu_timeseries(frame, paths["daily_amu"])
    plot_pre_post_heatmaps(frame, paths["heatmap"])
    plot_event_study(frame, paths["event_study"])
    plot_friction_gradient(frame, paths["friction_gradient"])
    plot_compression_decomposition(frame, paths["compression"])
    plot_regression_coefficients(frame, paths["regression_coefficients"], **build_kwargs)
    # Hand-built schematic figures (data-independent) so a fresh clone reproduces
    # every figure in the paper, not just the data-driven ones.
    for created in write_arbitrage_paths_figures(output_root):
        paths[created.stem] = created
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
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=plot_frame, x="day", y=PANEL_AMU_COLUMN, hue="series", ax=ax)
    _add_event_markers(ax)
    ax.set_ylabel("$U_u$ (bp)")
    ax.set_xlabel("Day")
    ax.set_title("$U_u$ by date")
    return _finalize_figure(fig, output_path)


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
    event_amu_col = "mean_amu_unconditional_bp"  # fig 9 plots unconditional AMU
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
    y_col = "mean_amu_unconditional_bp"  # headline AMU, consistent with the rest of the paper
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


def _add_event_markers(ax: plt.Axes) -> None:
    for label, event_day in event_dates.items():
        ax.axvline(event_day, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
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