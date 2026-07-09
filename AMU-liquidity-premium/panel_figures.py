from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from amu_config import CONFIG
from panel_regressions import build_liquidity_analysis_panel, filter_analysis_panel

event_dates = {
    label: pd.Timestamp(value)
    for label, value in CONFIG["figures"]["event_dates"].items()
}


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
    }
    plot_daily_amu_timeseries(frame, paths["daily_amu"])
    plot_pre_post_heatmaps(frame, paths["heatmap"])
    plot_event_study(frame, paths["event_study"])
    plot_friction_gradient(frame, paths["friction_gradient"])
    plot_compression_decomposition(frame, paths["compression"])
    return paths


def plot_daily_amu_timeseries(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    plot_frame = (
        filtered.groupby(["day", "exchange", "ref_sym"], as_index=False)["mean_amu_bp"]
        .mean()
        .assign(series=lambda df: df["exchange"] + " | " + df["ref_sym"])
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=plot_frame, x="day", y="mean_amu_bp", hue="series", ax=ax)
    _add_event_markers(ax)
    ax.set_ylabel("Mean AMU (bp)")
    ax.set_xlabel("Day")
    ax.set_title("Daily AMU by exchange and underlying")
    return _finalize_figure(fig, output_path)


def plot_pre_post_heatmaps(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, post_flag, title in zip(axes, (0.0, 1.0), ("Pre-2024", "Post-2024"), strict=False):
        heatmap = (
            filtered.loc[filtered["post_2024"] == post_flag]
            .groupby(["rel_strike_bucket", "tte_bucket"], as_index=False)["mean_amu_bp"]
            .mean()
            .pivot(index="tte_bucket", columns="rel_strike_bucket", values="mean_amu_bp")
        )
        if heatmap.empty:
            _annotate_empty_panel(ax, f"No observations for {title.lower()} sample")
            ax.set_xlabel("Relative strike bucket")
            ax.set_title(title)
            continue
        heatmap = heatmap.sort_index().sort_index(axis=1)
        heatmap.columns = [f"{float(value):.2f}" for value in heatmap.columns]
        sns.heatmap(heatmap, cmap="RdBu_r", center=0.0, robust=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Relative strike bucket")
    axes[0].set_ylabel("TTE bucket")
    fig.suptitle("Mean AMU across strike and maturity buckets")
    return _finalize_figure(fig, output_path)


def plot_event_study(frame: pd.DataFrame, output_path: str | Path | None = None, window: int = 60) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    event_rows = []
    for label, event_day in event_dates.items():
        tmp = filtered.copy()
        tmp["event"] = label
        tmp["rel_day"] = (tmp["day"] - event_day).dt.days
        event_rows.append(tmp.loc[tmp["rel_day"].between(-window, window)])
    plot_frame = pd.concat(event_rows, ignore_index=True)
    plot_frame = plot_frame.groupby(["exchange", "event", "rel_day"], as_index=False)["mean_amu_bp"].mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    if plot_frame.empty:
        for ax in axes:
            _annotate_empty_panel(ax, "No event windows fall inside the selected sample")
        axes[-1].set_xlabel("Days relative to event")
        fig.suptitle("Event-study windows around 2024 regime markers")
        return _finalize_figure(fig, output_path)
    for ax, exchange in zip(axes, sorted(plot_frame["exchange"].unique()), strict=False):
        sns.lineplot(data=plot_frame.loc[plot_frame["exchange"] == exchange], x="rel_day", y="mean_amu_bp", hue="event", ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(exchange)
        ax.set_ylabel("Mean AMU (bp)")
    axes[-1].set_xlabel("Days relative to event")
    fig.suptitle("Event-study windows around 2024 regime markers")
    return _finalize_figure(fig, output_path)


def plot_friction_gradient(frame: pd.DataFrame, output_path: str | Path | None = None, bins: int = 20) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, column, title in zip(
        axes,
        ("spread_bp", "depth_proxy"),
        ("MMA versus option spread", "MMA versus log quote depth"),
        strict=False,
    ):
        binned = _binned_means(filtered, column, bins)
        sns.scatterplot(data=binned, x=column, y="mean_amu_bp", ax=ax)
        sns.lineplot(data=binned, x=column, y="mean_amu_bp", ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel(column)
    axes[0].set_ylabel("Mean MMA (bp)")
    fig.suptitle("Liquidity-friction gradients")
    return _finalize_figure(fig, output_path)


def plot_compression_decomposition(frame: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    filtered = filter_analysis_panel(frame)
    segments = pd.concat(
        [
            _segment_delta(filtered, "ref_sym", {"BTC": filtered["eth"] == 0.0, "ETH": filtered["eth"] == 1.0}),
            _segment_delta(filtered, "moneyness", {"ATM": filtered["nonatm"] == 0.0, "Non-ATM": filtered["nonatm"] == 1.0}),
            _segment_delta(filtered, "tte", {"Long": filtered["short_tte"] == 0.0, "Short": filtered["short_tte"] == 1.0}),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=segments, x="segment", y="delta_amu_bp", hue="group", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Post-2024 minus pre-2024 mean AMU (bp)")
    ax.set_xlabel("Segment family")
    ax.set_title("AMU compression by segment")
    return _finalize_figure(fig, output_path)


def _add_event_markers(ax: plt.Axes) -> None:
    for label, event_day in event_dates.items():
        ax.axvline(event_day, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(event_day, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)


def _binned_means(frame: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
    work = frame[[column, "mean_amu_bp"]].dropna().copy()
    work["bin"] = pd.qcut(work[column], q=min(bins, work[column].nunique()), duplicates="drop")
    return work.groupby("bin", as_index=False).agg({column: "mean", "mean_amu_bp": "mean"})


def _segment_delta(frame: pd.DataFrame, segment: str, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, mask in groups.items():
        segment_frame = frame.loc[mask]
        pre = segment_frame.loc[segment_frame["post_2024"] == 0.0, "mean_amu_bp"].mean()
        post = segment_frame.loc[segment_frame["post_2024"] == 1.0, "mean_amu_bp"].mean()
        rows.append({"segment": segment, "group": label, "delta_amu_bp": post - pre})
    return pd.DataFrame(rows)


def _finalize_figure(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def _annotate_empty_panel(ax: plt.Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)