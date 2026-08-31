"""Cross-exchange option-quote differences.

Single-venue market unfairness is, by construction, tightly linked to that venue's own
spread. A cross-venue discrepancy is not: it compares the SAME contract (underlying,
strike, expiry) quoted on two exchanges at the same instant, so a gap between Deribit's
quote and OKX's quote is a genuine best-execution (NBBO-style) inconsistency that neither
book's spread can explain away.

This module matches each Deribit contract to its OKX twin on (timestamp, ref_sym, strike,
expiry), converts both to USD-per-unit-underlying quotes (the ``_xS`` prices from
compute_pcp_metrics, so inverse/linear conventions and contract multipliers are removed),
and reports the four quote differences

    d_call_bid = call_bid^Deribit - call_bid^OKX,   d_call_ask, d_put_bid, d_put_ask

averaged over the matched surface to one point per (day, underlying). Positive means
Deribit quotes the richer price. It is deliberately independent of the panel/pcpb caches
(it reads the aligned parquets directly) so it runs without a rebuild.
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from amu_config import PCP_METRIC_KWARGS, bootstrap_repo_root, keep_sampled_day

PROJECT_ROOT = bootstrap_repo_root(Path(__file__).resolve())
PUBLICATION_DIR = Path(__file__).resolve().parent
from tardis.process_pcp import compute_pcp_metrics  # noqa: E402

logger = logging.getLogger(__name__)

_KEYS = ["timestamp", "ref_sym", "strike", "exp"]
_QUOTES = ["call_bid_price_xS", "call_ask_price_xS", "put_bid_price_xS", "put_ask_price_xS"]
_DIFFS = {
    "d_call_bid": ("call_bid_price_xS", r"$\Delta$ call bid"),
    "d_call_ask": ("call_ask_price_xS", r"$\Delta$ call ask"),
    "d_put_bid": ("put_bid_price_xS", r"$\Delta$ put bid"),
    "d_put_ask": ("put_ask_price_xS", r"$\Delta$ put ask"),
}


def _aligned_path(exchange: str, day: str, raw_data_dir: str = "datasets/{exchange}/") -> str | None:
    root = PROJECT_ROOT / raw_data_dir.format(exchange=exchange)
    hits = sorted(root.glob(f"{exchange}_aligned_put_call_quotes_trades_chain_{day}_5min.parquet"))
    return str(hits[0]) if hits else None


def _quotes_for(path: str, ref_syms: tuple[str, ...]) -> pl.DataFrame | None:
    from amu_statistics import filter_ticks  # deferred: avoids a heavy import at module load

    raw = pl.read_parquet(path)
    if raw.is_empty():
        return None
    block = compute_pcp_metrics(raw, **PCP_METRIC_KWARGS)
    missing = [c for c in _KEYS + _QUOTES if c not in block.columns]
    if missing:
        logger.warning("cross_exchange_quotes: %s missing %s -- skipping", os.path.basename(path), missing)
        return None
    # Keep only genuinely two-sided, liquid quotes (the same moneyness band, spread cap and
    # min-depth screen as the rest of the analysis). Without this the comparison is dominated
    # by one venue's placeholder/absent quotes (e.g. a 0.0005 BTC bid on a deep-ITM call),
    # which measures illiquidity, not cross-venue mispricing of actively-quoted contracts.
    block = filter_ticks(block)
    out = block.filter(pl.col("ref_sym").is_in(list(ref_syms))).select(_KEYS + _QUOTES).drop_nulls(_QUOTES)
    positive = pl.all_horizontal([pl.col(c) > 0 for c in _QUOTES])
    return out.filter(positive)


def build_cross_exchange_quote_diff(
    from_date: str,
    to_date: str,
    *,
    exchanges: tuple[str, str] = ("deribit", "okex"),
    ref_syms: tuple[str, ...] = ("BTCUSD", "ETHUSD"),
) -> pd.DataFrame:
    """Daily mean of the four cross-exchange (exchanges[0] - exchanges[1]) quote
    differences per underlying, over the matched (timestamp, ref_sym, strike, expiry)
    surface. Columns: day, ref_sym, d_call_bid, d_call_ask, d_put_bid, d_put_ask, n_pairs."""
    left, right = exchanges
    lo, hi = date.fromisoformat(from_date), date.fromisoformat(to_date)

    def _days(ex: str) -> set[str]:
        pat = str(PROJECT_ROOT / f"datasets/{ex}/{ex}_aligned_put_call_quotes_trades_chain_*_5min.parquet")
        out = set()
        for p in glob.glob(pat):
            d = os.path.basename(p).split("_")[-2]
            try:
                if lo <= date.fromisoformat(d) <= hi:
                    out.add(d)
            except ValueError:
                continue
        return out

    common = [d for d in sorted(_days(left) & _days(right)) if keep_sampled_day(date.fromisoformat(d))]
    logger.info("cross_exchange_quotes: %d common days in [%s, %s] (after SUBSAMPLE_DAYS)", len(common), from_date, to_date)
    frames: list[pd.DataFrame] = []
    for day in common:
        lp, rp = _aligned_path(left, day), _aligned_path(right, day)
        if lp is None or rp is None:
            continue
        ldf, rdf = _quotes_for(lp, ref_syms), _quotes_for(rp, ref_syms)
        if ldf is None or rdf is None or ldf.is_empty() or rdf.is_empty():
            continue
        joined = ldf.join(rdf, on=_KEYS, how="inner", suffix="_r")
        if joined.is_empty():
            continue
        diffs = joined.with_columns(
            [(pl.col(src) - pl.col(f"{src}_r")).alias(name) for name, (src, _) in _DIFFS.items()]
        )
        daily = (
            diffs.group_by("ref_sym")
            .agg(
                [pl.col(name).mean().alias(name) for name in _DIFFS]
                + [pl.len().alias("n_pairs")]
            )
            .with_columns(pl.lit(day).alias("day"))
        )
        frames.append(daily.to_pandas())
    if not frames:
        logger.warning("cross_exchange_quotes: no matched contracts in [%s, %s]", from_date, to_date)
        return pd.DataFrame(columns=["day", "ref_sym", *_DIFFS, "n_pairs"])
    out = pd.concat(frames, ignore_index=True)
    out["day"] = pd.to_datetime(out["day"])
    return out.sort_values(["ref_sym", "day"]).reset_index(drop=True)


def plot_cross_exchange_quote_diff(
    from_date: str,
    to_date: str,
    output_dir: str | Path,
    *,
    exchanges: tuple[str, str] = ("deribit", "okex"),
    ref_syms: tuple[str, ...] = ("BTCUSD", "ETHUSD"),
    window: str = "30D",
) -> Path:
    """One subplot per underlying: the four cross-exchange quote differences over time,
    smoothed with a trailing 1-month rolling mean. Positive = exchanges[0] richer."""
    left, right = exchanges
    daily = build_cross_exchange_quote_diff(from_date, to_date, exchanges=exchanges, ref_syms=ref_syms)
    present = [rs for rs in ref_syms if rs in set(daily["ref_sym"].unique())]
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(max(1, len(present)), 1, figsize=(11, 4.2 * max(1, len(present))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rs in zip(axes, present):
        sub = daily[daily["ref_sym"] == rs].set_index("day").sort_index()
        for name, (_, label) in _DIFFS.items():
            series = sub[name].rolling(window, min_periods=1).mean()
            ax.plot(series.index, series.values, linewidth=1.8, label=label)
        ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.7)
        ax.set_ylabel(f"{rs}\n{left} $-$ {right} (\\$)")
    axes[0].legend(title="", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Day")
    fig.suptitle(f"Cross-exchange option-quote differences ({left} $-$ {right}, 1-month rolling mean)")
    output_path = Path(output_dir) / "cross_exchange_quote_diff.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("cross_exchange_quotes: wrote %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = plot_cross_exchange_quote_diff(
        os.environ.get("FROM_DATE", "2020-01-01"),
        os.environ.get("TO_DATE", "2026-06-05"),
        os.environ.get("ARTIFACTS_DIR", str(PUBLICATION_DIR)),
    )
    print(out)


if __name__ == "__main__":
    main()
