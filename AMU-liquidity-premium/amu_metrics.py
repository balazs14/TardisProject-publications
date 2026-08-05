"""Shared tickpath AMU/MMA definitions.

Single source of truth so the panel (`amu_panel.py`) and the statistics-frame
aggregations (`amu_statistics.py`) can never drift apart.

A *tickpath* is one (tick, path) pair. Each synchronized quote ("tick") for a
matched put--call pair contributes four tickpaths, one per join path
(`bck_joincall`, `bck_joinput`, `fwd_joincall`, `fwd_joinput`).

For any bucket:

* MMA is the mean of the path payoff over *all* tickpaths.
* AMU comes in two explicit flavours built from the same positive-part sum:

  - conditional (`amu_conditional_bp_expr`): sum over positive tickpaths of
    clip(mma, 0, max_amu_bp) / number of positive tickpaths -- the mean
    executable size *given* an executable path exists.
  - unconditional (`amu_unconditional_bp_expr`): the same positive-part sum
    divided by *all* tickpaths -- i.e. (AMU>0 frequency) x (conditional size),
    the expected executable wedge per quote.

The aggregation is expressed as additive accumulators -- a positive-part sum, a
positive tickpath count, and the tick count -- so it composes correctly across
files/blocks; each AMU flavour is a ratio of accumulated totals.
"""
from __future__ import annotations

import functools
import operator

import polars as pl

JOIN_PATH_COLUMNS: tuple[str, ...] = (
    "bck_joincall_bp",
    "bck_joinput_bp",
    "fwd_joincall_bp",
    "fwd_joinput_bp",
)


def _add(exprs: list[pl.Expr]) -> pl.Expr:
    return functools.reduce(operator.add, exprs)


def positive_part_sum_expr(max_amu_bp: float, delta: float = 0.0) -> pl.Expr:
    """Sum over the four tickpaths of the positive part of ``m - delta``, clipped to
    ``[0, max_amu_bp]``.

    ``delta = c - c0`` shifts the executability threshold from the cost ``c0`` already
    netted into ``m`` to an arbitrary cost ``c``: a tickpath contributes ``(m-delta)``
    exactly when ``m > delta`` (i.e. the gross wedge exceeds ``c``). ``delta = 0``
    recovers the sum of MMA over the positive tickpaths at the baseline cost.
    """
    return _add(
        [(pl.col(column) - delta).clip(lower_bound=0.0, upper_bound=max_amu_bp).sum() for column in JOIN_PATH_COLUMNS]
    )


def positive_count_expr(delta: float = 0.0) -> pl.Expr:
    """Number of tickpaths executable at cost offset ``delta`` (those with ``m > delta``);
    ``delta = 0`` is the baseline positive-MMA count across the four paths."""
    return _add([(pl.col(column) > delta).sum() for column in JOIN_PATH_COLUMNS])


def any_positive_count_expr() -> pl.Expr:
    """Number of ticks with at least one positive-MMA path."""
    condition = _add(
        [(pl.col(column) > 0).cast(pl.Int64) for column in JOIN_PATH_COLUMNS]
    )
    return (condition > 0).cast(pl.Int64).sum()


def positive_part_sum_dollar_expr(max_amu_bp: float, *, index_col: str = "index") -> pl.Expr:
    """Dollar (per one notional) analogue of ``positive_part_sum_expr`` at the baseline
    cost. Each tickpath's winsorized bp wedge ``clip(m, 0, max_amu_bp)`` is converted to
    a dollar wedge per notional by ``m_bp/1e4 * index``; the per-tick ``index`` multiplies
    each row before summing, so the executable-path set and winsorization match the bp
    measure exactly and only the units differ."""
    return _add(
        [
            (pl.col(column).clip(lower_bound=0.0, upper_bound=max_amu_bp) * pl.col(index_col) / 1.0e4).sum()
            for column in JOIN_PATH_COLUMNS
        ]
    )


def positive_part_sum_capital_bp_expr(
    max_amu_bp: float, *, index_col: str = "index", capital_col: str = "opt_val"
) -> pl.Expr:
    """Capital-bp analogue of ``positive_part_sum_expr``: the same winsorized bp wedge
    expressed in basis points of the option capital (sum of the call and put mid prices),
    i.e. ``clip(m, 0, max_amu_bp) * index / opt_val``. Shares the bp path set and cap."""
    return _add(
        [
            (pl.col(column).clip(lower_bound=0.0, upper_bound=max_amu_bp) * pl.col(index_col) / pl.col(capital_col)).sum()
            for column in JOIN_PATH_COLUMNS
        ]
    )


def tickpath_amu_agg_exprs(max_amu_bp: float, *, sum_alias: str, count_alias: str) -> list[pl.Expr]:
    """Group-by aggregation exprs for the AMU numerator and denominator."""
    return [
        positive_part_sum_expr(max_amu_bp).alias(sum_alias),
        positive_count_expr().cast(pl.Float64).alias(count_alias),
    ]


def tickpath_amu_agg_exprs_flavored(
    max_amu_bp: float,
    *,
    sum_alias: str,
    count_alias: str,
    sum_dollar_alias: str,
    sum_capital_alias: str,
    index_col: str = "index",
    capital_col: str = "opt_val",
) -> list[pl.Expr]:
    """AMU accumulators in all three units: the bp positive-part sum and positive count
    (as in ``tickpath_amu_agg_exprs``), plus the dollar and capital-bp positive-part sums
    built from the same winsorized wedges. The shared ``count_alias`` is the denominator
    for every conditional flavour; ``n_ticks`` (formed by the caller) for the
    unconditional ones."""
    return [
        positive_part_sum_expr(max_amu_bp).alias(sum_alias),
        positive_count_expr().cast(pl.Float64).alias(count_alias),
        positive_part_sum_dollar_expr(max_amu_bp, index_col=index_col).alias(sum_dollar_alias),
        positive_part_sum_capital_bp_expr(max_amu_bp, index_col=index_col, capital_col=capital_col).alias(
            sum_capital_alias
        ),
    ]


def amu_ratio_expr(sum_col: str, count_col: str) -> pl.Expr:
    """AMU = accumulated positive-part sum / accumulated positive tickpath count."""
    return (
        pl.when(pl.col(count_col) > 0)
        .then(pl.col(sum_col) / pl.col(count_col))
        .otherwise(None)
    )


# Number of join paths per tick (tickpaths per tick). Total tickpaths in a bucket
# is PATHS_PER_TICK * (number of ticks).
PATHS_PER_TICK: int = len(JOIN_PATH_COLUMNS)


def amu_conditional_bp_expr(sum_col: str, positive_count_col: str) -> pl.Expr:
    """Conditional AMU (bp): positive-part sum / number of positive-MMA tickpaths.

    The mean executable-arbitrage *size* conditional on an executable path existing.
    """
    return amu_ratio_expr(sum_col, positive_count_col)


def amu_unconditional_bp_expr(sum_col: str, num_ticks_col: str, paths_per_tick: int = PATHS_PER_TICK) -> pl.Expr:
    """Unconditional AMU (bp): positive-part sum / total tickpaths.

    Equals (AMU>0 frequency) x (conditional size): the expected executable wedge
    per quote, averaging in the zero (non-executable) tickpaths. ``num_ticks_col``
    holds the tick count; total tickpaths = ``paths_per_tick * num_ticks``.
    """
    denom = paths_per_tick * pl.col(num_ticks_col)
    return (
        pl.when(pl.col(num_ticks_col) > 0)
        .then(pl.col(sum_col) / denom)
        .otherwise(None)
    )
