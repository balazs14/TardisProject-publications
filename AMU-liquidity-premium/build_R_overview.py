#!/usr/bin/env python3
"""Regenerate every figure/table/regression on the unfairness rate R (the
extensive margin) instead of U_c/U_u, into ``artifacts_R/``, and assemble a
standalone ``figures-tables-overview-R.pdf`` for a before/after comparison.

R = unconditional / conditional = share of positive tickpaths, a fraction in
[0, 1] (the bp units cancel). This driver only swaps *which column* each figure
and regression reads; it changes nothing about the stored data.

What runs where:
  * Panel-based artifacts (daily series, heatmap, event study, friction gradient,
    compression, cost curve, cost table, regressions) are rebuilt from the cached
    ``amu_panel`` parquet -- no tick data required, so they run anywhere.
  * Tick-based artifacts (by-strike/tte curves, summary table, MMA histogram,
    spreads, robustness grid, multi-exchange date plots) need the pcpb tick frame.
    They are regenerated only when that data/cache is reachable; otherwise the
    U-version is carried over unchanged so the document still compiles.

Usage:
    python build_R_overview.py                 # panel artifacts + compile
    python build_R_overview.py --with-ticks    # also try the tick-based figures
    python build_R_overview.py --no-compile    # write artifacts/tex, skip pdflatex
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from amu_config import bootstrap_repo_root

REPO_ROOT = bootstrap_repo_root(Path(__file__).resolve())
HERE = Path(__file__).resolve().parent

import amu_statistics
import panel_figures
import panel_regressions

SRC = HERE / "artifacts"
DST = HERE / "artifacts_R"
OVERVIEW_SRC = HERE / "figures-tables-overview.tex"
OVERVIEW_DST = HERE / "figures-tables-overview-R.tex"
RATE_DEPENDENT = "mean_amu_rate_c30"


def _find_panel_parquet() -> Path:
    hits = sorted(SRC.glob("amu_frames_*__amu_panel.parquet"))
    if not hits:
        raise SystemExit(
            f"No cached amu_panel parquet in {SRC}. Run the normal rebuild first "
            "so the panel is materialized, then re-run this driver."
        )
    return hits[-1]


def _date_range_from_name(parquet: Path) -> tuple[str, str]:
    m = re.search(r"amu_frames_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})__", parquet.name)
    if not m:
        raise SystemExit(f"Cannot parse date range from {parquet.name}")
    return m.group(1), m.group(2)


def _set_rate_metric() -> None:
    """Repoint every metric knob at the rate. Each consumer reads these globals
    at call time, so setting them here is enough."""
    panel_figures.set_amu_metric("rate")
    panel_regressions.AMU_DEPENDENT = RATE_DEPENDENT
    amu_statistics.AMU_STAT_METRIC = "amu_rate"


def _seed_from_u_version() -> None:
    """Copy the U-version artifacts across so every \\input/\\includegraphics in
    the overview resolves; the rate-mode writers overwrite the panel-based ones."""
    DST.mkdir(exist_ok=True)
    for f in SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, DST / f.name)


def _regen_panel(panel: pl.DataFrame) -> list[str]:
    made: list[str] = []
    # Build the analysis frame once (rate columns included) and reuse it for both
    # the tables and the figures -- passing frame= avoids the double-panel rebuild
    # and the kwarg clash that a prebuilt panel= otherwise triggers downstream.
    frame = panel_regressions.build_liquidity_analysis_panel(panel=panel)
    try:
        paths, _ = panel_regressions.write_regression_tables(DST, frame=frame)
        made += [Path(p).name for p in paths.values()]
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] panel regression tables failed: {exc}")
    try:
        paths = panel_figures.generate_all_figures(DST, frame=frame)
        made += [Path(p).name for p in paths.values()]
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] panel figures failed: {exc}")
    return made


def _regen_ticks(from_date: str, to_date: str) -> list[str]:
    made: list[str] = []
    try:
        # Prefer an existing tick cache; force_recreate stays False so we never
        # try to rebuild the 18 GB frame from source here.
        paths = amu_statistics.generate_all_statistics(
            DST, from_date=from_date, to_date=to_date, force_recreate_cache=False,
        )
        for p in paths.values():
            made += [Path(x).name for x in (p if isinstance(p, list) else [p])]
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] tick-based figures skipped (need the pcpb tick data): {exc}")
    return made


# Caption relabels for the R version. Each entry rewrites the caption of a figure
# or table whose data switched to the rate R (= share of positive tickpaths, a
# fraction in [0, 1]) so the text matches what is plotted. Precise string matches:
# if a source caption is later edited, the corresponding replace simply no-ops.
_R_CAPTION_REPS: list[tuple[str, str]] = [
    ("Daily unconditional unfairness $U_u$ by exchange and underlying, 2020--2026, with the BTCUSD index overlaid. It is high",
     "Daily unfairness rate $R$ (share of positive tickpaths) by exchange and underlying, 2020--2026, with the BTCUSD index overlaid. It is high"),
    ("All-market average unconditional unfairness $U_u$, 2023--2024",
     "All-market average unfairness rate $R$, 2023--2024"),
    ("Event-study windows for the unconditional unfairness $U_u$ around",
     "Event-study windows for the unfairness rate $R$ around"),
    (r"Mean \emph{conditional} unfairness $U_c$ across relative-strike and maturity buckets, pre- versus post-2024. The high-$U_c$ interior",
     r"Mean unfairness rate $R$ across relative-strike and maturity buckets, pre- versus post-2024. The high-$R$ interior"),
    (r"Pre- to post-2024 change in \emph{conditional} unfairness $U_c$ by segment",
     r"Pre- to post-2024 change in the unfairness rate $R$ by segment"),
    ("Regressions of the cell-level unconditional unfairness $U_u$. Column (1) regresses $U_u$ on the post-2024 indicator",
     "Regressions of the cell-level unfairness rate $R$. Column (1) regresses $R$ on the post-2024 indicator"),
    ("the change in $U_u$, in standard deviations of $U_u$,",
     "the change in $R$, in standard deviations of $R$,"),
    ("Mean MMA against average put--call spread, log quote depth, and stale-quote fraction, and by underlying and exchange. Mean MMA falls almost one-for-one with the spread",
     "Mean unfairness rate $R$ against average put--call spread, log quote depth, and stale-quote fraction, and by underlying and exchange. $R$ moves with the spread"),
    ("Conditional (solid) and unconditional (dashed) unfairness versus the assumed round-trip cost, pre- (blue) versus post-2024 (red), pooled across markets and paths; the dotted line marks the primary cost (\\CostBp~bp). The pre-to-post gap in unconditional unfairness $U_u$ is approximately flat in cost; the gap in conditional unfairness $U_c$ shrinks sharply as cost rises. The unconditional measure is therefore the natural object for the regime comparison.",
     "Pooled unfairness rate $R$ versus the assumed round-trip cost, pre- (blue) versus post-2024 (red); the dotted line marks the primary cost (\\CostBp~bp). A higher cost admits fewer positive tickpaths, so $R$ falls with cost."),
]

_R_BANNER = (
    r"\section*{Figures and tables --- overview (unfairness rate $R$)}"
    "\n\n"
    r"\noindent\textit{R version: the panel-based figures and the regression, cost, "
    r"heatmap, compression, event-study, friction-gradient and daily/event date panels "
    r"plot the unfairness rate $R=$ share of positive tickpaths (a fraction in $[0,1]$). "
    r"Tick-based figures that were not regenerated keep their basis-point form; run "
    r"\texttt{build\_R\_overview.py --with-ticks} with the tick data mounted to switch "
    r"those too.}\\[1ex]"
)


def _write_overview_tex() -> None:
    tex = OVERVIEW_SRC.read_text()
    tex = tex.replace("artifacts/", "artifacts_R/")
    for old, new in _R_CAPTION_REPS:
        tex = tex.replace(old, new)
    tex = tex.replace(r"\section*{Figures and tables --- overview}", _R_BANNER, 1)
    OVERVIEW_DST.write_text(tex)


def _compile() -> None:
    build = HERE / "tmp" / "_buildR"
    build.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"-output-directory={build}", OVERVIEW_DST.name],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    pdf = build / (OVERVIEW_DST.stem + ".pdf")
    if pdf.exists():
        shutil.copy2(pdf, HERE / pdf.name)
        print(f"  wrote {HERE / pdf.name}")
    else:
        print("  [warn] pdflatex did not produce a PDF; check tmp/_buildR log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ticks", action="store_true", help="also regenerate tick-based figures")
    ap.add_argument("--no-compile", action="store_true", help="write artifacts/tex only")
    args = ap.parse_args()

    parquet = _find_panel_parquet()
    from_date, to_date = _date_range_from_name(parquet)
    print(f"panel: {parquet.name}  ({from_date}..{to_date})")

    _seed_from_u_version()
    panel = pl.read_parquet(parquet)
    _set_rate_metric()
    print(f"metric -> R  (dependent={RATE_DEPENDENT})")

    made = _regen_panel(panel)
    print(f"  panel-based R artifacts: {len(made)}")
    if args.with_ticks:
        made += _regen_ticks(from_date, to_date)

    _write_overview_tex()
    print(f"wrote {OVERVIEW_DST.name}")
    if not args.no_compile:
        _compile()


if __name__ == "__main__":
    sys.exit(main())
