"""Selective artifact regeneration via the ONLY_RECREATE_ARTIFACTS env var.

Set ONLY_RECREATE_ARTIFACTS to a comma-separated list of tags to regenerate only
those artifacts and skip everything else (each generator keeps its existing files
for the ones it skips). A tag is the LaTeX reference string used in
figures-tables-overview.tex (e.g. ``fig:friction_timeseries``,
``tab:regression_coefficients``); the artifact filename stem (e.g.
``friction_timeseries``) is also accepted.

Examples
--------
    ONLY_RECREATE_ARTIFACTS=fig:friction_timeseries          # one figure
    ONLY_RECREATE_ARTIFACTS=fig:friction_timeseries,tab:robustness_grid

When the variable is unset or empty, everything is regenerated (normal full run).
"""
from __future__ import annotations

import os


def only_recreate() -> set[str] | None:
    """The requested tag set, or None when the full run is requested."""
    raw = os.environ.get("ONLY_RECREATE_ARTIFACTS", "").strip()
    if not raw:
        return None
    return {t.strip() for t in raw.split(",") if t.strip()}


def wanted(*tags: str) -> bool:
    """True if any of ``tags`` was requested (or the full run is active)."""
    only = only_recreate()
    return only is None or any(t in only for t in tags)


def any_wanted(*tags: str) -> bool:
    """True if a generator that produces ``tags`` should run at all. Lets a
    generator early-return (skipping an expensive data load) when none of the
    artifacts it owns were requested."""
    return wanted(*tags)
