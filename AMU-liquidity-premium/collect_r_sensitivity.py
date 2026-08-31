"""Collect AMU pre/post across the r-sensitivity runs into a LaTeX table.

``collect_rows()`` parses each run's ``artifacts_r<r>/text_numbers.tex`` (falls back
to the baseline ``artifacts/`` directory for the config r=0.05) into a list of
``(r, pre, post, delta)`` rows, and ``write_table(rows)`` writes
``artifacts/r_sensitivity_table.tex``. This is the sibling of the cost-sensitivity
table (``write_cost_sensitivity_table_from_parquet`` in amu_statistics.py); the two
sit side by side in the paper. Run standalone to refresh from existing runs without
re-sweeping:
    python collect_r_sensitivity.py [r1 r2 ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_RVALUES = ["0.00", "0.025", "0.05", "0.075", "0.10"]
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{([-0-9.]+)\}")


def read_macros(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = MACRO.search(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def find_text_numbers(r: str) -> Path | None:
    candidate = Path(f"artifacts_r{r}") / "text_numbers.tex"
    if candidate.exists():
        return candidate
    # Baseline r=0.05 is the committed run under the artifacts/ symlink.
    if abs(float(r) - 0.05) < 1e-9:
        baseline = Path("artifacts") / "text_numbers.tex"
        if baseline.exists():
            return baseline
    return None


def write_table(rows: list[tuple[float, float, float, float]]) -> Path:
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$r$ & $R$ pre & $R$ post & $\Delta R$ \\",
        r"\midrule",
    ]
    for r, pre, post, delta in rows:
        lines.append(rf"{r:.3f} & {pre:.3f} & {post:.3f} & {delta:+.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path("artifacts") / "r_sensitivity_table.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def collect_rows(rvalues: list[str]) -> list[tuple[float, float, float, float]]:
    rows = []
    for r in rvalues:
        path = find_text_numbers(r)
        if path is None:
            print(f"skip r={r}: no text_numbers.tex found")
            continue
        d = read_macros(path)
        if "AmuPreRate" not in d or "AmuPostRate" not in d:
            print(f"skip r={r}: {path} has no AmuPreRate/AmuPostRate macros -- re-run the r-sweep to refresh")
            continue
        pre, post = float(d["AmuPreRate"]), float(d["AmuPostRate"])
        rows.append((float(r), pre, post, post - pre))
    rows.sort()
    return rows


def main(rvalues: list[str]) -> None:
    rows = collect_rows(rvalues)
    if not rows:
        print("no r-sensitivity rows found; nothing written")
        return
    table = write_table(rows)
    print(f"wrote {table} with {len(rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_RVALUES)
