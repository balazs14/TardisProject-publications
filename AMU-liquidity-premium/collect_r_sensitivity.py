"""Collect AMU pre/post across the r-sensitivity runs into a LaTeX table.

Reads artifacts_r<r>/text_numbers.tex (falls back to the baseline artifacts/
directory for the config r=0.05) and writes artifacts/r_sensitivity_table.tex.
Usage: python collect_r_sensitivity.py [r1 r2 ...]
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


def main(rvalues: list[str]) -> None:
    rows = []
    for r in rvalues:
        path = find_text_numbers(r)
        if path is None:
            print(f"skip r={r}: no text_numbers.tex found")
            continue
        d = read_macros(path)
        pre, post = float(d["AmuPreBp"]), float(d["AmuPostBp"])
        rows.append((float(r), pre, post, post - pre))
    rows.sort()
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$r$ & AMU pre & AMU post & $\Delta$ (post$-$pre) \\",
        r"\midrule",
    ]
    for r, pre, post, delta in rows:
        lines.append(rf"{r:.3f} & {pre:.2f} & {post:.2f} & {delta:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path("artifacts") / "r_sensitivity_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_RVALUES)
