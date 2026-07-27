"""Schematic of the backward join-call parity arbitrage (illustrative, exaggerated).

Left panel  -- the instrument: a call as a hockey-stick payoff, plus its current
top-of-book quote (bid/ask), to fix intuition that a quote is a price, not a payoff.
Right panel -- the trade: the real call quote sits below its parity-implied
synthetic quote (put + discounted futures). The maker JOINS -- rests a passive buy
at the real call bid -- and, once hit, CROSSES the synthetic market, selling the
parity-equivalent synthetic at its bid. The locked wedge is
C^{syn}_{bid} - C_{bid}, net of the round-trip cost (matches the nested backward
join-call path in Figure 1). Not from data.

Run:  python figure_backward_joincall.py [ARTIFACTS_DIR]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C_REAL = "#2E5A87"   # real quoted (call) market -- blue
C_SYN = "#E8890C"    # synthetic / put -- orange (same on left and right)
C_PUT = C_SYN        # put shares the synthetic orange
C_JOIN = "#2E5A87"   # passive join
C_CROSS = "#C0392B"  # aggressive cross
C_EDGE = "#1B7A3D"   # captured wedge (green)
C_FK = "#555555"     # forward-minus-strike (distinct from the wedge)


def write_backward_joincall_figure(artifacts_dir: str | Path) -> Path:
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.8), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---------- Left: payoffs, quotes, and the synthetic building blocks ----------
    K, F = 10.0, 13.0
    S = np.linspace(2.0, 20.0, 400)
    call_val = 0.5 * ((S - K) + np.sqrt((S - K) ** 2 + 9.0))   # smooth call value
    put_val = call_val - (S - K)                              # parity (illustrative): C - P = S - K
    axL.plot(S, np.clip(S - K, 0.0, None), color="#a9c0d6", lw=1.4, ls="--")
    axL.plot(S, np.clip(K - S, 0.0, None), color="#f2c489", lw=1.4, ls="--")
    axL.plot(S, call_val, color=C_REAL, lw=2.4, label="call value")
    axL.plot(S, put_val, color=C_SYN, lw=2.4, label="put value")
    S0 = F
    cv = 0.5 * ((S0 - K) + np.sqrt((S0 - K) ** 2 + 9.0))
    pv = cv - (S0 - K)

    def qbar(x, lo, hi, color, halfw):
        """Bid/ask bar styled like the right panel: light fill, solid bid/ask lines."""
        axL.add_patch(Rectangle((x - halfw, lo), 2 * halfw, hi - lo, facecolor=color,
                                alpha=0.16, edgecolor="none", zorder=3))
        for y in (lo, hi):
            axL.plot([x - halfw, x + halfw], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=6)

    # real call quote at S0 -- the two building blocks (put, F-K) sit alongside it
    qbar(S0, cv - 0.8, cv + 0.8, C_REAL, 0.42)
    axL.text(S0 + 0.6, cv, "real call\nbid/ask", va="center", ha="left", fontsize=11.5, color=C_REAL, fontweight="bold")
    # put quote -- same vertical line as the call
    qbar(S0, pv - 0.5, pv + 0.5, C_SYN, 0.42)
    axL.text(S0 + 0.6, pv, "put\nbid/ask", va="center", ha="left", fontsize=11.5, color=C_SYN, fontweight="bold")
    # forward minus strike on the axis (distinct colour from the wedge)
    axL.annotate("", xy=(F, -1.0), xytext=(K, -1.0), arrowprops=dict(arrowstyle="<->", color=C_FK, lw=2.2))
    axL.text((K + F) / 2.0, -1.55, r"$F-K$", ha="center", va="top", fontsize=13, color=C_FK, fontweight="bold")
    axL.axvline(K, ls=":", color="#ccc", lw=1.0)
    axL.text(K, -0.4, "K", ha="center", va="top", fontsize=12)
    axL.axvline(F, ls=":", color="#ccc", lw=1.0)
    axL.text(F, -0.4, r"$S_0\!\approx\!F$", ha="center", va="top", fontsize=12)
    axL.set_xlabel("underlying price  $S$", fontsize=13)
    axL.set_ylabel("value", fontsize=13)
    axL.set_title(r"Building blocks: synthetic $=$ put $+\,(F{-}K)e^{-rT}$", fontsize=13.5, fontweight="bold")
    axL.legend(fontsize=10.5, loc="upper center", framealpha=0.9, ncol=2)
    axL.set_ylim(-2.0, 12.0)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # ---------- Right: the two markets overlap (nested/straddling); the maker edge ----------
    axR.set_xlim(0, 10)
    axR.set_ylim(0.0, 9.0)
    cbid, cask = 2.8, 5.6          # real call bid/ask
    sbid, sask = 4.2, 7.0          # synthetic bid/ask -- overlaps the real market in price
    rx0, rx1 = 2.3, 5.3            # real call band x-extent
    sx0, sx1 = 4.4, 7.4           # synthetic band x-extent (overlaps the real band)

    for (x0, x1, lo, hi, color) in ((rx0, rx1, cbid, cask, C_REAL), (sx0, sx1, sbid, sask, C_SYN)):
        axR.add_patch(Rectangle((x0, lo), x1 - x0, hi - lo, facecolor=color, alpha=0.15, edgecolor="none", zorder=2))
        for y in (lo, hi):
            axR.plot([x0, x1], [y, y], color=color, lw=2.6, zorder=5)
    axR.text((rx0 + rx1) / 2, cask + 0.25, "real call", ha="center", va="bottom", fontsize=13, fontweight="bold", color=C_REAL)
    axR.text((sx0 + sx1) / 2, sask + 0.25, r"synthetic  $P+(F{-}K)e^{-rT}$", ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_SYN)
    axR.text(rx0 - 0.15, cbid, "bid", ha="right", va="center", fontsize=11.5, color=C_REAL)
    axR.text(rx0 - 0.15, cask, "ask", ha="right", va="center", fontsize=11.5, color=C_REAL)
    axR.text(sx1 + 0.15, sbid, "bid", ha="left", va="center", fontsize=11.5, color=C_SYN)
    axR.text(sx1 + 0.15, sask, "ask", ha="left", va="center", fontsize=11.5, color=C_SYN)

    # JOIN at the real call bid
    axR.annotate("JOIN\nrest a passive buy\nat the call bid", xy=(3.1, cbid), xytext=(2.7, cbid - 1.9),
                 fontsize=11.5, fontweight="bold", color=C_JOIN, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=C_JOIN, lw=2.2))
    # CROSS at the synthetic bid
    axR.annotate("CROSS\nsell the synthetic:\ntake its bid", xy=(6.6, sbid), xytext=(6.95, sbid - 2.5),
                 fontsize=11.5, fontweight="bold", color=C_CROSS, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=C_CROSS, lw=2.2))
    # captured wedge: the small gap between the real-call bid and the synthetic bid
    xw = 4.7
    axR.annotate("", xy=(xw, sbid), xytext=(xw, cbid), arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=2.6))
    axR.text(xw - 0.25, (cbid + sbid) / 2,
             "locked wedge\n(net of cost)\n$m = C^{\\mathrm{syn}}_{bid}-C_{bid}$",
             fontsize=10.5, color=C_EDGE, fontweight="bold", va="center", ha="right")
    axR.set_title("Join the call, cross the synthetic", fontsize=15, fontweight="bold")
    axR.set_ylabel("price", fontsize=14)
    axR.set_xticks([])
    axR.set_yticks([])
    for s in ("top", "right", "bottom"):
        axR.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.9, bottom=0.11, wspace=0.2)
    out = art / "backward_joincall_schematic.pdf"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    print("wrote", write_backward_joincall_figure(sys.argv[1] if len(sys.argv) > 1 else "artifacts"))
