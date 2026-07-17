"""Schematic figure of the MMA/AMU arbitrage configurations.

Real quoted vs parity-implied (synthetic) option market, three configurations:

  (1) Crossed / disjoint : C_ask < C^syn_bid -> one guaranteed riskless arbitrage.
  (2) Nested (syn inside real): two potential MMA paths
          forward-joincall  = C^syn_bid - C_bid   (left gap)
          backward-joincall = C_ask - C^syn_ask   (right gap)
      (the two join-put payoffs are the negatives of these).
  (3) Straddling (partial overlap): two potential MMA paths
          forward-joincall  = C^syn_bid - C_bid   (left gap)
          forward-joinput   = C^syn_ask - C_ask   (right gap)

Nested/straddling carry no guaranteed profit: a path pays only if the maker is
joined and then hit. This is a hand-built conceptual figure (illustrative levels),
not derived from data. It is regenerated on every rebuild so it always lands in
whichever artifacts directory the paper uses.

Outputs (into the target artifacts dir):
  arbitrage_paths_crossed_noncrossed.pdf   side-by-side 1x3  (used in the paper)
  arbitrage_paths_stacked.pdf              stacked 3x1
  arbitrage_paths_case{1,2,3}_*.pdf        three separate panels

Run standalone:  python figure_arbitrage_paths.py [ARTIFACTS_DIR]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C_REAL = "#2E5A87"   # real quoted market
C_SYN = "#B8860B"    # synthetic (parity-implied) market
C_ARB = "#1B7A3D"    # guaranteed arbitrage
C_POT = "#C0392B"    # potential-MMA path

Y_REAL, Y_SYN, H = 0.66, 0.40, 0.11
Y_ARROW = 0.30

CASES = {
    "crossed": dict(
        title="(1) Crossed / disjoint", tag=r"$C_{ask} < C^{\mathrm{syn}}_{bid}$",
        real=(7, 12), syn=(16, 21),
        bands=[(12, 16, "arb", "guaranteed arbitrage", r"$C^{\mathrm{syn}}_{bid}-C_{ask}$")],
    ),
    "nested": dict(
        title="(2) Nested", tag=r"synthetic inside real",
        real=(7, 21), syn=(12, 16),
        bands=[(7, 12, "pot", "fwd, join call", r"$C^{\mathrm{syn}}_{bid}-C_{bid}$"),
               (16, 21, "pot", "bck, join call", r"$C_{ask}-C^{\mathrm{syn}}_{ask}$")],
    ),
    "straddling": dict(
        title="(3) Straddling", tag=r"partial overlap",
        real=(7, 16), syn=(12, 21),
        bands=[(7, 12, "pot", "fwd, join call", r"$C^{\mathrm{syn}}_{bid}-C_{bid}$"),
               (16, 21, "pot", "fwd, join put", r"$C^{\mathrm{syn}}_{ask}-C_{ask}$")],
    ),
}

SUP = "Real quoted vs parity-implied (synthetic) option market: three configurations"
KEYS = ["crossed", "nested", "straddling"]
_SEPARATE_NAMES = {"crossed": "case1_crossed", "nested": "case2_nested", "straddling": "case3_straddling"}


def _bar(ax, y, lo, hi, color, label, fs):
    ax.add_patch(Rectangle((lo, y), hi - lo, H, facecolor=color, alpha=0.85,
                           edgecolor="black", lw=1.0, zorder=3))
    for x, t in ((lo, "bid"), (hi, "ask")):
        ax.plot([x, x], [y - 0.025, y + H + 0.025], color="black", lw=0.9, zorder=4)
        ax.text(x, y + H + 0.03, t, ha="center", va="bottom", fontsize=fs - 0.5)
    ax.text(3.4, y + H / 2, label, ha="right", va="center", fontsize=fs,
            color=color, fontweight="bold")


def draw_case(ax, key, fs=8.0, title_fs=11.0):
    c = CASES[key]
    _bar(ax, Y_REAL, *c["real"], C_REAL, "real", fs)
    _bar(ax, Y_SYN, *c["syn"], C_SYN, "synthetic", fs)
    for (x0, x1, kind, lab, formula) in c["bands"]:
        col = C_ARB if kind == "arb" else C_POT
        ax.add_patch(Rectangle((x0, Y_SYN - 0.03), x1 - x0,
                               (Y_REAL + H) - (Y_SYN - 0.03),
                               facecolor=col, alpha=0.13, edgecolor="none", zorder=1))
        ax.annotate("", xy=(x1, Y_ARROW), xytext=(x0, Y_ARROW),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
        xc = (x0 + x1) / 2.0
        ax.text(xc, Y_ARROW - 0.05, lab, ha="center", va="top",
                fontsize=fs, color=col, fontweight="bold")
        ax.text(xc, Y_ARROW - 0.135, formula, ha="center", va="top",
                fontsize=fs - 0.8, color=col)
    ax.text(0.5, 1.02, c["title"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=title_fs, fontweight="bold")
    ax.text(0.5, 0.92, c["tag"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=title_fs - 2.6, style="italic", color="#444")
    ax.annotate("", xy=(24.3, 0.045), xytext=(3.8, 0.045),
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.9))
    ax.text(24.3, 0.075, "option value", ha="right", va="bottom",
            fontsize=fs - 1.2, color="#777")
    ax.set_xlim(2.5, 25)
    ax.set_ylim(0, 0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _synthetic_tracking_figure(art: Path) -> Path:
    """Fair real quotes track the matching synthetic quote to within +/- cost.

    The band centres are labelled with the explicit no-cost synthetic prices
    P +/- (F -/+ K) e^{-rT} to stress that the centre is the frictionless
    synthetic market and the +/- cost cushion is what the trading cost buys.
    """
    fig, ax = plt.subplots(figsize=(11.6, 3.9))
    cost = 3.0
    syn_bid, syn_ask = 10.0, 22.0
    c_bid, c_ask = 11.6, 20.4
    y_syn, y_real, h = 0.56, 0.28, 0.10

    ax.add_patch(Rectangle((syn_bid - cost, 0.14), 2 * cost, 0.60, facecolor=C_REAL, alpha=0.10, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((syn_ask - cost, 0.14), 2 * cost, 0.60, facecolor=C_ARB, alpha=0.10, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((syn_bid, y_syn), syn_ask - syn_bid, h, facecolor=C_SYN, alpha=0.85, edgecolor="black", lw=1, zorder=3))
    ax.add_patch(Rectangle((c_bid, y_real), c_ask - c_bid, h, facecolor=C_REAL, alpha=0.85, edgecolor="black", lw=1, zorder=3))

    def tick(x, y, lab, fs, dy=0.03):
        ax.plot([x, x], [y - 0.02, y + h + 0.02], color="black", lw=0.9, zorder=4)
        ax.text(x, y + h + dy, lab, ha="center", va="bottom", fontsize=fs)

    tick(syn_bid, y_syn, r"$P_{bid}+(F_{bid}-K)e^{-rT}$", 8.5)
    tick(syn_ask, y_syn, r"$P_{ask}+(F_{ask}-K)e^{-rT}$", 8.5)
    tick(c_bid, y_real, r"$C_{bid}$", 9.5)
    tick(c_ask, y_real, r"$C_{ask}$", 9.5)
    ax.text(2.2, y_syn + h / 2, "synthetic\n(no-cost)", ha="right", va="center", color="#8a6d0b", fontweight="bold", fontsize=9)
    ax.text(2.2, y_real + h / 2, "real", ha="right", va="center", color=C_REAL, fontweight="bold", fontsize=9)

    for xc, col in ((syn_bid, C_REAL), (syn_ask, C_ARB)):
        ax.annotate("", xy=(xc + cost, 0.10), xytext=(xc - cost, 0.10), arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
        ax.text(xc, 0.055, r"$\pm\,\mathrm{cost}$", ha="center", va="top", color=col, fontsize=8.5)
    ax.text(syn_bid, 0.78, "fair " + r"$C_{bid}$", ha="center", va="bottom", color=C_REAL, fontsize=8.5)
    ax.text(syn_ask, 0.78, "fair " + r"$C_{ask}$", ha="center", va="bottom", color="#1B7A3D", fontsize=8.5)

    ax.annotate("", xy=(33.5, 0.02), xytext=(3.0, 0.02), arrowprops=dict(arrowstyle="->", color="#777", lw=0.9))
    ax.text(33.5, 0.05, "option value", ha="right", va="bottom", fontsize=8, color="#777")
    ax.set_title(r"Fair quotes track the synthetic market within $\pm$cost", fontsize=11.5, fontweight="bold")
    ax.set_xlim(1.5, 34)
    ax.set_ylim(0, 0.90)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.87, bottom=0.05)
    path = art / "amu_fair_band_synthetic.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _schematic_mma_distribution_figure(art: Path) -> Path:
    """Schematic MMA distribution: competitive quoting concentrates the mode at
    -cost, well inside the uniform benchmark band (-2 cost, 0); the mass that
    crosses zero is the executable AMU."""
    cost = 30.0
    edges = np.arange(-100.0, 62.0, 4.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    scale = 9.0
    dens = np.exp(-np.abs(centers + cost) / scale)
    dens *= 1.0 + 0.18 * np.clip((centers + cost) / cost, 0.0, None)  # mild right skew (AMU tail)
    dens /= dens.sum() * 4.0  # normalise to unit area
    peak = dens.max()

    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    # uniform benchmark (same unit area): height 1/(2 cost) over (-2 cost, 0)
    h_bench = 1.0 / (2.0 * cost)
    ax.add_patch(Rectangle((-2 * cost, 0.0), 2 * cost, h_bench, fill=False, edgecolor="#8E44AD", lw=1.4, ls="--", zorder=4))
    ax.text(-2 * cost + 1.0, h_bench + peak * 0.05, r"uniform benchmark (width $2\,$cost)", ha="left", va="bottom", fontsize=8.2, color="#8E44AD")

    colors = [C_POT if c > 0 else C_REAL for c in centers]
    ax.bar(centers, dens, width=3.6, color=colors, alpha=0.80, edgecolor="white", lw=0.3, zorder=2)

    ax.axvline(0.0, color="black", lw=1.0, zorder=3)
    ax.plot([-cost, -cost], [0.0, peak * 1.04], color="#333", ls="--", lw=1.1, zorder=3)
    ax.text(-cost, peak * 1.07, r"$-\,$cost", ha="center", va="bottom", fontsize=9)
    ax.text(0.0, peak * 1.07, "0", ha="center", va="bottom", fontsize=9)
    ax.annotate("no maker arbitrage", xy=(-cost, peak * 0.55), xytext=(-92, peak * 0.86),
                fontsize=8.5, color="#333", arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))
    ax.text(26, peak * 0.30, "AMU\n(executable)", ha="center", va="center", color=C_POT, fontsize=9, fontweight="bold")

    ax.set_xlim(-100, 60)
    ax.set_ylim(0, peak * 1.22)
    ax.set_xlabel("MMA (bp)")
    ax.set_ylabel("density")
    ax.set_title("Schematic MMA distribution", fontsize=11.5, fontweight="bold")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.13)
    path = art / "amu_schematic_mma_distribution.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_arbitrage_paths_figures(artifacts_dir: str | Path) -> list[Path]:
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # A) side by side (1x3) -- the version referenced by the paper.
    figA, axesA = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax, key in zip(axesA, KEYS):
        draw_case(ax, key, fs=8.0)
    figA.suptitle(SUP, y=1.00, fontsize=12.5, fontweight="bold")
    figA.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.04, wspace=0.28)
    path_a = art / "arbitrage_paths_crossed_noncrossed.pdf"
    figA.savefig(path_a, bbox_inches="tight")
    plt.close(figA)
    outputs.append(path_a)

    # B) stacked (3x1)
    figB, axesB = plt.subplots(3, 1, figsize=(7.6, 10.2))
    for ax, key in zip(axesB, KEYS):
        draw_case(ax, key, fs=9.5, title_fs=12.5)
    figB.suptitle(SUP, y=0.995, fontsize=13, fontweight="bold")
    figB.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.02, hspace=0.42)
    path_b = art / "arbitrage_paths_stacked.pdf"
    figB.savefig(path_b, bbox_inches="tight")
    plt.close(figB)
    outputs.append(path_b)

    # C) three separate panels
    for key in KEYS:
        figC, axC = plt.subplots(1, 1, figsize=(6.2, 4.2))
        draw_case(axC, key, fs=9.5, title_fs=13)
        figC.subplots_adjust(left=0.11, right=0.98, top=0.85, bottom=0.04)
        path_c = art / f"arbitrage_paths_{_SEPARATE_NAMES[key]}.pdf"
        figC.savefig(path_c, bbox_inches="tight")
        plt.close(figC)
        outputs.append(path_c)

    # No-AMU schematics: fair quotes track the synthetic market within +/- cost,
    # and the competitive equilibrium concentrates the MMA distribution at -cost.
    outputs.append(_synthetic_tracking_figure(art))
    outputs.append(_schematic_mma_distribution_figure(art))

    return outputs


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    for created in write_arbitrage_paths_figures(target):
        print("wrote", created)
