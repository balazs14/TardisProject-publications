from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from amu_config import CONFIG, bootstrap_repo_root
REPO_ROOT = bootstrap_repo_root(Path(__file__).resolve())

from amu_panel import build_amu_panel
from amu_metrics import PATHS_PER_TICK
from amu_statistics import dataframe_to_tabular_tex
from tardis.utils import debug_runtime


logger = logging.getLogger(__name__)

# Fixed effects are per-specification (see each RegressionSpec.fixed_effects):
# specs (1)-(2) are pooled OLS with explicit segment dummies, spec (3) adds cell
# fixed effects. Set to False only to disable all fixed effects for debugging.
ENABLE_FIXED_EFFECTS = True



SPEC_DISPLAY_NAMES = {
    "amu_spec1": "(1)",
    "amu_spec1fe": "(2)",
    "amu_spec2": "(3)",
    "amu_spec3": "(4)",
    "amu_spec_depth": "(5)",
    "amu_spec_spread": "(6)",
    "amu_spec_stale": "(7)",
}

# Column order in the combined coefficient table: (1) Post, (2) Post+cell FE,
# (3) Post+segment dummies (pooled OLS), (4) Post+frictions+cell FE.
# The PCA/PC1 index (former column (5)) was dropped from the paper: because PCA
# is an orthogonal rotation, all three components span the same space as the raw
# frictions and reproduce (4) exactly, while PC1 alone is a rank-1 control that
# merely under-controls. The amu_specpca helpers below are retained but unwired.
MAIN_SPEC_ORDER = ("amu_spec1", "amu_spec1fe", "amu_spec2", "amu_spec3", "amu_spec_depth", "amu_spec_spread", "amu_spec_stale")

TERM_DISPLAY_NAMES = {
    "post_2024": "$\\mathrm{Post}_t$",
    "average_put_call_spread_bp": "$\\mathrm{Spr}_{g,t}$",
    "log_mean_min_quote_size_dollar": "$\\mathrm{Depth}_{g,t}$",
    "stale_proxy": "$\\mathrm{Stale}_{g,t}$",
    "liquidity_pc": "$\\mathrm{Liq}_{g,t}$",
    "okx": "$\\mathrm{OKX}_g$",
    "eth": "$\\mathrm{ETH}_g$",
    "atm": "$\\mathrm{ATM}_g$",
    "short_tte": "$\\mathrm{NearExp}_g$",
    "post_2024_x_eth": "$\\mathrm{Post}_t \\times \\mathrm{ETH}_g$",
    "post_2024_x_atm": "$\\mathrm{Post}_t \\times \\mathrm{ATM}_g$",
    "post_2024_x_short_tte": "$\\mathrm{Post}_t \\times \\mathrm{NearExp}_g$",
}


filters_cfg = CONFIG["filters"]
regression_cfg = CONFIG["regression"]

post_2024_start = pd.Timestamp(regression_cfg["post_2024_start"]).date()
nonatm_distance = float(regression_cfg["nonatm_distance"])
short_tte_cutoff = float(regression_cfg["short_tte_cutoff"])
rel_strike_min_default = float(filters_cfg["rel_strike_min"])
rel_strike_max_default = float(filters_cfg["rel_strike_max"])
spread_bp_max_default = int(filters_cfg["spread_bp_max"])
min_quote_size_dollar_default = float(filters_cfg["min_quote_size_dollar"])
min_mma_bp_default = float(filters_cfg["min_mma_bp"])
max_mma_bp_default = float(filters_cfg["max_mma_bp"])


@dataclass(frozen=True)
class RegressionSpec:
    name: str
    dependent: str
    regressors: tuple[str, ...]
    fixed_effects: tuple[str, ...]


FE_ABSORBED_MAIN_EFFECTS = {"eth", "atm", "short_tte"}


def build_liquidity_analysis_panel(
    panel: pl.DataFrame | None = None,
    **build_kwargs,
) -> pd.DataFrame:
    if panel is None:
        logger.debug("build_liquidity_analysis_panel building panel from source with kwargs=%s", sorted(build_kwargs.keys()))
        panel = build_amu_panel(**_build_kwargs_with_defaults(build_kwargs))
    else:
        logger.debug("build_liquidity_analysis_panel received prebuilt polars panel shape=(%s, %s)", panel.height, panel.width)

    frame = panel.to_pandas().copy()
    frame["day"] = pd.to_datetime(frame["day"])
    frame["mean_mma_bp"] = 0.5 * (frame["mean_mma_bck_bp"] + frame["mean_mma_fwd_bp"])
    frame["std_mma_bp"] = frame[["mean_mma_bck_bp", "mean_mma_fwd_bp"]].std(axis=1, ddof=0)
    frame["post_2024"] = (frame["day"].dt.date >= post_2024_start).astype(float)
    frame["average_put_call_spread_bp"] = 0.5 * (
        frame["mean_call_spread_bp"] + frame["mean_put_spread_bp"]
    )
    frame["log_mean_min_quote_size_dollar"] = np.log(frame["mean_min_quote_size_dollar"].clip(lower=1.0))
    frame["stale_proxy"] = frame[["frac_call_stale", "frac_put_stale", "frac_spot_stale"]].mean(axis=1)
    frame["eth"] = frame["ref_sym"].str.contains("ETH", na=False).astype(float)
    frame["okx"] = frame["exchange"].str.contains("okex", case=False, na=False).astype(float)
    frame["nonatm"] = (frame["rel_strike_bucket"].sub(1.0).abs() > nonatm_distance).astype(float)
    frame["atm"] = 1.0 - frame["nonatm"]
    frame["short_tte"] = (frame["tte_bucket"] <= short_tte_cutoff).astype(float)
    frame["post_2024_x_eth"] = frame["post_2024"] * frame["eth"]
    frame["post_2024_x_atm"] = frame["post_2024"] * frame["atm"]
    frame["post_2024_x_short_tte"] = frame["post_2024"] * frame["short_tte"]
    assert AMU_DEPENDENT in frame.columns, (
        f"Expected {AMU_DEPENDENT} in panel. Recreate cached amu_panel parquet files "
        "(they now carry mean_amu_conditional_bp and mean_amu_unconditional_bp)."
    )
    frame["cell_id"] = (
        frame["exchange"]
        + "|"
        + frame["ref_sym"]
        + "|"
        + frame["rel_strike_bucket"].round(6).astype(str)
        + "|"
        + frame["tte_bucket"].round(6).astype(str)
    )
    logger.debug(
        "build_liquidity_analysis_panel output shape=%s day_range=%s..%s exchanges=%s ref_syms=%s",
        frame.shape,
        frame["day"].min() if not frame.empty else None,
        frame["day"].max() if not frame.empty else None,
        sorted(frame["exchange"].dropna().unique().tolist()) if "exchange" in frame else [],
        sorted(frame["ref_sym"].dropna().unique().tolist()) if "ref_sym" in frame else [],
    )
    return frame.sort_values(["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"])


@debug_runtime("filter_analysis_panel")
def filter_analysis_panel(
    frame: pd.DataFrame,
    rel_strike_min: float = rel_strike_min_default,
    rel_strike_max: float = rel_strike_max_default,
    spread_bp_max: int = spread_bp_max_default,
    min_quote_size_dollar: float = min_quote_size_dollar_default,
    min_mma_bp: float = min_mma_bp_default,
    max_mma_bp: float = max_mma_bp_default,
    *,
    apply_rel_strike: bool = True,
) -> pd.DataFrame:
    required = {
        "mean_call_spread_bp",
        "mean_put_spread_bp",
        "mean_min_quote_size_dollar",
        "mean_mma_bp",
    }
    if apply_rel_strike:
        required.add("mean_rel_strike")
    missing = required - set(frame.columns)
    assert not missing, f"Missing panel filter columns: {sorted(missing)}"

    mask = (
        pd.to_numeric(frame["mean_call_spread_bp"], errors="coerce").between(0, spread_bp_max)
        & pd.to_numeric(frame["mean_put_spread_bp"], errors="coerce").between(0, spread_bp_max)
        & pd.to_numeric(frame["mean_min_quote_size_dollar"], errors="coerce").ge(min_quote_size_dollar)
        & pd.to_numeric(frame["mean_mma_bp"], errors="coerce").between(min_mma_bp, max_mma_bp)
    )
    if apply_rel_strike:
        mask = mask & pd.to_numeric(frame["mean_rel_strike"], errors="coerce").between(rel_strike_min, rel_strike_max)
    filtered = frame.loc[mask].copy()
    logger.debug(
        "filter_analysis_panel input_rows=%d output_rows=%d apply_rel_strike=%s rel_strike=[%s, %s] spread_bp_max=%s min_quote_size_dollar=%s mma_bp=[%s, %s]",
        len(frame),
        len(filtered),
        apply_rel_strike,
        rel_strike_min,
        rel_strike_max,
        spread_bp_max,
        min_quote_size_dollar,
        min_mma_bp,
        max_mma_bp,
    )
    return filtered


# The three correlated liquidity/friction descriptors summarized by the PCA
# control below (spec (5)). Order matters for the reported loadings table.
LIQUIDITY_PCA_INPUTS: tuple[str, ...] = (
    "log_mean_min_quote_size_dollar",
    "average_put_call_spread_bp",
    "stale_proxy",
)
LIQUIDITY_PCA_INPUT_LABELS = {
    "log_mean_min_quote_size_dollar": "Depth",
    "average_put_call_spread_bp": "Spread",
    "stale_proxy": "Stale",
}
# Populated by add_liquidity_pc(): per-regime PC1 loadings and variance share,
# consumed by write_liquidity_pca_table() for the manuscript.
_LIQUIDITY_PC_LOADINGS: dict[str, dict[str, float]] = {}


def add_liquidity_pc(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``liquidity_pc``: the first principal component of the three
    standardized liquidity descriptors (Depth, Spread, Stale), fit *separately*
    on the pre- and post-2024 analysis samples.

    The three controls are strongly collinear; the leading component is a single
    synthetic liquidity index. It is oriented so higher values mean *worse*
    liquidity (positive loading on Spread) and z-scored within each regime, so
    the pooled column is comparable across the break. Rows outside the analysis
    filter get NaN and drop out of the regression exactly like the other specs.
    """
    frame = frame.copy()
    frame["liquidity_pc"] = np.nan
    analysis_index = filter_analysis_panel(frame).index
    _LIQUIDITY_PC_LOADINGS.clear()
    for regime, label in ((0.0, "pre"), (1.0, "post")):
        idx = frame.index[(frame.index.isin(analysis_index)) & (frame["post_2024"] == regime)]
        block = frame.loc[idx, list(LIQUIDITY_PCA_INPUTS)].apply(pd.to_numeric, errors="coerce")
        idx = block.dropna().index
        x = frame.loc[idx, list(LIQUIDITY_PCA_INPUTS)].to_numpy(dtype=float)
        if len(idx) < len(LIQUIDITY_PCA_INPUTS) + 1:
            logger.warning("add_liquidity_pc regime=%s has too few rows (%d)", label, len(idx))
            continue
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=0)
        sd[sd == 0.0] = 1.0
        z = (x - mu) / sd
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        loading = vt[0]
        # Orient so the component rises with illiquidity (positive on Spread).
        spread_pos = LIQUIDITY_PCA_INPUTS.index("average_put_call_spread_bp")
        if loading[spread_pos] < 0:
            loading = -loading
        score = z @ loading
        score_sd = score.std(ddof=0)
        frame.loc[idx, "liquidity_pc"] = score / (score_sd if score_sd > 0 else 1.0)
        var_ratio = float(singular[0] ** 2 / np.sum(singular ** 2))
        _LIQUIDITY_PC_LOADINGS[label] = {
            **{name: float(loading[i]) for i, name in enumerate(LIQUIDITY_PCA_INPUTS)},
            "var_explained": var_ratio,
            "nobs": float(len(idx)),
        }
    logger.debug("add_liquidity_pc loadings=%s", _LIQUIDITY_PC_LOADINGS)
    return frame


def write_liquidity_pca_table(output_dir: str | Path) -> Path:
    """Emit the PC1 loadings + variance-explained table (pre vs post 2024)."""
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Pre-2024 & Post-2024 \\",
        r"\midrule",
    ]
    for name in LIQUIDITY_PCA_INPUTS:
        label = LIQUIDITY_PCA_INPUT_LABELS[name]
        pre = _LIQUIDITY_PC_LOADINGS.get("pre", {}).get(name, float("nan"))
        post = _LIQUIDITY_PC_LOADINGS.get("post", {}).get(name, float("nan"))
        lines.append(rf"{label} & {pre:.3f} & {post:.3f} \\")
    lines.append(r"\midrule")
    pre_v = _LIQUIDITY_PC_LOADINGS.get("pre", {}).get("var_explained", float("nan"))
    post_v = _LIQUIDITY_PC_LOADINGS.get("post", {}).get("var_explained", float("nan"))
    lines.append(rf"Var.\ explained & {pre_v:.0%} & {post_v:.0%} \\".replace("%", r"\%"))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path = Path(output_dir) / "regression_liquidity_pca_table.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.debug("write_liquidity_pca_table wrote %s loadings=%s", path, _LIQUIDITY_PC_LOADINGS)
    return path


# AMU dependent variable for every spec below. The panel carries both explicit
# flavours; the regressions use the UNCONDITIONAL (per-quote = frequency x
# conditional) AMU. Switch to "mean_amu_conditional_bp" for the conditional size.
AMU_DEPENDENT = "mean_amu_uncond_bp_c30"

# Covariance estimator for the reported standard errors. Cluster keys must be
# columns carried in the analysis panel. An empty tuple falls back to HC1
# (heteroskedasticity-robust, independence across cell-days). The default is
# two-way clustering by cell and day (Cameron--Gelbach--Miller): it allows a
# cell's residuals to be correlated across its days AND a day's residuals to be
# correlated across cells, which is the right structure for this panel. Only the
# covariance/standard errors change; point estimates are identical.
SE_CLUSTERS: tuple[str, ...] = ("cell_id", "day")


def _se_description() -> str:
    """Human-readable SE label, emitted as \\RegSEdesc for the table captions."""
    keys = tuple(SE_CLUSTERS)
    if not keys:
        return "HC1"
    pretty = {"cell_id": "cell", "day": "day"}
    names = [pretty.get(k, k) for k in keys]
    if len(names) == 1:
        return f"cluster-robust (by {names[0]})"
    joined = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 2 else " and ".join(names)
    return f"two-way cluster-robust (by {joined})"

# Nested AMU specifications (HC1 SEs):
#   (1) Post                          -> RQ2 (regime effect)
#   (2) + OKX/ETH/ATM/NearExp         -> RQ1 (cross-sectional segments, linear dummies)
#   (3) Post + Depth/Spread/Stale, cell fixed effects -> RQ3. The segment/bucket
#       dependence is nonlinear, so it is absorbed by cell fixed effects rather
#       than the linear dummies of (2); a residual Post is compression not
#       explained by the frictions.
_SPEC1_REGRESSORS = ("post_2024",)
_SPEC2_REGRESSORS = _SPEC1_REGRESSORS + ("okx", "eth", "atm", "short_tte")
_SPEC3_REGRESSORS = (
    "post_2024",
    "log_mean_min_quote_size_dollar",
    "average_put_call_spread_bp",
    "stale_proxy",
)


def amu_spec1_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec1", dependent=AMU_DEPENDENT, regressors=_SPEC1_REGRESSORS, fixed_effects=())


def amu_spec2_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec2", dependent=AMU_DEPENDENT, regressors=_SPEC2_REGRESSORS, fixed_effects=())


def amu_spec3_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec3", dependent=AMU_DEPENDENT, regressors=_SPEC3_REGRESSORS, fixed_effects=("cell_id",))


# Column (5): the three collinear frictions of spec (4) replaced by their first
# principal component (a single synthetic liquidity index), with cell FE.
_SPECPCA_REGRESSORS = ("post_2024", "liquidity_pc")


def amu_specpca_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_specpca", dependent=AMU_DEPENDENT, regressors=_SPECPCA_REGRESSORS, fixed_effects=("cell_id",))


def run_amu_specpca_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_specpca_regression_spec(), panel=panel, **build_kwargs)


def amu_spec1fe_regression_spec() -> RegressionSpec:
    # Column (4): specification (1) with cell fixed effects added -- the post-2024
    # indicator plus cell FE and no other controls (the within-cell regime effect).
    return RegressionSpec(name="amu_spec1fe", dependent=AMU_DEPENDENT, regressors=_SPEC1_REGRESSORS, fixed_effects=("cell_id",))


def run_amu_spec1fe_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec1fe_regression_spec(), panel=panel, **build_kwargs)


def run_amu_spec1_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec1_regression_spec(), panel=panel, **build_kwargs)


def run_amu_spec2_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec2_regression_spec(), panel=panel, **build_kwargs)


def run_amu_spec3_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec3_regression_spec(), panel=panel, **build_kwargs)


# Single-friction specs (5)-(7): Post plus ONE friction, with cell fixed effects.
# They show how much of the Post effect each friction absorbs on its own. Post
# stays large with any single friction; only all three together (spec (4)) remove it.
def amu_spec_depth_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec_depth", dependent=AMU_DEPENDENT,
                          regressors=("post_2024", "log_mean_min_quote_size_dollar"), fixed_effects=("cell_id",))


def amu_spec_spread_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec_spread", dependent=AMU_DEPENDENT,
                          regressors=("post_2024", "average_put_call_spread_bp"), fixed_effects=("cell_id",))


def amu_spec_stale_regression_spec() -> RegressionSpec:
    return RegressionSpec(name="amu_spec_stale", dependent=AMU_DEPENDENT,
                          regressors=("post_2024", "stale_proxy"), fixed_effects=("cell_id",))


def run_amu_spec_depth_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec_depth_regression_spec(), panel=panel, **build_kwargs)


def run_amu_spec_spread_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec_spread_regression_spec(), panel=panel, **build_kwargs)


def run_amu_spec_stale_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
    return _run_regression(amu_spec_stale_regression_spec(), panel=panel, **build_kwargs)


def write_text_macros(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Emit scalar numbers used in the manuscript prose as LaTeX \\newcommand macros.

    Every magnitude the text cites should come from here (or from an \\input table),
    never be typed by hand. ``frame`` is the filtered analysis panel.
    """
    cost_bp = int(round(float(CONFIG["pcp"]["cost_per_notional"]) * 10000.0))
    amu = pd.to_numeric(frame[AMU_DEPENDENT], errors="coerce")
    weights = pd.to_numeric(frame["n_obs"], errors="coerce") if "n_obs" in frame.columns else pd.Series(1.0, index=frame.index)
    post = pd.to_numeric(frame["post_2024"], errors="coerce")

    def wmean(mask: pd.Series) -> float:
        a, w = amu[mask], weights[mask]
        ok = a.notna() & w.notna() & (w > 0)
        return float(np.average(a[ok].to_numpy(float), weights=w[ok].to_numpy(float))) if ok.any() else float("nan")

    macros = {
        "CostBp": f"{cost_bp}",
        "AmuMeanBp": f"{wmean(amu.notna()):.2f}",
        "AmuPreBp": f"{wmean(post == 0.0):.2f}",
        "AmuPostBp": f"{wmean(post == 1.0):.2f}",
    }
    lines = ["% Auto-generated by write_text_macros; do not edit by hand."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    # SE label kept in lock-step with the covariance actually used, so a caption
    # can never say "clustered" while the numbers are HC1. Preamble sets a
    # \providecommand fallback, so this renews it.
    lines.append(rf"\renewcommand{{\RegSEdesc}}{{{_se_description()}}}")
    path = Path(output_dir) / "text_numbers.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.debug("write_text_macros wrote %s macros=%s", path, macros)
    return path


# Cost tag the headline unconditional-AMU dependent is built at (e.g. "c30").
_AMU_COST_TAG = AMU_DEPENDENT.rsplit("_", 1)[-1]

_FREQ_SIZE_CHANNELS = (
    ("log_amu_freq", "Unfairness rate $R$"),
    ("log_amu_condsize", "Cond.\\ size $U_c$"),
    ("log_amu_uncond", "$U_u$"),
)


def _frequency_size_channels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the extensive/intensive AMU channels to a copy of the analysis panel.

    At the headline cost tag the unconditional AMU factors exactly as
    ``uncond = phi * size`` where ``phi`` = positive tickpaths / total tickpaths
    (extensive margin) and ``size`` = positive-part sum / positive tickpaths
    (intensive margin). Logs are taken on the positive-mass sample (poscount > 0),
    so a log-linear regression of each channel on the same design decomposes the
    Post effect additively: beta(log uncond) = beta(log phi) + beta(log size)."""
    possum = f"amu_possum_bp_{_AMU_COST_TAG}"
    poscount = f"amu_poscount_{_AMU_COST_TAG}"
    missing = {possum, poscount, "n_obs"} - set(panel.columns)
    assert not missing, f"frequency/size decomposition needs {sorted(missing)} in the panel"
    out = panel.copy()
    total_tickpaths = PATHS_PER_TICK * pd.to_numeric(out["n_obs"], errors="coerce")
    pos_count = pd.to_numeric(out[poscount], errors="coerce")
    pos_sum = pd.to_numeric(out[possum], errors="coerce")
    positive = pos_count > 0
    out["amu_freq"] = pos_count / total_tickpaths
    out["amu_condsize_bp"] = pos_sum.where(positive) / pos_count.where(positive)
    out["amu_uncond_channel_bp"] = pos_sum / total_tickpaths
    out["log_amu_freq"] = np.log(out["amu_freq"].where(positive))
    out["log_amu_condsize"] = np.log(out["amu_condsize_bp"].where(positive))
    out["log_amu_uncond"] = np.log(out["amu_uncond_channel_bp"].where(positive))
    return out


def write_frequency_size_table(panel: pd.DataFrame, output_dir: str | Path) -> Path:
    """Two-part (extensive/intensive) decomposition of the post-2024 AMU
    compression. Regresses log(phi), log(size) and log(uncond) on Post with cell
    fixed effects on a common positive-mass sample; the Post semi-elasticities are
    additive by construction (log uncond = log phi + log size)."""
    enriched = _frequency_size_channels(panel)
    columns: list[str] = []
    post_row: dict[str, str] = {}
    r2_row: dict[str, str] = {}
    obs_row: dict[str, str] = {}
    for dependent, label in _FREQ_SIZE_CHANNELS:
        spec = RegressionSpec(
            name=f"freqsize_{dependent}",
            dependent=dependent,
            regressors=_SPEC1_REGRESSORS,
            fixed_effects=("cell_id",),
        )
        result = _run_regression(spec, panel=enriched)
        row = result.loc[result["term"] == "post_2024"].iloc[0]
        coef = float(row["coefficient"])
        columns.append(label)
        post_row[label] = f"{_format_signed_decimal(coef, decimals=3)}{_significance_stars(float(row['t_stat']))}"
        r2_row[label] = f"{float(row['r2']):.3f}"
        obs_row[label] = f"{int(row['nobs']):,}".replace(",", "{,}")
    table = pd.DataFrame(
        [post_row, r2_row, obs_row],
        index=[r"$\mathrm{Post}_t$", r"Total $R^2$", "Observations"],
        columns=columns,
    )
    table.index.name = "Channel"
    path = Path(output_dir) / "frequency_size_decomposition_table.tex"
    dataframe_to_tabular_tex(table, path)
    logger.debug("write_frequency_size_table wrote %s", path)
    return path


_COST_TABLE_BP = (10, 20, 30, 40, 50)


def _panel_pooled_amu(sub: pd.DataFrame, cost_bp: int, *, conditional: bool) -> float:
    """n_obs-pooled AMU (bp) over a panel subset at round-trip cost ``cost_bp``.

    Built from the same accumulators (positive-part sum, positive count, tick
    count) as the headline regression dependent, so the unconditional value at the
    baseline cost reproduces the cell-weighted panel mean exactly:
    ``sum(possum) / (4 * sum(n_obs))``."""
    possum = pd.to_numeric(sub[f"amu_possum_bp_c{cost_bp:02d}"], errors="coerce").sum()
    if conditional:
        denom = pd.to_numeric(sub[f"amu_poscount_c{cost_bp:02d}"], errors="coerce").sum()
    else:
        denom = PATHS_PER_TICK * pd.to_numeric(sub["n_obs"], errors="coerce").sum()
    return float(possum / denom) if denom > 0 else float("nan")


def write_cost_sensitivity_table_from_panel(panel: pd.DataFrame, output_dir: str | Path) -> Path:
    """Unconditional AMU pre/post-2024 across a small grid of round-trip costs, on
    the same filtered, n_obs-weighted panel the regressions use -- so the baseline
    -cost column reproduces the headline pre/post AMU exactly. Replaces the old
    pooled-tick-histogram table, which used a different (tick-level) estimand."""
    filtered = filter_analysis_panel(panel)
    pre = filtered[filtered["post_2024"] == 0.0]
    post = filtered[filtered["post_2024"] == 1.0]
    lines = [r"\begin{tabular}{rrrr}", r"\toprule", r"cost (bp) & pre & post & $\Delta$ \\", r"\midrule"]
    for cost_bp in _COST_TABLE_BP:
        pre_v = _panel_pooled_amu(pre, cost_bp, conditional=False)
        post_v = _panel_pooled_amu(post, cost_bp, conditional=False)
        lines.append(rf"{cost_bp} & {pre_v:.2f} & {post_v:.2f} & {post_v - pre_v:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = Path(output_dir) / "cost_sensitivity_table.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.debug("write_cost_sensitivity_table_from_panel wrote %s", path)
    return path


@debug_runtime("write_regression_tables")
def write_regression_tables(output_dir: str | Path, **build_kwargs) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    panel = build_liquidity_analysis_panel(**build_kwargs)
    combined_results: list[pd.DataFrame] = []
    logger.debug(
        "write_regression_tables output_dir=%s panel_shape=%s",
        output_root,
        panel.shape,
    )
    for runner in (
        run_amu_spec1_regression,
        run_amu_spec1fe_regression,
        run_amu_spec2_regression,
        run_amu_spec3_regression,
        run_amu_spec_depth_regression,
        run_amu_spec_spread_regression,
        run_amu_spec_stale_regression,
    ):
        result = runner(panel=panel)
        combined_results.append(result)
    results = pd.concat(combined_results, ignore_index=True)
    summary_table = _regression_summary_table(results)
    coefficient_table = _regression_coefficient_table(results)

    for spec_name in results["specification"].dropna().unique().tolist():
        spec_slice = results.loc[results["specification"] == spec_name].copy()
        path = output_root / f"{spec_name}.csv"
        spec_slice.to_csv(path, index=False)
        paths[str(spec_name)] = path
        logger.debug("write_regression_tables wrote %s rows=%d", path, len(spec_slice))

    summary_path = output_root / "regression_model_summary_table.tex"
    coefficient_path = output_root / "regression_coefficients_table.tex"
    effects_path = output_root / "regression_effects_table.tex"
    dataframe_to_tabular_tex(summary_table, summary_path)
    dataframe_to_tabular_tex(coefficient_table, coefficient_path)
    dataframe_to_tabular_tex(_regression_effects_table(results), effects_path)
    paths["regression_model_summary_table"] = summary_path
    paths["regression_coefficients_table"] = coefficient_path
    paths["regression_effects_table"] = effects_path
    paths["text_numbers"] = write_text_macros(filter_analysis_panel(panel), output_root)
    paths["frequency_size_decomposition_table"] = write_frequency_size_table(panel, output_root)
    paths["cost_sensitivity_table"] = write_cost_sensitivity_table_from_panel(panel, output_root)
    logger.debug(
        "write_regression_tables wrote summary_tex=%s coefficient_tex=%s combined_rows=%d",
        summary_path,
        coefficient_path,
        len(results),
    )
    return paths, panel


@debug_runtime("_run_regression")
def _run_regression(
    spec: RegressionSpec,
    panel: pd.DataFrame | pl.DataFrame | None = None,
    **build_kwargs,
) -> pd.DataFrame:
    effective_spec = _effective_regression_spec(spec)
    frame = filter_analysis_panel(_coerce_analysis_panel(panel, **build_kwargs))
    regression_frame = _regression_frame(frame, effective_spec)
    logger.debug(
        "_run_regression spec=%s dependent=%s regressors=%s effective_regressors=%s fixed_effects=%s fixed_effects_enabled=%s regression_shape=%s",
        effective_spec.name,
        effective_spec.dependent,
        spec.regressors,
        effective_spec.regressors,
        effective_spec.fixed_effects,
        ENABLE_FIXED_EFFECTS,
        regression_frame.shape,
    )
    _warn_on_constant_regressors(regression_frame, effective_spec)
    coefficients, stderr, t_stat, r2 = _fit_ols_robust(regression_frame, effective_spec)
    fixed_effects_components = _fixed_effects_components_for_spec(effective_spec)
    dependent_std = float(np.std(pd.to_numeric(regression_frame[effective_spec.dependent], errors="coerce").to_numpy(dtype=float)))
    rows = []
    for term in effective_spec.regressors:
        regressor_std = float(np.std(pd.to_numeric(regression_frame[term], errors="coerce").to_numpy(dtype=float)))
        beta = float(coefficients[term])
        economic_significance = np.nan if dependent_std <= 0.0 else beta * (regressor_std / dependent_std)
        rows.append(
            {
                "specification": effective_spec.name,
                "dependent": effective_spec.dependent,
                "term": term,
                "coefficient": coefficients[term],
                "std_error": stderr[term],
                "t_stat": t_stat[term],
                "nobs": len(regression_frame),
                "r2": r2,
                "economic_significance": economic_significance,
                "fixed_effects": "|".join(fixed_effects_components),
            }
        )
    result = pd.DataFrame(rows)
    logger.debug("_run_regression spec=%s result_rows=%d r2=%.6f", effective_spec.name, len(result), r2)
    return result


def _effective_regression_spec(spec: RegressionSpec) -> RegressionSpec:
    # Each spec carries its own fixed effects; regressors are kept as declared.
    if not ENABLE_FIXED_EFFECTS:
        return RegressionSpec(spec.name, spec.dependent, spec.regressors, ())
    return spec


def _coerce_analysis_panel(panel: pd.DataFrame | pl.DataFrame | None, **build_kwargs) -> pd.DataFrame:
    if panel is None:
        logger.debug("_coerce_analysis_panel building analysis panel from source")
        return build_liquidity_analysis_panel(**build_kwargs)
    if isinstance(panel, pl.DataFrame):
        logger.debug("_coerce_analysis_panel converting polars panel shape=(%s, %s)", panel.height, panel.width)
        return build_liquidity_analysis_panel(panel=panel)
    logger.debug("_coerce_analysis_panel copying pandas panel shape=%s", panel.shape)
    return panel.copy()


def _build_kwargs_with_defaults(build_kwargs: dict[str, object]) -> dict[str, object]:
    kwargs = dict(build_kwargs)
    kwargs.setdefault("raw_data_dir", str(REPO_ROOT / "datasets/{exchange}/"))
    return kwargs


def _regression_frame(frame: pd.DataFrame, spec: RegressionSpec) -> pd.DataFrame:
    base = [spec.dependent, *spec.regressors, *spec.fixed_effects]
    # Carry the cluster keys (even when they are not regressors or fixed effects)
    # so the fitter can build cluster-robust standard errors. They are ignored by
    # the design matrix, which only reads regressors + fixed effects.
    cluster_cols = [c for c in SE_CLUSTERS if c in frame.columns and c not in base]
    columns = base + cluster_cols
    regression_frame = frame.loc[:, columns].dropna().reset_index(drop=True)
    logger.debug(
        "_regression_frame spec=%s selected_columns=%s rows=%d",
        spec.name,
        columns,
        len(regression_frame),
    )
    return regression_frame


def _warn_on_constant_regressors(frame: pd.DataFrame, spec: RegressionSpec) -> None:
    for term in spec.regressors:
        values = pd.to_numeric(frame[term], errors="coerce").dropna()
        if values.empty:
            logger.warning(
                "regression '%s': regressor '%s' has no non-null numeric values after filtering",
                spec.name,
                term,
            )
            continue
        if values.nunique(dropna=True) <= 1:
            logger.warning(
                "regression '%s': regressor '%s' is constant (value=%s); coefficient may be weakly identified",
                spec.name,
                term,
                values.iloc[0],
            )


def _regression_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.loc[results["specification"].isin(MAIN_SPEC_ORDER)]
        .drop_duplicates(subset=["specification"])
        .loc[:, ["specification", "dependent", "nobs", "r2"]]
        .copy()
    )
    summary["specification"] = summary["specification"].map(SPEC_DISPLAY_NAMES).fillna(summary["specification"])
    summary["nobs"] = summary["nobs"].map(lambda value: f"{int(value):,}")
    summary["r2"] = summary["r2"].map(lambda value: f"{value:.3f}")
    summary["dependent"] = summary["dependent"].map({
        "mean_mma_bp": "MMA",
        "mean_amu_conditional_bp": "$U_c$",
        "mean_amu_unconditional_bp": "$U_u$",
        "std_mma_bp": "Std MMA",
    }).fillna(summary["dependent"])
    return summary.set_index("specification")


def _regression_coefficient_table(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    spec_order = [spec for spec in MAIN_SPEC_ORDER if spec in results["specification"].unique()]
    term_order = [
        "post_2024",
        "okx",
        "eth",
        "atm",
        "short_tte",
        "log_mean_min_quote_size_dollar",
        "average_put_call_spread_bp",
        "stale_proxy",
    ]

    # One column per specification. Each coefficient cell is stacked over three
    # rows: the coefficient (with significance stars), the HC1 standard error in
    # parentheses, and the standardized (economic-significance) effect in a bold,
    # bracketed third row. Folding the old separate Eff column into this third
    # row halves the column count, so more specifications fit across the page.
    spec_labels = [SPEC_DISPLAY_NAMES.get(spec, spec) for spec in spec_order]
    output_rows: list[dict[str, str]] = []
    output_index: list[str] = []

    for term in term_order:
        if term not in results["term"].values:
            continue
        coef_row = {column: "" for column in spec_labels}
        se_row = {column: "" for column in spec_labels}
        eff_row = {column: "" for column in spec_labels}
        for spec, spec_label in zip(spec_order, spec_labels):
            spec_slice = results.loc[(results["specification"] == spec) & (results["term"] == term)]
            if spec_slice.empty:
                continue
            row = spec_slice.iloc[0]
            coef_row[spec_label] = (
                f"{_format_signed_decimal(float(row['coefficient']), decimals=3)}{_significance_stars(float(row['t_stat']))}"
            )
            se_row[spec_label] = f"({float(row['std_error']):.3f})"
            eff_row[spec_label] = _format_bracket_bold_percent(row["economic_significance"])
        output_rows.extend([coef_row, se_row, eff_row])
        output_index.extend([TERM_DISPLAY_NAMES.get(term, term), "", ""])

    for summary_label, key in (("Total $R^2$", "r2"), ("Observations", "nobs"), ("Fixed effects", "fixed_effects")):
        summary_row = {column: "" for column in spec_labels}
        for spec, spec_label in zip(spec_order, spec_labels):
            spec_slice = results.loc[results["specification"] == spec]
            if spec_slice.empty:
                continue
            row = spec_slice.iloc[0]
            if key == "r2":
                value = f"{float(row['r2']):.3f}"
            elif key == "nobs":
                value = f"{int(row['nobs']):,}"
            else:
                fixed_effects_components = _fixed_effects_components_from_results(results, spec)
                value = ", ".join(fixed_effects_components)
            summary_row[spec_label] = value
        output_rows.append(summary_row)
        output_index.append(summary_label)

    table = pd.DataFrame(output_rows, index=output_index, columns=spec_labels)
    table.index.name = "Term"
    return table


def _regression_effects_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compact one-row-per-term table showing only the standardized (percent)
    effect with significance stars, plus the R^2 and fixed-effects rows. A slide
    -friendly digest of the full coefficient table."""
    results = results.copy()
    spec_order = [spec for spec in MAIN_SPEC_ORDER if spec in results["specification"].unique()]
    spec_labels = [SPEC_DISPLAY_NAMES.get(spec, spec) for spec in spec_order]
    term_order = [
        "post_2024", "okx", "eth", "atm", "short_tte",
        "log_mean_min_quote_size_dollar", "average_put_call_spread_bp", "stale_proxy",
    ]
    rows: list[dict[str, str]] = []
    index: list[str] = []
    for term in term_order:
        if term not in results["term"].values:
            continue
        row = {label: "" for label in spec_labels}
        for spec, label in zip(spec_order, spec_labels):
            spec_slice = results.loc[(results["specification"] == spec) & (results["term"] == term)]
            if spec_slice.empty:
                continue
            r = spec_slice.iloc[0]
            row[label] = _format_pct_effect(r["economic_significance"], r["t_stat"])
        rows.append(row)
        index.append(TERM_DISPLAY_NAMES.get(term, term))
    for summary_label, key in (("Total $R^2$", "r2"), ("Fixed effects", "fixed_effects")):
        row = {label: "" for label in spec_labels}
        for spec, label in zip(spec_order, spec_labels):
            spec_slice = results.loc[results["specification"] == spec]
            if spec_slice.empty:
                continue
            r = spec_slice.iloc[0]
            row[label] = f"{float(r['r2']):.3f}" if key == "r2" else ", ".join(_fixed_effects_components_from_results(results, spec))
        rows.append(row)
        index.append(summary_label)
    table = pd.DataFrame(rows, index=index, columns=spec_labels)
    table.index.name = "Std.\\ effect"
    return table


def _write_single_spec_coefficient_table(
    results: pd.DataFrame,
    spec_name: str,
    term_order: tuple[str, ...],
    path: str | Path,
) -> None:
    """Write a one-equation coefficient table (Coef + SE + standardized effect)."""
    spec_rows = results.loc[results["specification"] == spec_name]
    by_term = {row["term"]: row for _, row in spec_rows.iterrows()}
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r" & Coef & Eff \\",
        r"Term &  &  \\",
        r"\midrule",
    ]
    for term in term_order:
        if term not in by_term:
            continue
        row = by_term[term]
        coef = f"{_format_signed_decimal(float(row['coefficient']), decimals=3)}{_significance_stars(float(row['t_stat']))}"
        econ = "" if pd.isna(row["economic_significance"]) else _format_signed_percent(float(row["economic_significance"]))
        lines.append(f"{TERM_DISPLAY_NAMES.get(term, term)} & {coef} & {econ} \\\\")
        lines.append(f" & ({float(row['std_error']):.3f}) &  \\\\")
    if not spec_rows.empty:
        first = spec_rows.iloc[0]
        fe = ", ".join(_fixed_effects_components_from_results(results, spec_name))
        lines.append(rf"Total $R^2$ & {float(first['r2']):.3f} &  \\")
        lines.append(rf"Observations & {int(first['nobs']):,} &  \\".replace(",", "{,}"))
        lines.append(rf"Fixed effects & {fe} &  \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n")
    logger.debug("_write_single_spec_coefficient_table wrote %s rows=%d", path, len(spec_rows))


def _significance_stars(t_stat: float) -> str:
    abs_t = abs(float(t_stat))
    if abs_t >= 2.58:
        return "***"
    if abs_t >= 1.96:
        return "**"
    if abs_t >= 1.645:
        return "*"
    return ""


def _format_signed_decimal(value: float, *, decimals: int = 3) -> str:
    formatted = f"{float(value):.{decimals}f}"
    return formatted if value < 0 else rf"\phantom{{-}}{formatted}"


def _format_signed_percent(value: float) -> str:
    # Round first and hide tiny magnitudes that collapse to zero.
    rounded = int(np.rint(float(value) * 100.0))
    if rounded == 0:
        return ""
    # Right-justify percentage text in a fixed-width box for visual alignment.
    return rf"\makebox[4em][r]{{{rounded:d}\%}}"


def _format_bracket_bold_percent(value: float) -> str:
    """Economic significance as a bold, bracketed percentage for the third row
    of each coefficient cell (e.g. ``[\\textbf{-30\\%}]``). Empty when it rounds
    to zero, matching the old Eff column's suppression of tiny effects."""
    if pd.isna(value):
        return ""
    rounded = int(np.rint(float(value) * 100.0))
    if rounded == 0:
        return ""
    return rf"[\textbf{{{rounded:d}\%}}]"


def _format_pct_effect(value: float, t_stat: float) -> str:
    """Bold signed percentage effect with significance stars, for the compact
    effects-only table (e.g. ``\\textbf{-30\\%}***``)."""
    if pd.isna(value):
        return ""
    rounded = int(np.rint(float(value) * 100.0))
    if rounded == 0:
        return "0\\%"
    return rf"\textbf{{{rounded:d}\%}}{_significance_stars(float(t_stat))}"


def _fixed_effects_components_for_spec(spec: RegressionSpec) -> list[str]:
    if not ENABLE_FIXED_EFFECTS:
        return ["None"]
    if not spec.fixed_effects:
        return ["None"]
    labels = {
        "cell_id": [r"\texttt{cell\_id}"],
        "day": [r"\texttt{day}"],
    }
    components: list[str] = []
    for effect in spec.fixed_effects:
        components.extend(labels.get(effect, [effect]))
    return components if components else ["None"]


def _fixed_effects_components_from_results(results: pd.DataFrame, spec_name: str) -> list[str]:
    if "fixed_effects" in results.columns:
        value = results.loc[results["specification"] == spec_name, "fixed_effects"].dropna()
        if not value.empty:
            raw = str(value.iloc[0]).strip()
            if not raw:
                return ["None"]
            return [part for part in raw.split("|") if part] or ["None"]
    return ["None"]


def _cluster_meat(x: np.ndarray, residuals: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Cluster ``MEAT = sum_g (X_g' u_g)(X_g' u_g)'`` for integer cluster ``codes``.

    Implemented as ``S' S`` where ``S[g] = sum_{i in g} x_i u_i`` -- the per-cluster
    score, accumulated with ``np.add.at``. This is the only change from HC1, whose
    meat sums ``x_i x_i' u_i^2`` over individual rows.
    """
    n_groups = int(codes.max()) + 1
    scores = np.zeros((n_groups, x.shape[1]))
    np.add.at(scores, codes, x * residuals[:, None])
    return scores.T @ scores, n_groups


def _robust_vcov(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    frame: pd.DataFrame,
    n_params: int,
) -> np.ndarray:
    """Sandwich covariance: HC1 if ``SE_CLUSTERS`` is empty, else one-way or
    two-way (Cameron--Gelbach--Miller) cluster-robust using the cluster columns
    carried in ``frame``."""
    n = x.shape[0]
    cluster_keys = [k for k in SE_CLUSTERS if k in frame.columns]

    if not cluster_keys:
        scale = n / max(n - n_params, 1)
        meat = x.T @ (x * (residuals[:, None] ** 2))
        return scale * (xtx_inv @ meat @ xtx_inv)

    def one_way(codes: np.ndarray) -> np.ndarray:
        meat, n_groups = _cluster_meat(x, residuals, codes)
        adj = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / max(n - n_params, 1))
        return adj * (xtx_inv @ meat @ xtx_inv)

    if len(cluster_keys) == 1:
        codes = pd.factorize(frame[cluster_keys[0]])[0]
        return one_way(codes)

    # Two-way: V = V_a + V_b - V_{a intersect b}. The intersection of (cell, day)
    # is the individual cell-day, so its component is the HC-style per-row meat.
    a = pd.factorize(frame[cluster_keys[0]])[0]
    b = pd.factorize(frame[cluster_keys[1]])[0]
    ab = pd.factorize(pd.Series(list(zip(a.tolist(), b.tolist()))))[0]
    return one_way(a) + one_way(b) - one_way(ab)


@debug_runtime("_fit_ols_robust")
def _fit_ols_robust(frame: pd.DataFrame, spec: RegressionSpec) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    design = _design_matrix(frame, spec)
    y = frame[spec.dependent].to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    logger.debug(
        "_fit_ols_robust spec=%s y_shape=%s x_shape=%s n_fixed_effects=%d se=%s",
        spec.name,
        y.shape,
        x.shape,
        len(design.columns) - 1 - len(spec.regressors),
        _se_description(),
    )
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    vcov = _robust_vcov(x, residuals, xtx_inv, frame, n_params=x.shape[1])
    stderr = np.sqrt(np.clip(np.diag(vcov), a_min=0.0, a_max=None))
    coefs = pd.Series(beta, index=design.columns)
    se = pd.Series(stderr, index=design.columns)
    t_stat = coefs.divide(se.replace(0.0, np.nan))
    r2 = 1.0 - residuals.var() / np.var(y)
    logger.debug(
        "_fit_ols_robust spec=%s design_cols=%d residual_var=%.6f r2=%.6f",
        spec.name,
        len(design.columns),
        float(residuals.var()),
        float(r2),
    )
    return coefs, se, t_stat, float(r2)


@debug_runtime("_design_matrix")
def _design_matrix(frame: pd.DataFrame, spec: RegressionSpec) -> pd.DataFrame:
    parts = [pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)]
    parts.append(frame.loc[:, spec.regressors].astype(float))
    if ENABLE_FIXED_EFFECTS:
        for fixed_effect in spec.fixed_effects:
            dummies = pd.get_dummies(frame[fixed_effect], prefix=fixed_effect, drop_first=True, dtype=float)
            logger.debug(
                "_design_matrix spec=%s fixed_effect=%s levels=%d dummy_cols=%d",
                spec.name,
                fixed_effect,
                frame[fixed_effect].nunique(dropna=True),
                dummies.shape[1],
            )
            parts.append(dummies)
    elif spec.fixed_effects:
        logger.debug(
            "_design_matrix spec=%s skipping fixed effects for debug speed fixed_effects=%s",
            spec.name,
            spec.fixed_effects,
        )
    design = pd.concat(parts, axis=1)
    logger.debug("_design_matrix spec=%s final_shape=%s", spec.name, design.shape)
    return design