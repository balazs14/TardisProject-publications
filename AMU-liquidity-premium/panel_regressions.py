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
}

# Column order in the combined coefficient table: (1) Post, (2) Post+cell FE,
# (3) Post+segment dummies (pooled OLS), (4) Post+frictions+cell FE.
MAIN_SPEC_ORDER = ("amu_spec1", "amu_spec1fe", "amu_spec2", "amu_spec3")

TERM_DISPLAY_NAMES = {
    "post_2024": "$\\mathrm{Post}_t$",
    "average_put_call_spread_bp": "$\\mathrm{Spr}_{g,t}$",
    "log_mean_min_quote_size_dollar": "$\\mathrm{Depth}_{g,t}$",
    "stale_proxy": "$\\mathrm{Stale}_{g,t}$",
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
    frame["mean_mma_bp"] = 0.5 * (frame["mean_mma_fwd_bp"] + frame["mean_mma_bck_bp"])
    frame["std_mma_bp"] = frame[["mean_mma_fwd_bp", "mean_mma_bck_bp"]].std(axis=1, ddof=0)
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


# ARP dependent variable for every spec below. The panel carries both explicit
# flavours; the regressions use the UNCONDITIONAL (per-quote = frequency x
# conditional) ARP. Switch to "mean_amu_conditional_bp" for the conditional size.
AMU_DEPENDENT = "mean_amu_uncond_bp_c20"

# Nested ARP specifications (HC1 SEs):
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
        "AmuMeanBp": f"{wmean(amu.notna()):.1f}",
        "AmuPreBp": f"{wmean(post == 0.0):.1f}",
        "AmuPostBp": f"{wmean(post == 1.0):.1f}",
    }
    lines = ["% Auto-generated by write_text_macros; do not edit by hand."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    path = Path(output_dir) / "text_numbers.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.debug("write_text_macros wrote %s macros=%s", path, macros)
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
    dataframe_to_tabular_tex(summary_table, summary_path)
    dataframe_to_tabular_tex(coefficient_table, coefficient_path)
    paths["regression_model_summary_table"] = summary_path
    paths["regression_coefficients_table"] = coefficient_path
    paths["text_numbers"] = write_text_macros(filter_analysis_panel(panel), output_root)
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
    coefficients, stderr, t_stat, r2 = _fit_ols_with_hc1(regression_frame, effective_spec)
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
    columns = [spec.dependent, *spec.regressors, *spec.fixed_effects]
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
        "mean_amu_conditional_bp": "ARP (cond)",
        "mean_amu_unconditional_bp": "ARP (uncond)",
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

    columns = pd.MultiIndex.from_tuples(
        [(SPEC_DISPLAY_NAMES.get(spec, spec), "Coef") for spec in spec_order]
        + [(SPEC_DISPLAY_NAMES.get(spec, spec), r"Eff") for spec in spec_order]
    )
    output_rows: list[dict[tuple[str, str], str]] = []
    output_index: list[str] = []

    for term in term_order:
        if term not in results["term"].values:
            continue
        coef_row = {column: "" for column in columns}
        se_row = {column: "" for column in columns}
        for spec in spec_order:
            spec_slice = results.loc[(results["specification"] == spec) & (results["term"] == term)]
            if spec_slice.empty:
                continue
            row = spec_slice.iloc[0]
            spec_label = SPEC_DISPLAY_NAMES.get(spec, spec)
            coef_row[(spec_label, "Coef")] = (
                f"{_format_signed_decimal(float(row['coefficient']), decimals=3)}{_significance_stars(float(row['t_stat']))}"
            )
            se_row[(spec_label, "Coef")] = f"({float(row['std_error']):.3f})"
            coef_row[(spec_label, r"Eff")] = (
                "" if pd.isna(row["economic_significance"]) else _format_signed_percent(float(row["economic_significance"]))
            )
        output_rows.append(coef_row)
        output_index.append(TERM_DISPLAY_NAMES.get(term, term))
        output_rows.append(se_row)
        output_index.append("")

    for summary_label, key in (("Total $R^2$", "r2"), ("Observations", "nobs"), ("Fixed effects", "fixed_effects")):
        summary_row = {column: "" for column in columns}
        for spec in spec_order:
            spec_slice = results.loc[results["specification"] == spec]
            if spec_slice.empty:
                continue
            row = spec_slice.iloc[0]
            spec_label = SPEC_DISPLAY_NAMES.get(spec, spec)
            if key == "r2":
                value = f"{float(row['r2']):.3f}"
            elif key == "nobs":
                value = f"{int(row['nobs']):,}"
            else:
                fixed_effects_components = _fixed_effects_components_from_results(results, spec)
                value = ", ".join(fixed_effects_components)
            summary_row[(spec_label, "Coef")] = value
        output_rows.append(summary_row)
        output_index.append(summary_label)

    table = pd.DataFrame(output_rows, index=output_index, columns=columns)
    table.index.name = "Term"
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


@debug_runtime("_fit_ols_with_hc1")
def _fit_ols_with_hc1(frame: pd.DataFrame, spec: RegressionSpec) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    design = _design_matrix(frame, spec)
    y = frame[spec.dependent].to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    logger.debug(
        "_fit_ols_with_hc1 spec=%s y_shape=%s x_shape=%s n_fixed_effects=%d",
        spec.name,
        y.shape,
        x.shape,
        len(design.columns) - 1 - len(spec.regressors),
    )
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    hc1_scale = len(frame) / max(len(frame) - x.shape[1], 1)
    meat = x.T @ (x * (residuals[:, None] ** 2))
    vcov = hc1_scale * (xtx_inv @ meat @ xtx_inv)
    stderr = np.sqrt(np.clip(np.diag(vcov), a_min=0.0, a_max=None))
    coefs = pd.Series(beta, index=design.columns)
    se = pd.Series(stderr, index=design.columns)
    t_stat = coefs.divide(se.replace(0.0, np.nan))
    r2 = 1.0 - residuals.var() / np.var(y)
    logger.debug(
        "_fit_ols_with_hc1 spec=%s design_cols=%d residual_var=%.6f r2=%.6f",
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