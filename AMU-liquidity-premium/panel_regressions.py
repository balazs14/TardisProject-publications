from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from amu_panel import build_amu_panel


POST_2024_START = date(2024, 1, 10)
OTM_DISTANCE = 0.05
SHORT_TTE_CUTOFF = 0.35


@dataclass(frozen=True)
class RegressionSpec:
	name: str
	dependent: str
	regressors: tuple[str, ...]
	fixed_effects: tuple[str, ...]


def build_liquidity_analysis_panel(
	panel: pl.DataFrame | None = None,
	**build_kwargs,
) -> pd.DataFrame:
	if panel is None:
		panel = build_amu_panel(**_build_kwargs_with_defaults(build_kwargs))

	frame = panel.to_pandas().copy()
	frame["day"] = pd.to_datetime(frame["day"])
	frame["mean_amu_bp"] = 0.5 * (frame["mean_amu_fwd_bp"] + frame["mean_amu_bck_bp"])
	frame["post_2024"] = (frame["day"].dt.date >= POST_2024_START).astype(float)
	frame["spread_bp"] = frame["mean_bigger_opt_spread_bp"]
	frame["depth_proxy"] = np.log(frame["mean_min_quote_size_dollar"].clip(lower=1.0))
	frame["stale_proxy"] = frame[["frac_call_stale", "frac_put_stale", "frac_spot_stale"]].mean(axis=1)
	frame["eth"] = frame["ref_sym"].str.contains("ETH", na=False).astype(float)
	frame["otm"] = (frame["rel_strike_bucket"].sub(1.0).abs() > OTM_DISTANCE).astype(float)
	frame["short_tte"] = (frame["tte_bucket"] <= SHORT_TTE_CUTOFF).astype(float)
	frame["post_2024_x_eth"] = frame["post_2024"] * frame["eth"]
	frame["post_2024_x_otm"] = frame["post_2024"] * frame["otm"]
	frame["post_2024_x_short_tte"] = frame["post_2024"] * frame["short_tte"]
	frame["cell_id"] = (
		frame["exchange"]
		+ "|"
		+ frame["ref_sym"]
		+ "|"
		+ frame["rel_strike_bucket"].round(6).astype(str)
		+ "|"
		+ frame["tte_bucket"].round(6).astype(str)
	)
	return frame.sort_values(["day", "exchange", "ref_sym", "rel_strike_bucket", "tte_bucket"])


def baseline_regression_spec() -> RegressionSpec:
	# Post2024 varies only by day, so day fixed effects would absorb it.
	return RegressionSpec(
		name="baseline_post2024",
		dependent="mean_amu_bp",
		regressors=("post_2024", "spread_bp", "depth_proxy", "stale_proxy"),
		fixed_effects=("cell_id",),
	)


def forward_regression_spec() -> RegressionSpec:
	# The directional post-2024 term is likewise identified off within-cell time variation.
	return RegressionSpec(
		name="forward_component",
		dependent="mean_amu_fwd_bp",
		regressors=("post_2024", "spread_bp", "depth_proxy"),
		fixed_effects=("cell_id",),
	)


def backward_regression_spec() -> RegressionSpec:
	return RegressionSpec(
		name="backward_component",
		dependent="mean_amu_bck_bp",
		regressors=("post_2024", "spread_bp", "depth_proxy"),
		fixed_effects=("cell_id",),
	)


def interaction_regression_spec() -> RegressionSpec:
	# Cell fixed effects absorb ETH/OTM/ShortTTE main effects; day fixed effects absorb Post2024.
	return RegressionSpec(
		name="interaction_segments",
		dependent="mean_amu_bp",
		regressors=("post_2024_x_eth", "post_2024_x_otm", "post_2024_x_short_tte"),
		fixed_effects=("cell_id", "day"),
	)


def run_baseline_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
	return _run_regression(baseline_regression_spec(), panel=panel, **build_kwargs)


def test_run_baseline_regression() -> None:
	from tardis import test_utils as tu

	result = run_baseline_regression(panel=_test_window_panel())
	tu.assert_df_equal(
		result[["term", "coefficient", "std_error", "t_stat", "nobs", "r2"]].round(6),
		"""                     term  coefficient  std_error       t_stat  nobs        r2
0               post_2024     2.305008   1.719039     1.340870  9029  0.999874
1               spread_bp    -0.502342   0.000488 -1028.482787  9029  0.999874
2             depth_proxy    -3.480819   2.103244    -1.654977  9029  0.999874
3             stale_proxy  -282.389795  33.175495    -8.511999  9029  0.999874
""",
	)


def run_forward_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
	return _run_regression(forward_regression_spec(), panel=panel, **build_kwargs)


def test_run_forward_regression() -> None:
	from tardis import test_utils as tu

	result = run_forward_regression(panel=_test_window_panel())
	tu.assert_df_equal(
		result[["term", "coefficient", "std_error", "t_stat", "nobs", "r2"]].round(6),
		"""                     term  coefficient  std_error       t_stat  nobs        r2
0               post_2024   232.664140  18.775876    12.391653  9029  0.998417
1               spread_bp    -0.990967   0.000393 -2523.827214  9029  0.998417
2             depth_proxy  -275.835519  24.297960   -11.352209  9029  0.998417
""",
	)


def run_backward_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
	return _run_regression(backward_regression_spec(), panel=panel, **build_kwargs)


def test_run_backward_regression() -> None:
	from tardis import test_utils as tu

	result = run_backward_regression(panel=_test_window_panel())
	tu.assert_df_equal(
		result[["term", "coefficient", "std_error", "t_stat", "nobs", "r2"]].round(6),
		"""                     term  coefficient  std_error     t_stat  nobs        r2
0               post_2024  -233.865529  19.324254 -12.102176  9029  0.456401
1               spread_bp    -0.014282   0.001043 -13.699674  9029  0.456401
2             depth_proxy   283.537968  24.621937  11.515665  9029  0.456401
""",
	)


def run_interaction_regression(panel: pd.DataFrame | pl.DataFrame | None = None, **build_kwargs) -> pd.DataFrame:
	return _run_regression(interaction_regression_spec(), panel=panel, **build_kwargs)


def test_run_interaction_regression() -> None:
	from tardis import test_utils as tu

	result = run_interaction_regression(panel=_test_window_panel())
	tu.assert_df_equal(
		result[["term", "coefficient", "std_error", "t_stat", "nobs", "r2"]].round(6),
		"""                      term  coefficient   std_error     t_stat  nobs        r2
0        post_2024_x_eth -1860.417468  338.424581  -5.497288  9029  0.473381
1        post_2024_x_otm -3173.503009  252.483007 -12.569175  9029  0.473381
2  post_2024_x_short_tte   714.262711  285.429938   2.502410  9029  0.473381
""",
	)


def write_regression_tables(output_dir: str | Path, **build_kwargs) -> dict[str, Path]:
	panel = build_liquidity_analysis_panel(**build_kwargs)
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	paths: dict[str, Path] = {}
	for runner in (
		run_baseline_regression,
		run_forward_regression,
		run_backward_regression,
		run_interaction_regression,
	):
		result = runner(panel=panel)
		path = output_root / f"{result['specification'].iloc[0]}.csv"
		result.to_csv(path, index=False)
		paths[result["specification"].iloc[0]] = path
	return paths


def _run_regression(
	spec: RegressionSpec,
	panel: pd.DataFrame | pl.DataFrame | None = None,
	**build_kwargs,
) -> pd.DataFrame:
	frame = _coerce_analysis_panel(panel, **build_kwargs)
	regression_frame = _regression_frame(frame, spec)
	coefficients, stderr, t_stat, r2 = _fit_ols_with_hc1(regression_frame, spec)
	rows = []
	for term in spec.regressors:
		rows.append(
			{
				"specification": spec.name,
				"dependent": spec.dependent,
				"term": term,
				"coefficient": coefficients[term],
				"std_error": stderr[term],
				"t_stat": t_stat[term],
				"nobs": len(regression_frame),
				"r2": r2,
			}
		)
	return pd.DataFrame(rows)


def _coerce_analysis_panel(panel: pd.DataFrame | pl.DataFrame | None, **build_kwargs) -> pd.DataFrame:
	if panel is None:
		return build_liquidity_analysis_panel(**build_kwargs)
	if isinstance(panel, pl.DataFrame):
		return build_liquidity_analysis_panel(panel=panel)
	return panel.copy()


def _build_kwargs_with_defaults(build_kwargs: dict[str, object]) -> dict[str, object]:
	kwargs = dict(build_kwargs)
	kwargs.setdefault("raw_data_dir", str(REPO_ROOT / "datasets/{exchange}/"))
	return kwargs


def _test_window_panel() -> pd.DataFrame:
	return build_liquidity_analysis_panel(from_date="2024-01-01", to_date="2024-01-20")


def _regression_frame(frame: pd.DataFrame, spec: RegressionSpec) -> pd.DataFrame:
	columns = [spec.dependent, *spec.regressors, *spec.fixed_effects]
	return frame.loc[:, columns].dropna().reset_index(drop=True)


def _fit_ols_with_hc1(frame: pd.DataFrame, spec: RegressionSpec) -> tuple[pd.Series, pd.Series, pd.Series, float]:
	design = _design_matrix(frame, spec)
	y = frame[spec.dependent].to_numpy(dtype=float)
	x = design.to_numpy(dtype=float)
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
	return coefs, se, t_stat, float(r2)


def _design_matrix(frame: pd.DataFrame, spec: RegressionSpec) -> pd.DataFrame:
	parts = [pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)]
	parts.append(frame.loc[:, spec.regressors].astype(float))
	for fixed_effect in spec.fixed_effects:
		dummies = pd.get_dummies(frame[fixed_effect], prefix=fixed_effect, drop_first=True, dtype=float)
		parts.append(dummies)
	return pd.concat(parts, axis=1)