# AMU liquidity-premium — session handoff

Status note for resuming work after a restart. Everything below is already saved
to disk in this folder. The paper compiles (0 LaTeX errors): `liquidity-premium.tex`
(23 pp) and `figures-tables-overview.tex` (10 pp).

## Repo geography (quick reminder)
- Paper + pipeline: `publications/AMU-liquidity-premium/` (this folder).
- Arbitrage math source of truth: `tardis/process_pcp.py` (in `TardisProject/`, **not** here).
- `artifacts` is a **symlink → `artifacts_long_nofilterstale`** (the active build the paper `\input`s).
- `artifacts_long` is an older separate 18 GB build; nothing in the paper uses it (being shipped to external disk).
- Two tex files kept in sync for all floats/captions: `liquidity-premium.tex` (full) and `figures-tables-overview.tex` (floats only).

## Completed this session

### 1. `fwd`↔`bck` global rename (nominal only)
- Decision: `process_pcp.py` is authoritative (forward *sells* the call). Per user, this was a **pure label swap**, formulas unchanged.
- Swapped `fwd`↔`bck` and `forward`↔`backward` across both .tex files, all `publications/*.py` (variable names + column-string literals), and `amu_config.toml` + `amu_config.frozen.toml`.
- Protected from the swap: "carried forward" (tex/toml) and "Backward compatibility/-compatible" comments (py).
- `process_pcp.py` NOT touched → no cache schema change; column literals still resolve; AMU/MMA aggregate symmetrically so outputs are numerically identical.
- Line 153 buy/sell sentence: left as-is per user ("backward/join-call he buys at C_bid; forward/join-put he sells").
- Fig 2/Fig 3 captions use `cost'` (≡ cost − ½(F_ask−F_bid)e^{−rT}).

### 2. Intro citations filled
- 6 `TODO:cite` placeholders in the Introduction resolved.
- 5 new verified refs added to `bibl.bib`: `demsetz1968cost`, `ho1981optimal`, `gagnon2010multimarket`, `krishnamurthy2002bond`, `ofek2004limited` (plus reuse of existing keys).

### 3. PCA liquidity control — new regression spec (5)
- `panel_regressions.py`: `add_liquidity_pc()` builds `liquidity_pc` = PC1 of standardized (Depth, Spread, Stale), **fit separately pre/post 2024**, oriented so higher = less liquid, z-scored within regime. Spec (5) = `post_2024 + liquidity_pc + cell FE`; replaces the three collinear frictions of spec (4).
- New display: `SPEC_DISPLAY_NAMES["amu_specpca"]="(5)"`, `TERM_DISPLAY_NAMES["liquidity_pc"]`, added to `MAIN_SPEC_ORDER`, term_order, and the `write_regression_tables` runner loop. Loadings table via `write_liquidity_pca_table()`.
- Tex: spec (5) equation, `Liq` abbreviation, results paragraph, Table 6 loadings float, updated coefficient-table caption — in **both** tex files.
- **Full-data numbers already committed** to `artifacts/`: spec (5) Post = −6.6 bp (t≈−205), Liq = +8.41 (t≈+380), R²=0.558; loadings Depth −0.50/−0.39, Spread +0.57/+0.69, Stale +0.65/+0.61, var ≈ 53%/46% (pre/post).
- These were produced via a within-cell (Frisch–Waugh) transform verified to match the shipped HC1 `_fit_ols_with_hc1` to 1e-12, reusing the committed CSVs for specs (1)–(4). A normal `write_regression_tables` rerun reproduces the identical table.

### 4. r-sensitivity
- Analytic paragraph added near the cost-sensitivity discussion (§rq2): r enters only via e^{−rT}, which multiplies the small (F−K) / (F_ask−F_bid); negligible where AMU concentrates; pre/post difference more robust still.
- Guarded results table `tab:r_sensitivity` in both tex files: shows a placeholder until the sweep writes `artifacts/r_sensitivity_table.tex`, then auto-fills.
- Machinery for the **full recompute** (run outside the sandbox — heavy, rebuilds the ~18 GB tick frame per r):
  - `amu_config.py`: env override `PCP_R_OVERRIDE` injected into `CONFIG["pcp"]["r"]` (propagates to `PCP_R` and the `pcp_metric_kwargs` in amu_panel/amu_statistics).
  - `run_r_sensitivity.sh` — loop over r values → `run_r_sensitivity_one.py` (per-r pipeline into `artifacts_r<r>/`) → `collect_r_sensitivity.py` (writes `artifacts/r_sensitivity_table.tex`).
  - Usage: `./run_r_sensitivity.sh 0.00 0.025 0.05 0.075 0.10`  (user is running this now, artifacts to external disk).

## Queued / pending
- **HC1 std errors (task from earlier queue):** the code **already uses HC1** (`_fit_ols_with_hc1` in `panel_regressions.py`). Likely just needs confirmation it's what was wanted, or a caption/wording check — not a reimplementation.
- **After the r-sweep finishes:** run `python collect_r_sensitivity.py 0.00 0.025 0.05 0.075 0.10` (if not already called by the .sh), then recompile; verify `tab:r_sensitivity` fills and the flat-in-r story holds.
- **Optional cleanup:** rebuild default points at the shipped-off dir — line 23 of `rebuild_liquidity_premium_publication.sh`: `ARTIFACTS_DIR=${ARTIFACTS_DIR:-artifacts_long}`. Change to `artifacts_long_nofilterstale` so a bare rebuild matches current outputs (user asked to consider this; not yet done).
- **Disposable:** `tmp/gen_provisional_regtable.py`, `review_response.md` are old scratch files.

## Sandbox quirks (for a fresh session)
- Python < 3.11: no `tomllib`. Shim: `mkdir -p /tmp/shim && printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py` and set `PYTHONPATH="<pubdir>:<TardisProject>:/tmp/shim"`.
- `agsm.bst` is not in the sandbox texlive → citations render as `(????)` in sandbox builds only; they resolve on your machine. Compile-error checks (`grep -c '^!'`) and cross-ref checks are still valid.
- ~3.9 GB RAM, 45 s bash cap: the full 764-cell dense FE regression (~3 GB design) can OOM here — that's why spec (5) was computed via the within-transform. Full regeneration is fine on your machine.
- To regenerate tables to match current committed output, run the pipeline with `ARTIFACTS_DIR=artifacts_long_nofilterstale` (the default in the script is the *other* dir).

## To reproduce the current paper PDF
Run your normal rebuild with `ARTIFACTS_DIR=artifacts_long_nofilterstale` (so bibtex + `agsm.bst` resolve citations), then `latexmk` both tex files.
