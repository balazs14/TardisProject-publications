# AMU Liquidity Premium Reproduction Notes

## Start here (initial data alignment)

```bash
cd TardisProject

./generic_loop.sh -N 8 -S 2020-01-01 -E 2026-06-05 ./tardisctl align-put-call-quotes-trades-chain --exchange okex --loglevel INFO --date DATE

./generic_loop.sh -N 8 -S 2020-01-01 -E 2026-06-05 ./tardisctl align-put-call-quotes-trades-chain --exchange deribit --loglevel INFO --date DATE
```

Replace `DATE` with the calendar date being processed by `generic_loop.sh`.

## Build the panel regressions and figures

From the repository root, the liquidity-premium analysis layer now has a single rebuild entrypoint:

```bash
cd TardisProject
bash publications/AMU-liquidity-premium/rebuild_liquidity_premium_publication.sh
```

This script does four things:

1. rebuilds the AMU baseline artifacts directly from the daily aligned dataset parquet files,
2. builds the grouped daily AMU panel features used by the paper,
3. writes the regression tables into `publications/AMU-liquidity-premium/regressions/`,
4. writes the figure files into `publications/AMU-liquidity-premium/liquidity_figures/`.

Generated AMU baseline artifacts in `publications/AMU-liquidity-premium/`:

- `summary_daily_option_coverage_table.tex`
- `amu_summary_table.tex`
- `okex_btcusd_4_spreads.pdf`
- `okex_ethusd_4_spreads.pdf`
- `deribit_btcusd_4_spreads.pdf`
- `deribit_ethusd_4_spreads.pdf`
- `multi_exchange_amu_bps_by_date.pdf`
- `multi_exchange_amu_bps_by_tte.pdf`
- `multi_exchange_amu_bps_by_rel_strike.pdf`

Generated figure files:

- `liquidity_figures/liquidity_daily_amu.png`
- `liquidity_figures/liquidity_pre_post_heatmap.png`
- `liquidity_figures/liquidity_event_study.png`
- `liquidity_figures/liquidity_friction_gradient.png`
- `liquidity_figures/liquidity_compression_decomposition.png`

Generated regression files:

- `regressions/baseline_post2024.csv`
- `regressions/forward_component.csv`
- `regressions/backward_component.csv`
- `regressions/interaction_segments.csv`

## Build the PDF

The same script also runs LaTeX with intermediates in `build/`.

```bash
cd TardisProject
BUILD_PDF=1 bash publications/AMU-liquidity-premium/rebuild_liquidity_premium_publication.sh
```

For a data-only rebuild without LaTeX:

```bash
cd TardisProject
BUILD_PDF=0 bash publications/AMU-liquidity-premium/rebuild_liquidity_premium_publication.sh
```

To restrict the sample window:

```bash
cd TardisProject
FROM_DATE=2024-01-01 TO_DATE=2024-12-31 BUILD_PDF=0 bash publications/AMU-liquidity-premium/rebuild_liquidity_premium_publication.sh
```

## Notes

- `amu_statistics.py` now reads `datasets/{exchange}/{exchange}_aligned_put_call_quotes_trades_chain_*_5min.parquet` directly and does not depend on `publications/AMU/` or a `compacted.parquet` intermediate.
- The regression code lives in `publications/AMU-liquidity-premium/panel_regressions.py`.
- The figure code lives in `publications/AMU-liquidity-premium/panel_figures.py`.
- The manuscript includes the generated figures from the local `liquidity_figures/` directory.
