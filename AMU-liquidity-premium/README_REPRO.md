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

This script does three things:

1. builds the grouped daily AMU panel features used by the paper,
2. writes the regression tables into `publications/AMU-liquidity-premium/regressions/`,
3. writes the figure files into `publications/AMU-liquidity-premium/liquidity_figures/`.

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

- The regression code lives in `publications/AMU-liquidity-premium/panel_regressions.py`.
- The figure code lives in `publications/AMU-liquidity-premium/panel_figures.py`.
- The manuscript includes the generated figures from the local `liquidity_figures/` directory.
