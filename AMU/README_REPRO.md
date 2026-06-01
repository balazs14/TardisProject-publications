# AMU Rebuild Notes

## One-line rebuild

```bash
bash publications/AMU/rebuild_amu_publication.sh
```

## What this regenerates

- `publications/AMU/summary_daily_option_coverage.png`
- `publications/AMU/amu_summary.png`
- `publications/AMU/multi_exchange_amu_bps_by_date.png`
- `publications/AMU/multi_exchange_amu_bps_by_rel_strike.png`
- `publications/AMU/multi_exchange_amu_bps_by_tte.png`
- Executed notebook: `publications/AMU/multi_exchange_plots.ipynb`

## Useful toggles

- Rebuild compacted parquet from aligned files:

```bash
AMU_RECREATE_PCPB=1 bash publications/AMU/rebuild_amu_publication.sh
```

- Dry run (print plan only):

```bash
DRY_RUN=1 bash publications/AMU/rebuild_amu_publication.sh
```

- Also build PDF (`main.tex`):

```bash
BUILD_PDF=1 bash publications/AMU/rebuild_amu_publication.sh
```

This uses a task-like LaTeX flow with intermediates in `publications/AMU/build/`
and SyncTeX metadata at `publications/AMU/build/main.synctex.gz`.
The script also copies `publications/AMU/build/main.pdf` to
`publications/AMU/main.pdf` for convenience.

## Future-me checklist

1. Update hash log header inside `publications/AMU/rebuild_amu_publication.sh`.
2. Confirm datasets exist under `datasets/deribit` and `datasets/okex`.
3. Run one-line rebuild command.
4. If compacted data is missing and no seed exists, rerun with `AMU_RECREATE_PCPB=1`.
