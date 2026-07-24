#!/usr/bin/env bash
# r-sensitivity sweep: recompute AMU across discount rates and collect the
# pre/post drop into artifacts/r_sensitivity_table.tex.
#
# HEAVY -- each r triggers a full ~18 GB tick-level rebuild (r enters e^{-rT}
# before aggregation, so unlike `cost` it cannot be swept from the panel). Run
# OUTSIDE the sandbox, on the machine that holds the datasets.
#
# Usage:   ./run_r_sensitivity.sh [r1 r2 ...]
# Example: ./run_r_sensitivity.sh 0.00 0.025 0.05 0.075 0.10
# Env:     FROM_DATE, TO_DATE, VENV_PYTHON may be overridden.
set -euo pipefail
cd "$(dirname "$0")"

FROM_DATE="${FROM_DATE:-2020-01-01}"
TO_DATE="${TO_DATE:-2026-06-05}"
VENV_PYTHON="${VENV_PYTHON:-python3}"

RVALUES=("$@")
if [[ ${#RVALUES[@]} -eq 0 ]]; then
    RVALUES=(0.00 0.025 0.05 0.075 0.10)
fi

echo "r-sensitivity sweep over: ${RVALUES[*]}  (from=$FROM_DATE to=$TO_DATE)"
for r in "${RVALUES[@]}"; do
    adir="artifacts_r${r}"
    echo ">>> r=${r} -> ${adir}"
    PCP_R_OVERRIDE="$r" \
    ARTIFACTS_DIR="$adir" \
    FROM_DATE="$FROM_DATE" \
    TO_DATE="$TO_DATE" \
        "$VENV_PYTHON" run_r_sensitivity_one.py
done

echo ">>> collecting"
"$VENV_PYTHON" collect_r_sensitivity.py "${RVALUES[@]}"
echo "done. See artifacts/r_sensitivity_table.tex"
