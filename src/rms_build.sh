#!/usr/bin/env bash
# One command for the RMS second source (charter-rms D2 + D3):
#
#   src/rms_build.sh [d2|d3|all]        (default: all; `make rms` runs this)
#
# d2  1. stream module-1344 movement + Annual_Files out of the classic
#        2006-2020 monoliths (src/rms_extract_classic.sh; idempotent, skips
#        years whose <yr>.done marker exists)
#     2. classic product master (module 1344, all versions) -> parquet
#     3. every year 2006-2024 -> per-year parquet + stats.json
#        (src/rms_convert.py; 2021+ is streamed from the .tgz, never unpacked)
#     4. per-year log -> docs/rms-d2-log.org (committed)
# d3  5. market_panel_rms.parquet (src/rms_market_panel.py)
#
# Outputs: $OUT (default Data/NielsenScanner/rms), group-only (umask 027).
# Idempotent: a year with stats.json is skipped; FORCE=1 redoes steps 2-3.
#   PYTHON (env)  default: the CerealDemand venv (pandas 3.0.3), so the
#                 D3 parquet's str dtypes match the venv that tests it.
#   JOBS   (env)  parallel years in step 3, default 8.
set -euo pipefail
umask 027
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python}
DATA=${NIELSEN_DATA:-/global/scratch/fsa/fc_jevons/ligon/Data}
OUT=${RMS_OUT:-$DATA/NielsenScanner/rms}
JOBS=${JOBS:-8}
YEARS=${YEARS:-$(seq -s ' ' 2006 2024)}
stage=${1:-all}
force=()
[[ ${FORCE:-0} == 1 ]] && force=(--force)
export RMS_OUT=$OUT NIELSEN_DATA=$DATA
mkdir -p "$OUT/logs"

if [[ $stage == d2 || $stage == all ]]; then
    echo "[1/5] classic extraction (idempotent)  $(date -Is)"
    classic=()
    for y in $YEARS; do (( y <= 2020 )) && classic+=("$y"); done
    if (( ${#classic[@]} )); then src/rms_extract_classic.sh "${classic[@]}"; fi

    echo "[2/5] classic product master  $(date -Is)"
    "$PYTHON" src/rms_convert.py master "${force[@]}"

    echo "[3/5] per-year parquet (JOBS=$JOBS)  $(date -Is)"
    export PYTHON
    printf '%s\n' $YEARS | xargs -P "$JOBS" -I{} bash -c \
        'umask 027; "$PYTHON" src/rms_convert.py year {} '"${force[*]}"' \
             > "$RMS_OUT/logs/convert_{}.log" 2>&1 \
         && tail -n1 "$RMS_OUT/logs/convert_{}.log" \
         || { echo "{}: FAILED (see $RMS_OUT/logs/convert_{}.log)"; exit 255; }'

    echo "[4/5] per-year log  $(date -Is)"
    "$PYTHON" src/rms_convert.py log --out docs/rms-d2-log.org
fi

if [[ $stage == d3 || $stage == all ]]; then
    echo "[5/5] market_panel_rms.parquet  $(date -Is)"
    "$PYTHON" src/rms_market_panel.py
fi

chmod -R g+rX,o-rwx "$OUT"
echo "done  $(date -Is)"
