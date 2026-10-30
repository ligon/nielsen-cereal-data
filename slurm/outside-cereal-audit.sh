#!/usr/bin/env bash
# Read-only scan of the adopted RMS vintage in a pinned node-local clone.
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${AUDIT_PIN:?pin the Nielsen audit commit}"
: "${AUDIT_ROOT:?choose a separate durable output root}"
MIRROR=/global/home/users/ligon/mirrors/nielsen-cereal-data
PY=/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python
WORK=/local/job${SLURM_JOB_ID}/mirrors/nielsen-cereal-data
mkdir -p "$(dirname "$WORK")" "/local/job${SLURM_JOB_ID}/tmp"
git clone --quiet --no-checkout "$MIRROR" "$WORK"
git -C "$WORK" checkout --quiet --detach "$AUDIT_PIN"
[[ $(git -C "$WORK" rev-parse HEAD) == "$AUDIT_PIN" ]]
[[ $(nproc) == "$SLURM_CPUS_PER_TASK" ]]
export TMPDIR=/local/job${SLURM_JOB_ID}/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PANEL_RAW=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/raw
export PANEL_CLEAN=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-hhfix-2026-10-23
export RMS_OUT=/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/crosswalk-hhfix-2026-10-23
cd "$WORK"
"$PY" -m pytest -q tests/test_rms_outside_audit.py tests/test_rms_market_panel.py tests/test_brand_route.py
"$PY" src/rms_outside_audit.py --rms-dir "$RMS_OUT" --panel-raw "$PANEL_RAW" \
    --output "$AUDIT_ROOT/run-$SLURM_JOB_ID" --jobs "$SLURM_CPUS_PER_TASK" \
    --batch-size 500000
