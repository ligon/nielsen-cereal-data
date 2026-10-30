#!/usr/bin/env bash
# Pinned read-only movement scan; reuse adopted geography/week/category rules.
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${SIZE_PIN:?pin full Nielsen commit}"
: "${FOLLOWUP_ROOT:?choose a fresh durable evidence root}"
PY=/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python
WORK=/local/job${SLURM_JOB_ID}/mirrors/nielsen-cereal-data
mkdir -p "$(dirname "$WORK")" "/local/job${SLURM_JOB_ID}/tmp"
git clone --quiet --no-checkout /global/home/users/ligon/mirrors/nielsen-cereal-data "$WORK"
git -C "$WORK" checkout --quiet --detach "$SIZE_PIN"
[[ $(git -C "$WORK" rev-parse HEAD) == "$SIZE_PIN" ]]
[[ $(nproc) == "$SLURM_CPUS_PER_TASK" ]]
export TMPDIR=/local/job${SLURM_JOB_ID}/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export RMS_OUT=/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/crosswalk-hhfix-2026-10-23
export PANEL_CLEAN=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-hhfix-2026-10-23
cd "$WORK"
"$PY" -m pytest -q tests/test_rms_size_consensus.py tests/test_rms_size_candidates.py \
    tests/test_rms_outside_audit.py tests/test_rms_market_panel.py tests/test_brand_route.py
"$PY" src/rms_size_market_audit.py \
    --rms-dir /global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/crosswalk-hhfix-2026-10-23 \
    --audit-dir /global/scratch/fsa/fc_jevons/ligon/Data/CerealDemand/outside-cereal-audit-2026-10-29/run-40609757 \
    --candidate-map /global/scratch/fsa/fc_jevons/ligon/Data/CerealDemand/outside-size-recovery-2026-10-29-v2/metadata/size_candidates.parquet \
    --output "$FOLLOWUP_ROOT/market-scan/run-$SLURM_JOB_ID" --jobs "$SLURM_CPUS_PER_TASK"
