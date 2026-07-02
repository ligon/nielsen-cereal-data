#!/bin/bash
#SBATCH --job-name=nl-rebuild-households
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=log/rebuild-households-%j.out
#SBATCH --error=log/rebuild-households-%j.err

# Rebuild the household panel through aggregate_households (Stage 6) so it picks
# up the sodium mg->g normalization (commit 198cb2d, in the shared
# _merge_nutrition).  household_panel carries BOTH sodium AND a DERIVED Z_sodium,
# so an in-place column patch is not enough -- the aggregation must recompute the
# Z aggregates from gram-sodium.  Full range (config default 2004-2024).
#
# Direct invocation, NOT `make households` (whose `extract crosswalk` prereqs
# would needlessly re-run the heavy tarball extraction).  CPU burst on
# non-preemptible fc_jevons so the ~4M-transaction aggregation runs to completion.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p log
export PYTHONUNBUFFERED=1
VENV=.venv/bin/python

echo "=== node $(hostname); start $(date) ==="
echo "sodium BEFORE (current household_panel, expect ~150 mg):"
"$VENV" -c "import pandas as pd; d=pd.read_parquet('data/clean/household_panel.parquet'); print('  sodium mean', round(float(d['sodium'].mean()),4))" || true

echo
echo "=== aggregate_households (full 2004-2024) ==="
RC=0
"$VENV" src/aggregate_households.py --config config/default.yaml || RC=$?

echo
echo "=== sodium AFTER (expect ~0.15 g; Z_sodium likewise gram-scale) ==="
"$VENV" -c "
import pandas as pd
d = pd.read_parquet('data/clean/household_panel.parquet')
print('  sodium   mean', round(float(d['sodium'].mean()),5) if 'sodium' in d.columns else 'n/a')
print('  Z_sodium mean', round(float(d['Z_sodium'].mean()),5) if 'Z_sodium' in d.columns else 'n/a')
print('  rows', len(d))
" || true
echo "=== rc=$RC; end $(date) ==="
exit "$RC"
