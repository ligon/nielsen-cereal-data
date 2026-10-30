#!/bin/bash
#SBATCH --job-name=xw-rms-d3
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.out
#SBATCH --error=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.err

# charter-crosswalk A7, job 3 of 5 (docs/crosswalk-slurm-plan.org): the RMS
# panel on the new route, read against the new Homescan panel.
#
# NOT src/rms_build.sh d3: it runs rms_market_panel.py with no --outdir/--log
# and would overwrite the accepted market_panel_rms.parquet and the committed
# docs/rms-d3-log.org (ledger G3).  Instead:
#   PANEL_CLEAN = the vintage (RMP reads its 45 and market_size from
#                 market_panel.parquet there: rms_market_panel.py:124-131)
#   PANEL_RAW   = Homescan raw (2021-24 names for the shared brand route)
#   RMS_OUT     = left at the accepted dir (movement, stores, master: read-only)
#   --outdir    = $RMS_VINTAGE (panel + diagnostics/), --log in the vintage.
# The accepted D3 build (2026-10-19 10:22-10:26, this recipe's JOBS=8) took
# under 4 minutes; 90G / 1h is generous.
#
#   sbatch --dependency=afterok:<markets job> slurm/crosswalk-rms-d3.slurm

set -euo pipefail
# Bootstrap from a verified local clone before sourcing any repository code.
BOOT_MIRROR=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data
BOOT_REPO=/local/job${SLURM_JOB_ID:?}/mirrors/nielsen-cereal-data
XW_BRANCH_NIELSEN=${XW_BRANCH_NIELSEN:-crosswalk}
[[ ${XW_PIN_NIELSEN:-} =~ ^[0-9a-f]{40}$ ]] || { echo "XW FATAL: full Nielsen pin required" >&2; exit 2; }
[[ $(git -C "$BOOT_MIRROR" rev-parse refs/heads/$XW_BRANCH_NIELSEN) == "$XW_PIN_NIELSEN" ]] \
    || { echo "XW FATAL: durable Nielsen branch is not pinned" >&2; exit 2; }
mkdir -p "$(dirname "$BOOT_REPO")"
[[ ! -L $BOOT_REPO ]] || { echo "XW FATAL: local clone is a symlink" >&2; exit 2; }
if [[ ! -e $BOOT_REPO ]]; then
    git clone --quiet --single-branch --branch "$XW_BRANCH_NIELSEN" "$BOOT_MIRROR" "$BOOT_REPO"
fi
BOOT_STATUS=$(git -C "$BOOT_REPO" status --porcelain --untracked-files=all)
[[ $(git -C "$BOOT_REPO" rev-parse HEAD) == "$XW_PIN_NIELSEN" \
   && -z $BOOT_STATUS ]] \
    || { echo "XW FATAL: local Nielsen clone is dirty or unpinned" >&2; exit 2; }
cd "$BOOT_REPO"
source slurm/crosswalk-env.sh
xw_preflight
cd "$XW_REPO"
xw_refuse_outputs "$RMS_VINTAGE/market_panel_rms.parquet" "$VINTAGE/rms-d3-log.org"
[[ -f $VINTAGE/market_panel.parquet ]] || xw_die "no $VINTAGE/market_panel.parquet (job 2)"

echo; echo "=== rms_market_panel.py -> $RMS_VINTAGE $(date -Is) ==="
env -u RMS_OUT PANEL_CLEAN="$VINTAGE" PANEL_RAW="$RAW" JOBS=8 \
    "$PY" src/rms_market_panel.py --years 2006 2024 \
        --outdir "$RMS_VINTAGE" --log "$VINTAGE/rms-d3-log.org"

ls -l "$RMS_VINTAGE/market_panel_rms.parquet" "$RMS_VINTAGE"/diagnostics/
sha256sum "$RMS_VINTAGE"/market_panel_rms.parquet "$RMS_VINTAGE"/diagnostics/*.parquet \
    | tee -a "$VINTAGE/SHA256SUMS.new"
xw_accepted_unchanged
echo "=== done $(date -Is) ==="
