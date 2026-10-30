#!/usr/bin/env bash
# charter-crosswalk plan step 0 (docs/crosswalk-slurm-plan.org).  Run ONCE on
# the login node from the crosswalk checkout, BEFORE any sbatch:
#
#     cd /global/scratch/fsa/fc_jevons/ligon/mirrors/crosswalk-work/nielsen-cereal-data
#     bash slurm/crosswalk-step0-seed.sh
#
# Seconds, a few hundred MB of reads.  It
#   1. creates $VINTAGE/logs and $RMS_VINTAGE (group-only, umask 027);
#   2. records the sha256 of every accepted file (data/clean/*.parquet, the
#      accepted RMS panel and its diagnostics) in $VINTAGE/SHA256SUMS.accepted;
#      every job re-checks them at its end;
#   3. seeds nutrition and, for the original xw1 run, demographics.  The
#      combined xw2 run sets XW_REBUILD_DEMOG=1 and builds demographics in
#      job 1 so household_size is present;
#   4. gives $RMS_VINTAGE read-only symlinks to the accepted per-year RMS
#      parquet dirs and master, so tests/test_rms_market_panel.py and
#      src/rms_validate.py can run with RMS_OUT=$RMS_VINTAGE (they read the
#      panel AND movement/stores/master from RMS_OUT).
# It refuses to overwrite anything already in the vintage.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/crosswalk-env.sh

[[ $(realpath -m "$VINTAGE") != $(realpath -m "$OLD") ]] || xw_die "VINTAGE is data/clean"
[[ $(realpath -m "$RMS_VINTAGE") != $(realpath -m "$RMS") ]] || xw_die "RMS_VINTAGE is the accepted dir"
if [[ ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
    [[ $(realpath -m "$VINTAGE") != $(realpath -m "$XW1_VINTAGE") \
       && $(realpath -m "$RMS_VINTAGE") != $(realpath -m "$XW1_RMS_VINTAGE") ]] \
        || xw_die "combined output cannot be an xw1 directory"
    [[ ! -e $VINTAGE && ! -e $RMS_VINTAGE ]] \
        || xw_die "combined vintage already exists; inspect before retry"
fi

mkdir -p "$XW_LOGS" "$RMS_VINTAGE"

# 2. accepted files
if [[ -f $ACCEPTED_SUMS ]]; then
    echo "$ACCEPTED_SUMS exists; verifying instead of rewriting"
    sha256sum --quiet -c "$ACCEPTED_SUMS"
else
    xw_accepted_files | xargs sha256sum > "$ACCEPTED_SUMS"
    echo "recorded $(wc -l < "$ACCEPTED_SUMS") accepted files in $ACCEPTED_SUMS"
fi
if [[ ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
    xw_xw1_files | xargs sha256sum > "$VINTAGE/SHA256SUMS.xw1"
    echo "recorded $(wc -l < "$VINTAGE/SHA256SUMS.xw1") xw1 reference files"
fi

# 3. seed the unchanged inputs (no -p: keep the vintage group-only)
seed_files=(nutrition.parquet nutrition_2021_brands.parquet)
if [[ ${XW_REBUILD_DEMOG:-0} != 1 ]]; then
    seed_files+=(demographics_household.parquet demographics_market.parquet)
fi
for f in "${seed_files[@]}"; do
    [[ -e $VINTAGE/$f ]] && xw_die "$VINTAGE/$f already exists"
    cp --preserve=timestamps "$OLD/$f" "$VINTAGE/$f"
    cmp -s "$OLD/$f" "$VINTAGE/$f" || xw_die "seed copy differs: $f"
done
( cd "$VINTAGE" && sha256sum "${seed_files[@]}" ) > "$VINTAGE/SHA256SUMS.seed"
echo "seeded $VINTAGE:"; cat "$VINTAGE/SHA256SUMS.seed"

# 4. RMS symlink farm (relative links into the accepted dir; read-only use)
for y in $(seq 2006 2024); do
    [[ -e $RMS_VINTAGE/$y ]] || ln -s "../$y" "$RMS_VINTAGE/$y"
done
[[ -e $RMS_VINTAGE/master_products_1344.parquet ]] \
    || ln -s ../master_products_1344.parquet "$RMS_VINTAGE/master_products_1344.parquet"
ls -l "$RMS_VINTAGE" | head -5

# not -R: never walk into the accepted RMS dirs through the symlinks
chmod g+rX,o-rwx "$VINTAGE" "$XW_LOGS" "$RMS_VINTAGE"
echo "step 0 done $(date -Is)"
