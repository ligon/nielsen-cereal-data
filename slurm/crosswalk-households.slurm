#!/bin/bash
#SBATCH --job-name=xw-households
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.out
#SBATCH --error=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.err

# charter-crosswalk A7, job 1 of 5 (docs/crosswalk-slurm-plan.org).
# Adapted from slurm/rebuild-households.slurm (same savio3 / fc_jevons / 90G /
# 4h recipe; that job, 35491266, took 1m44s and 27 GB MaxRSS).  Differences:
# runs the crosswalk checkout, writes ONLY to the vintage dir ($VINTAGE) via
# --clean-dir, reads Homescan raw via --raw-dir, and never touches data/clean.
#
#   1. brand_crosswalk.py -> $VINTAGE/brand_name_map.parquet (the new
#      crosswalk artifact: the 2021-24 name map of src/brand_route.py)
#   2. aggregate_households.py -> $VINTAGE/household_panel.parquet and
#      household_base_shares.parquet (pinned 45, shared route; ledger G1:
#      CTL BR leaves, M-O-M Golden Puffs enters)
#   3. checks: j set == config/brands_45.csv; old vs new j sets and rows
#
# Submit from the crosswalk checkout, after step 0:
#   sbatch slurm/crosswalk-households.slurm

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
xw_preflight seed
cd "$XW_REPO"
xw_refuse_outputs "$VINTAGE/brand_name_map.parquet" \
    "$VINTAGE/household_panel.parquet" "$VINTAGE/household_base_shares.parquet"

if [[ ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
    echo; echo "=== 0. demographics with household_size $(date -Is) ==="
    [[ ! -e $VINTAGE/demographics_household.parquet && ! -e $VINTAGE/demographics_market.parquet ]] \
        || xw_die "combined vintage already has demographics; refusing overwrite"
    "$PY" src/demographics.py --config config/default.yaml \
        --raw-dir "$RAW" --clean-dir "$VINTAGE"
    OLD="$OLD" VINTAGE="$VINTAGE" "$PY" - <<'EOF'
import os
import pandas as pd
old, new = os.environ["OLD"], os.environ["VINTAGE"]
dh_old = pd.read_parquet(f"{old}/demographics_household.parquet")
dh_new = pd.read_parquet(f"{new}/demographics_household.parquet")
dm_old = pd.read_parquet(f"{old}/demographics_market.parquet")
dm_new = pd.read_parquet(f"{new}/demographics_market.parquet")
assert "household_size" not in dh_old and "household_size" in dh_new
size = dh_new["household_size"]
assert size.notna().all() and pd.api.types.is_integer_dtype(size)
assert size.between(1, 9).all()
assert dh_new.drop(columns="household_size").equals(dh_old)
assert dm_new.equals(dm_old)
print(f"  household_size: {len(size):,}/{len(size):,} non-null, range "
      f"{size.min()}..{size.max()}; other demographics equal accepted")
EOF
    sha256sum "$VINTAGE"/demographics_{household,market}.parquet \
        | tee -a "$VINTAGE/SHA256SUMS.new"
fi

echo; echo "=== 1. brand_crosswalk.py (name map) $(date -Is) ==="
"$PY" src/brand_crosswalk.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE"

echo; echo "=== 2. aggregate_households.py (full config range) $(date -Is) ==="
"$PY" src/aggregate_households.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE"

echo; echo "=== 3. checks $(date -Is) ==="
OLD="$OLD" VINTAGE="$VINTAGE" "$PY" - <<'EOF'
import os, sys
import pandas as pd
sys.path.insert(0, "src")
import brand_route as br
old, new = os.environ["OLD"], os.environ["VINTAGE"]
want = set(br.load_brands_45()["j"].astype("int64"))
cols = ["panel_year", "revenue"]      # j comes back as an index level
hn = pd.read_parquet(f"{new}/household_panel.parquet", columns=cols).reset_index()
ho = pd.read_parquet(f"{old}/household_panel.parquet", columns=cols).reset_index()
jn, jo = set(hn["j"].astype("int64")), set(ho["j"].astype("int64"))
print(f"  household_panel rows old {len(ho):,} new {len(hn):,}")
print(f"  j only in old: {sorted(jo - jn)}  only in new: {sorted(jn - jo)}"
      "  (ledger G1 expects 536746 CTL BR out, 591705 Golden Puffs in)")
ok = jn == want
print(f"  household_panel j set == config/brands_45.csv: {ok}")
bs = pd.read_parquet(f"{new}/household_base_shares.parquet")
print(f"  household_base_shares rows {len(bs):,}")
if os.environ.get("XW_REBUILD_DEMOG") == "1":
    xw1 = os.environ["XW1_VINTAGE"]
    h1 = pd.read_parquet(f"{xw1}/household_panel.parquet",
                         columns=["panel_year", "m", "S", "servings"])
    h2 = pd.read_parquet(f"{new}/household_panel.parquet",
                         columns=["panel_year", "m", "S", "servings"])
    common = h1.index.intersection(h2.index)
    print(f"  xw1 -> xw2 household rows: {len(h1):,} -> {len(h2):,}; "
          f"shared (i,t,j): {len(common):,}")
    for c in ("panel_year", "m", "S", "servings"):
        a, b = h1.loc[common, c], h2.loc[common, c]
        changed = ~((a == b) | (a.isna() & b.isna()))
        print(f"    {c} changed among shared rows: {int(changed.sum()):,}")
nm = pd.read_parquet(f"{new}/brand_name_map.parquet")
print(f"  brand_name_map: {len(nm)} names, {int(nm['j'].notna().sum())} placed")
sys.exit(0 if ok else 1)
EOF

sha256sum "$VINTAGE"/brand_name_map.parquet "$VINTAGE"/household_panel.parquet \
    "$VINTAGE"/household_base_shares.parquet | tee -a "$VINTAGE/SHA256SUMS.new"
xw_accepted_unchanged
echo "=== done $(date -Is) ==="
