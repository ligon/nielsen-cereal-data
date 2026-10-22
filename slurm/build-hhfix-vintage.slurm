#!/bin/bash
#SBATCH --job-name=nl-hhfix-vintage
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=log/build-hhfix-vintage-%j.out
#SBATCH --error=log/build-hhfix-vintage-%j.err

# Build the data/clean-hhfix-2026-10-22 vintage beside data/clean (branch
# fix/household-brands-hhsize-2026-10-22).  data/clean is only read.
#
#   defect 1: household_panel / household_base_shares re-ranked brands without
#             AM's CTL drop (CTL BR 536746 in, M-O-M GOLDEN PUFFS 591705 out);
#             both builders now take the pinned 45 (config/brands_45.csv).
#   defect 2: demographics_household gains household_size.
#
# Rebuilt: demographics_household, household_panel, household_base_shares.
# Copied unchanged from data/clean: market_panel, demographics_market,
# nutrition, nutrition_2021_brands, brand_crosswalk, brand_ranking.
# Gates (the job fails if any fails):
#   - demographics_household minus household_size .equals() the old file
#     (same (i, t) rows, same order, same columns): BLP agent draws unchanged;
#   - the regenerated demographics_market .equals() the old one (then the old
#     file is restored byte-for-byte);
#   - AM with the pinned 45, run into a scratch dir, reproduces the old
#     market_panel exactly (.equals());
#   - household_panel's j set is the pinned 45.
#
# Submit from the repo (or worktree) whose code should run:
#   sbatch slurm/build-hhfix-vintage.slurm
# DATA_REPO is the checkout holding data/{raw,clean} (default: the mirror).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p log
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
umask 027

DATA_REPO=${DATA_REPO:-/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data}
VENV=${VENV:-$DATA_REPO/.venv/bin/python}
RAW=$DATA_REPO/data/raw
OLD=$DATA_REPO/data/clean
VINTAGE=${VINTAGE:-$DATA_REPO/data/clean-hhfix-2026-10-22}
VERIFY=$DATA_REPO/log/hhfix-2026-10-22-am-verify-${SLURM_JOB_ID}

echo "=== node $(hostname); start $(date) ==="
CODE_REV="$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "code: $(pwd) @ $CODE_REV"
git status --short | sed 's/^/  dirty: /'
echo "old: $OLD"; echo "vintage: $VINTAGE"

if [ -e "$VINTAGE/household_panel.parquet" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "REFUSING: $VINTAGE already built (FORCE=1 to rebuild)"; exit 2
fi
mkdir -p "$VINTAGE" "$VERIFY"
chgrp fc_jevons "$VINTAGE" "$VERIFY" && chmod 2750 "$VINTAGE" "$VERIFY"

echo; echo "--- seed: copy the unchanged files ---"
for f in market_panel demographics_market nutrition nutrition_2021_brands \
         brand_crosswalk brand_ranking; do
    cp --preserve=timestamps "$OLD/$f.parquet" "$VINTAGE/$f.parquet" || exit 3
done
# the old demographics_household is NOT seeded: DG writes the new one first,
# and AH reads it from the vintage (household_size is excluded from its merge)
cp --preserve=timestamps "$OLD/nutrition.parquet" "$VERIFY/nutrition.parquet"

RC=0
echo; echo "=== demographics (DG) -> vintage ==="
"$VENV" src/demographics.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE" || RC=$?
[ "$RC" -ne 0 ] && { echo "DG FAILED rc=$RC"; exit "$RC"; }

echo; echo "=== aggregate_households (AH) -> vintage ==="
"$VENV" src/aggregate_households.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE" || RC=$?
[ "$RC" -ne 0 ] && { echo "AH FAILED rc=$RC"; exit "$RC"; }

echo; echo "=== aggregate_markets (AM, pinned 45) -> $VERIFY (check only) ==="
"$VENV" src/aggregate_markets.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VERIFY" || RC=$?
[ "$RC" -ne 0 ] && { echo "AM FAILED rc=$RC"; exit "$RC"; }

echo; echo "=== gates ==="
OLD="$OLD" VINTAGE="$VINTAGE" VERIFY="$VERIFY" "$VENV" - <<'EOF'
import os, shutil, sys
import pandas as pd
old, new, ver = (os.environ[k] for k in ("OLD", "VINTAGE", "VERIFY"))
rd = lambda d, f: pd.read_parquet(os.path.join(d, f + ".parquet"))
ok = True
def gate(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")

dh_o, dh_n = rd(old, "demographics_household"), rd(new, "demographics_household")
gate(f"demographics_household rows old={len(dh_o):,} new={len(dh_n):,}",
     len(dh_o) == len(dh_n))
gate("demographics_household minus household_size equals old (index, order, "
     "columns, dtypes)",
     dh_n.drop(columns="household_size").equals(dh_o))
hs = dh_n["household_size"]
gate(f"household_size non-null int in 1..9 (dtype {hs.dtype})",
     hs.notna().all() and pd.api.types.is_integer_dtype(hs)
     and hs.between(1, 9).all())

dm_n = rd(new, "demographics_market")
same_dm = dm_n.equals(rd(old, "demographics_market"))
gate("regenerated demographics_market equals old", same_dm)
if same_dm:
    shutil.copy2(os.path.join(old, "demographics_market.parquet"),
                 os.path.join(new, "demographics_market.parquet"))
    print("    restored the old demographics_market.parquet byte-for-byte")

mp_v, mp_o = rd(ver, "market_panel"), rd(old, "market_panel")
gate(f"AM (pinned 45) reproduces market_panel exactly ({len(mp_v):,} rows)",
     mp_v.equals(mp_o))
if not mp_v.equals(mp_o):
    print("    shapes", mp_v.shape, mp_o.shape)

pin = set(pd.read_csv("config/brands_45.csv")["j"].astype("int64"))
hp = rd(new, "household_panel")
gate("household_panel j set == pinned 45",
     set(hp.index.get_level_values("j")) == pin)
gate("household_panel has no household_size column",
     "household_size" not in hp.columns)
import pyarrow.parquet as pq
schema = lambda d: pq.read_schema(os.path.join(d, "household_panel.parquet")).names
gate("household_panel columns == old columns", schema(new) == schema(old))
bs = rd(new, "household_base_shares")
gate("household_base_shares j set within the pin",
     set(bs.index.get_level_values("j")) <= pin)
sys.exit(0 if ok else 1)
EOF
VRC=$?
echo; echo "--- vintage contents ---"
ls -l "$VINTAGE"
( cd "$VINTAGE" && sha256sum *.parquet ) | tee "$VINTAGE/SHA256SUMS"
{ echo "vintage: $(basename "$VINTAGE")"
  echo "built: $(date -Is) by Slurm job $SLURM_JOB_ID on $(hostname)"
  echo "code: nielsen-cereal-data $CODE_REV"
  echo "rebuilt: demographics_household (+household_size), household_panel, household_base_shares"
  echo "copied unchanged from data/clean: market_panel, demographics_market, nutrition,"
  echo "  nutrition_2021_brands, brand_crosswalk, brand_ranking"
  echo "gates rc: $VRC (see log/build-hhfix-vintage-$SLURM_JOB_ID.out)"; } > "$VINTAGE/VINTAGE.txt"
echo "=== gates rc=$VRC; end $(date) ==="
exit "$VRC"
