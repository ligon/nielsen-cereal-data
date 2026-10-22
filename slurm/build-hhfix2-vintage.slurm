#!/bin/bash
#SBATCH --job-name=nl-hhfix2-vintage
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=log/build-hhfix2-vintage-%j.out
#SBATCH --error=log/build-hhfix2-vintage-%j.err

# Build data/clean-hhfix2-2026-10-22 beside data/clean-hhfix-2026-10-22
# (branch hh-tiebreak-2026-10-22).  Both existing vintages are only read.
#
#   defect: aggregate_households picked a cell's (panel_year, DMA) by spend
#           with an unstable sort, so on an exact spend tie between two
#           panel years the winner depended on unrelated rows.  The rule is
#           now a total order (_dominant_panel_year): spend in cents, then
#           panel_year == year(t), then panel_year, then DMA_Cd.
#
# Rebuilt: household_panel, household_base_shares.
# Copied unchanged from clean-hhfix-2026-10-22: market_panel,
# demographics_household, demographics_market, nutrition,
# nutrition_2021_brands, brand_crosswalk, brand_ranking.
# Gates (the job fails if any fails):
#   - src/aggregate_markets.py and src/demographics.py unchanged since 86d05bb
#     (so the copied market/demographics files are what this code builds);
#   - household_panel: same (i, t, j) index in the same order, same columns
#     and dtypes as the base vintage; only S, m, panel_year may differ, and
#     only on rows whose new panel_year is year(t) (ties now go to the
#     calendar year); every tie cell from the diagnosis (TIES) has
#     panel_year == year(t);
#   - household_base_shares .equals() the base (Q1Y2014 lies inside panel
#     year 2014, 2013-12-29..2014-12-27, so it has no two-year cells); the
#     base file is then restored byte-for-byte.
#
# Submit from the worktree whose code should run (commit first):
#   sbatch slurm/build-hhfix2-vintage.slurm
# DATA_REPO is the checkout holding data/{raw,clean*} (default: the mirror).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p log
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
umask 027

DATA_REPO=${DATA_REPO:-/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data}
VENV=${VENV:-$DATA_REPO/.venv/bin/python}
RAW=$DATA_REPO/data/raw
BASE=${BASE:-$DATA_REPO/data/clean-hhfix-2026-10-22}
VINTAGE=${VINTAGE:-$DATA_REPO/data/clean-hhfix2-2026-10-22}
TIES=${TIES:-}

echo "=== node $(hostname); start $(date) ==="
CODE_REV="$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "code: $(pwd) @ $CODE_REV"
git status --short | sed 's/^/  dirty: /'
echo "base: $BASE"; echo "vintage: $VINTAGE"; echo "ties: ${TIES:-<none>}"

if [ -e "$VINTAGE/household_panel.parquet" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "REFUSING: $VINTAGE already built (FORCE=1 to rebuild)"; exit 2
fi
if ! git diff --quiet 86d05bb -- src/aggregate_markets.py src/demographics.py; then
    echo "REFUSING: AM/DG changed since 86d05bb; copied files would be stale"
    exit 4
fi
mkdir -p "$VINTAGE"
chgrp fc_jevons "$VINTAGE" && chmod 2750 "$VINTAGE"

echo; echo "--- seed: copy the unchanged files from the base vintage ---"
for f in market_panel demographics_household demographics_market nutrition \
         nutrition_2021_brands brand_crosswalk brand_ranking; do
    cp --preserve=timestamps "$BASE/$f.parquet" "$VINTAGE/$f.parquet" || exit 3
done

RC=0
echo; echo "=== aggregate_households (AH) -> vintage ==="
"$VENV" src/aggregate_households.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE" || RC=$?
[ "$RC" -ne 0 ] && { echo "AH FAILED rc=$RC"; exit "$RC"; }

echo; echo "=== gates ==="
BASE="$BASE" VINTAGE="$VINTAGE" TIES="$TIES" "$VENV" - <<'EOF'
import os, shutil, sys
import numpy as np
import pandas as pd
base, new, ties = (os.environ[k] for k in ("BASE", "VINTAGE", "TIES"))
rd = lambda d, f: pd.read_parquet(os.path.join(d, f + ".parquet"))
ok = True
def gate(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")

o, n = rd(base, "household_panel"), rd(new, "household_panel")
gate(f"household_panel rows base={len(o):,} new={len(n):,}", len(o) == len(n))
gate("household_panel index identical (values and order)",
     o.index.equals(n.index))
gate("household_panel columns and dtypes identical",
     list(o.columns) == list(n.columns) and (o.dtypes == n.dtypes).all())
diff = {c: ~((o[c] == n[c]) | (o[c].isna() & n[c].isna())).to_numpy()
        for c in o.columns}
moved = {c: int(d.sum()) for c, d in diff.items() if d.any()}
print(f"    columns with changed rows: {moved}")
gate("only S, m, panel_year changed", set(moved) <= {"S", "m", "panel_year"})
chg = np.logical_or.reduce([diff[c] for c in ("S", "m", "panel_year")])
year_t = n.index.get_level_values("t").str[-4:].astype(int)
gate(f"every changed row ({int(chg.sum()):,}) now has panel_year == year(t)",
     (n["panel_year"].to_numpy()[chg] == year_t[chg]).all())
gate("panel_year changed wherever S or m changed",
     (diff["panel_year"] == chg).all())
if ties:
    tc = pd.read_parquet(ties)[["household_code", "t", "brand_code_uc"]]
    idx = pd.MultiIndex.from_frame(tc, names=["i", "t", "j"])
    hit = n.reindex(idx)
    present = hit["panel_year"].notna()
    yt = hit.index.get_level_values("t").str[-4:].astype(int)
    gate(f"tie cells: {int(present.sum()):,} of {len(tc):,} present in the "
         f"panel; all have panel_year == year(t)",
         (hit.loc[present.to_numpy(), "panel_year"].to_numpy()
          == yt[present.to_numpy()]).all())
    in_ties = pd.Series(chg, index=n.index).loc[lambda s: s].index.isin(idx)
    print(f"    changed rows that are diagnosis exact-tie cells: "
          f"{int(in_ties.sum()):,} of {int(chg.sum()):,}")

bo, bn = rd(base, "household_base_shares"), rd(new, "household_base_shares")
same_bs = bo.equals(bn)
gate(f"household_base_shares equals base ({len(bn):,} rows)", same_bs)
if same_bs:
    shutil.copy2(os.path.join(base, "household_base_shares.parquet"),
                 os.path.join(new, "household_base_shares.parquet"))
    print("    restored the base household_base_shares.parquet byte-for-byte")
sys.exit(0 if ok else 1)
EOF
VRC=$?
echo; echo "--- vintage contents ---"
ls -l "$VINTAGE"
( cd "$VINTAGE" && sha256sum *.parquet ) | tee "$VINTAGE/SHA256SUMS"
{ echo "vintage: $(basename "$VINTAGE")"
  echo "built: $(date -Is) by Slurm job $SLURM_JOB_ID on $(hostname)"
  echo "code: nielsen-cereal-data $CODE_REV"
  echo "base: $(basename "$BASE")"
  echo "rebuilt: household_panel (deterministic cell panel_year tie-break),"
  echo "  household_base_shares (equal to base; base bytes kept if gate passed)"
  echo "copied unchanged from $(basename "$BASE"): market_panel, demographics_household,"
  echo "  demographics_market, nutrition, nutrition_2021_brands, brand_crosswalk, brand_ranking"
  echo "gates rc: $VRC (see $(pwd)/log/build-hhfix2-vintage-$SLURM_JOB_ID.out)"; } > "$VINTAGE/VINTAGE.txt"
echo "=== gates rc=$VRC; end $(date) ==="
exit "$VRC"
