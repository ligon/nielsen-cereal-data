#!/bin/bash
#SBATCH --job-name=xw-markets
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.out
#SBATCH --error=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.err

# charter-crosswalk A7, job 2 of 5 (docs/crosswalk-slurm-plan.org).
# Adapted from slurm/rebuild-markets.slurm (same savio3 / fc_jevons / 90G /
# 2h recipe; that job, 35521774, took 1m33s and 27 GB MaxRSS).  Differences:
# the crosswalk checkout, --raw-dir / --clean-dir (writes ONLY
# $VINTAGE/market_panel.parquet; no backup needed, data/clean is not touched),
# and checks that fit the crosswalk instead of the sodium ones:
#   - the panel's j set == config/brands_45.csv (A3);
#   - pre-2021 cells: revenue and servings identical to the accepted panel for
#     every j outside the two merge targets (Q2: 620554 Great Grains takes
#     620485, 557579 Trix takes 747763), cell for cell; changed cells by j;
#   - 2021-24 brand-year revenue old vs new for the brands the charter names.
# A pre-2021 change outside the merge targets is printed as UNEXPECTED and
# fails the job (the chain stops for EL to look).
#
#   sbatch --dependency=afterok:<households job> slurm/crosswalk-markets.slurm

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
xw_refuse_outputs "$VINTAGE/market_panel.parquet"

echo; echo "=== aggregate_markets.py (full config range) $(date -Is) ==="
"$PY" src/aggregate_markets.py --config config/default.yaml \
    --raw-dir "$RAW" --clean-dir "$VINTAGE"

echo; echo "=== checks $(date -Is) ==="
OLD="$OLD" VINTAGE="$VINTAGE" "$PY" - <<'EOF'
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import brand_route as br
old = pd.read_parquet(f"{os.environ['OLD']}/market_panel.parquet").reset_index()
new = pd.read_parquet(f"{os.environ['VINTAGE']}/market_panel.parquet").reset_index()
for d in (old, new):
    d["year"] = d["t"].str[-4:].astype(int)
ok = True
want = set(br.load_brands_45()["j"].astype("int64"))
good = set(new["j"].astype("int64")) == want
ok &= good
print(f"  rows old {len(old):,} new {len(new):,}; j set == brands_45: {good}")
merged_targets = set(br.load_merge_sets().values())
key = ["t", "m", "j"]
po, pn = old[old["year"] <= 2020], new[new["year"] <= 2020]
m = po.merge(pn, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
chg = (m["_merge"] != "both")
for c in ["revenue", "servings"]:
    a, b = m[f"{c}_old"].to_numpy(float), m[f"{c}_new"].to_numpy(float)
    chg |= ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True)
by_j = m[chg].groupby("j").size()
print(f"  pre-2021 cells changed (revenue/servings or presence): {int(chg.sum()):,}")
for j, n in by_j.items():
    tag = "merge target (Q2)" if j in merged_targets else "UNEXPECTED"
    print(f"    j={j}: {n:,} cells  [{tag}]")
    ok &= j in merged_targets
names = {557579: "Trix", 620554: "Great Grains", 579277: "Kashi Go Lean Crunch",
         620458: "Cocoa Pebbles", 557426: "Cinnamon Toast Crunch",
         557461: "Fiber One", 579923: "Frosted Mini-Wheats"}
ro = old.groupby(["j", "year"])["revenue"].sum()
rn = new.groupby(["j", "year"])["revenue"].sum()
print("  2021-24 brand-year revenue, new / old (inf: absent from the old panel):")
for j, nm in names.items():
    r = []
    for y in range(2021, 2025):
        o, n = ro.get((j, y), 0.0), rn.get((j, y), 0.0)
        r.append(n / o if o > 0 else (np.inf if n > 0 else np.nan))
    print(f"    {nm:24s} " + "  ".join(f"{y}: {v:6.3f}" for y, v in zip(range(2021, 2025), r)))
sys.exit(0 if ok else 1)
EOF

sha256sum "$VINTAGE"/market_panel.parquet | tee -a "$VINTAGE/SHA256SUMS.new"
xw_accepted_unchanged
echo "=== done $(date -Is) ==="
