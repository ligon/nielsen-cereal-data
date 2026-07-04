#!/bin/bash
#SBATCH --job-name=nl-rebuild-markets
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=log/rebuild-markets-%j.out
#SBATCH --error=log/rebuild-markets-%j.err

# Rebuild the market panel through aggregate_markets (Stage 3) so the DERIVED
# Z_sodium aggregate picks up the sodium mg->g normalization (commit 198cb2d).
# The 198cb2d in-place patch divided only the raw `sodium` column by 1000;
# market_panel also carries Z_sodium = sum_j S_jt * sodium_j, which was left
# at mg scale (verified stored/recomputed ratio == 1000.0000 exactly, vs
# 1.0000 for the other four Z columns).  Rebuilding recomputes the aggregates
# from gram-sodium at the source, mirroring the household-panel rebuild
# (job 35491266, slurm/rebuild-households.slurm).
#
# Direct invocation, NOT `make markets` (skip the heavy extract/crosswalk
# prereqs).  Same 90G savio3 fc_jevons recipe as the household rebuild: the
# dominant cost is loading the same purchases/trips/panelists raw data.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p log
export PYTHONUNBUFFERED=1
VENV=.venv/bin/python
PANEL=data/clean/market_panel.parquet
BACKUP="log/market_panel.pre-rebuild-${SLURM_JOB_ID}.parquet"

echo "=== node $(hostname); start $(date) ==="
echo "--- BEFORE: stale-Z_sodium diagnosis on the current panel ---"
"$VENV" - <<'EOF' || true
import pandas as pd
mp = pd.read_parquet("data/clean/market_panel.parquet").reset_index()
print(f"  sodium mean {mp['sodium'].mean():.5f} (expect ~0.146 g, already patched)")
for c in ["sugar", "fiber", "protein", "fat", "sodium"]:
    z = mp.assign(v=mp["shares"] * mp[c]).groupby(["t", "m"])["v"].sum()
    st = mp.drop_duplicates(["t", "m"]).set_index(["t", "m"])[f"Z_{c}"]
    print(f"  Z_{c}: stored/recomputed ratio mean {(st / z).mean():.4f}")
EOF

cp -p "$PANEL" "$BACKUP"
echo "backed up current panel to $BACKUP"

echo
echo "=== aggregate_markets (full config range 2004-2024) ==="
RC=0
"$VENV" src/aggregate_markets.py --config config/default.yaml || RC=$?
echo "aggregate_markets rc=$RC"
[ "$RC" -ne 0 ] && { echo "=== FAILED; end $(date) ==="; exit "$RC"; }

echo
echo "=== AFTER: verification (fails the job if inconsistent) ==="
"$VENV" - <<EOF
import sys
import numpy as np
import pandas as pd

new = pd.read_parquet("data/clean/market_panel.parquet").reset_index()
old = pd.read_parquet("$BACKUP").reset_index()
ok = True

# 1) Internal consistency: every stored Z_l == sum_j shares*zeta_l per (t,m).
for c in ["sugar", "fiber", "protein", "fat", "sodium"]:
    z = new.assign(v=new["shares"] * new[c]).groupby(["t", "m"])["v"].sum()
    st = new.drop_duplicates(["t", "m"]).set_index(["t", "m"])[f"Z_{c}"]
    r = (st / z).to_numpy()
    good = np.allclose(r, 1.0, rtol=1e-9)
    ok &= good
    print(f"  Z_{c}: ratio mean {r.mean():.6f}  [{'OK' if good else 'FAIL'}]")

# 2) Sodium stays gram-scale.
sod = new["sodium"].mean()
good = 0.10 < sod < 0.20
ok &= good
print(f"  sodium mean {sod:.5f} (expect ~0.146 g)  [{'OK' if good else 'FAIL'}]")

# 3) Nothing but Z_sodium moved vs the pre-rebuild panel (aligned on t,m,j).
key = ["t", "m", "j"]
merged = old.merge(new, on=key, suffixes=("_old", "_new"))
good = len(merged) == len(old) == len(new)
ok &= good
print(f"  rows old={len(old)} new={len(new)} aligned={len(merged)}  "
      f"[{'OK' if good else 'FAIL'}]")
for c in ["shares", "prices", "sugar", "fiber", "protein", "fat", "sodium",
          "Z_sugar", "Z_fiber", "Z_protein", "Z_fat", "W_exp", "revenue"]:
    close = np.allclose(merged[f"{c}_old"], merged[f"{c}_new"],
                        rtol=1e-9, equal_nan=True)
    ok &= close
    if not close:
        d = (merged[f"{c}_new"] - merged[f"{c}_old"]).abs().max()
        print(f"  {c}: CHANGED (max |diff| {d:.3g})  [FAIL]")
print("  unchanged-column sweep done")
rz = (merged["Z_sodium_old"] / merged["Z_sodium_new"]).to_numpy()
good = np.allclose(rz, 1000.0, rtol=1e-9)
ok &= good
print(f"  Z_sodium old/new ratio mean {rz.mean():.4f} (expect 1000)  "
      f"[{'OK' if good else 'FAIL'}]")

sys.exit(0 if ok else 1)
EOF
VRC=$?
echo "=== rc=$VRC; end $(date) ==="
exit "$VRC"
