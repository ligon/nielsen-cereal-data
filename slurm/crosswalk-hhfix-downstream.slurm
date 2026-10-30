#!/bin/bash
#SBATCH --job-name=xw-downstream
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-hhfix-2026-10-23/logs/%x-%j.out
#SBATCH --error=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-hhfix-2026-10-23/logs/%x-%j.err

# Combined crosswalk + household fixes: B3 -> B2 -> B5 (CPU only), job 5 of 5
# (docs/crosswalk-slurm-plan.org).  Runs CerealDemand (branch crosswalk) on the
# vintage.  Writes only under $VINTAGE (bartik_demog_*.parquet, b2/, laws/) and
# $RMS_VINTAGE/laws; never var/laws, never K-Aggregators/var.
#
# B3 BEFORE B2 (the task list says B2 -> B3): B2's new side fingerprints the 6
# DEMOG_BARTIK het specs, which read <datadir>/bartik_demog_<base>.parquet;
# on a new vintage CerealDemand never falls back to the old Bartik files
# (src/data.py _bartik_demog_path), so B2 needs B3's files first.
#
# Timings measured on this cluster (2026-10-19): B2 about 7 min and 2 GB per
# side; homog about 4 min (homog.rms took 241 s); Bartik builds read the
# 5.4M-row household panel (minutes).  het is NOT run here (GPU; EL's OK).
#
#   sbatch --dependency=afterok:<checks job> slurm/crosswalk-downstream.slurm

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
export XW_REPO
cd "$XW_CD"
xw_refuse_outputs "$VINTAGE/bartik_demog_Q1Y2004.parquet" \
    "$VINTAGE/b2/new.json" "$VINTAGE/laws/homog.xw2.law" \
    "$RMS_VINTAGE/laws/homog.rms-xw2.law" \
    "$RMS_VINTAGE/rms-validation-b5.org"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1     # as src/submit.py (emu_gmm enables x64 on import too)
# emu_gmm.recommended_env(): one JAX CPU device, BLAS single-threaded
export JAX_NUM_CPU_DEVICES=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset BARTIK_DEMOG_PATH CEREALDEMAND_LAW_STORE NIELSEN_RMS_DATADIR \
      NIELSEN_RMS_XW1_DATADIR NIELSEN_XW1_DATADIR \
      NIELSEN_RMS_XW2_DATADIR NIELSEN_XW2_DATADIR || true
B2=$VINTAGE/b2
XW_STORE=$VINTAGE/laws
RMS_STORE=$RMS_VINTAGE/laws
mkdir -p "$B2" "$XW_STORE" "$RMS_STORE"

echo; echo "=== B3: bartik_demog from the vintage household panel $(date -Is) ==="
# Q1Y2004 (panel het), Q1Y2014 (windowed het), Q1Y2006 (het on RMS; not
# read by B2 or B5, built for completeness).  build() stamps the sha256 of
# its inputs (household_panel, demographics_household) into the parquet (B1).
for base in Q1Y2004 Q1Y2014 Q1Y2006; do
    NIELSEN_DATADIR="$VINTAGE" BARTIK_BASELINE=$base "$PY" -c \
        "from src.bartik_demog import build; build('$base', '$VINTAGE', '$VINTAGE/bartik_demog_$base.parquet')"
done
VINTAGE="$VINTAGE" KAGG_VAR="$KAGG_VAR" CD_VAR="/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/var" \
"$PY" - <<'EOF'
import os
import numpy as np
import pandas as pd
v = os.environ["VINTAGE"]
old_dirs = {"Q1Y2004": os.environ["KAGG_VAR"], "Q1Y2014": os.environ["KAGG_VAR"],
            "Q1Y2006": os.environ["CD_VAR"]}
for base, od in old_dirs.items():
    new = pd.read_parquet(f"{v}/bartik_demog_{base}.parquet")
    p = f"{od}/bartik_demog_{base}.parquet"
    if not os.path.exists(p):
        print(f"  {base}: no old file at {p}"); continue
    old = pd.read_parquet(p)
    j_old = set(old.index.get_level_values("j")) if "j" in old.index.names else set()
    j_new = set(new.index.get_level_values("j")) if "j" in new.index.names else set()
    common = old.index.intersection(new.index)
    num = [c for c in old.columns if c in new.columns
           and np.issubdtype(old[c].dtype, np.number)]
    a = old.loc[common, num].to_numpy(float)
    b = new.loc[common, num].to_numpy(float)
    moved = int((~np.isclose(a, b, rtol=1e-9, equal_nan=True)).any(axis=1).sum())
    print(f"  {base}: rows old {len(old):,} new {len(new):,} common {len(common):,}; "
          f"common rows that moved {moved:,}; j only old {sorted(j_old - j_new)} "
          f"only new {sorted(j_new - j_old)}")
EOF

echo; echo "=== B2: fingerprints, old/xw1/xw2 (29 Specs + homog.rms) $(date -Is) ==="
NIELSEN_DATADIR="$OLD" "$PY" -m src.vintage_impact fingerprint --label old \
    --out "$B2/old.json" --datadir "$OLD" --rms-datadir "$RMS" --bartik-dir "$KAGG_VAR"
NIELSEN_DATADIR="$XW1_VINTAGE" "$PY" -m src.vintage_impact fingerprint --label xw1 \
    --out "$B2/xw1.json" --datadir "$XW1_VINTAGE" \
    --rms-datadir "$XW1_RMS_VINTAGE" --bartik-dir "$XW1_VINTAGE"
NIELSEN_DATADIR="$VINTAGE" "$PY" -m src.vintage_impact fingerprint --label xw2 \
    --out "$B2/new.json" --datadir "$VINTAGE" --rms-datadir "$RMS_VINTAGE" --bartik-dir "$VINTAGE"
"$PY" -m src.vintage_impact compare "$B2/old.json" "$B2/new.json" --out "$B2/impact.org"
"$PY" -m src.vintage_impact compare "$B2/xw1.json" "$B2/new.json" \
    --out "$B2/impact-xw1-to-xw2.org"

echo; echo "=== B5a: homog on the new Homescan panel (Spec.source xw2) $(date -Is) ==="
CEREALDEMAND_LAW_STORE="$XW_STORE" NIELSEN_XW2_DATADIR="$VINTAGE" NIELSEN_DATADIR="$OLD" \
"$PY" - <<'EOF'
import json, os, sys, time
from src.build import build_one
from src.spec import Spec
from src import lawio, nielsen_xw2
from src.provenance import file_sha256
from src.rms_homog_compare import summarize
sys.path.insert(0, os.path.join(os.environ["XW_REPO"], "src"))
from crosswalk_chain_gates import require_current_law
store = os.environ["CEREALDEMAND_LAW_STORE"]
spec = Spec("homog", source="xw2")
t0 = time.time()
if not lawio.exists(spec, store=store):
    _, uri = build_one(spec)
    os.chmod(uri, 0o640)
    print(f"  {spec.slug()}: saved {uri} ({time.time() - t0:.0f} s)")
panel = nielsen_xw2.market_panel_path()
require_current_law(lawio.read_provenance(spec, store=store),
                    source=spec.source, panel_path=panel,
                    panel_sha=file_sha256(panel),
                    producer_commit=lawio.producer_commit(),
                    expected_vintage="crosswalk-hhfix-2026-10-23")
rows = [summarize(Spec("homog"),
                  store="/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/var/laws"),
        summarize(spec, store=store)]
out = os.path.join(store, "homog_xw2_compare.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2, default=str)
print("| law | N | clusters | theta | SE | J_cluster | dof | p (wild) |")
for r in rows:
    print(f"| {r['slug']} | {r['N']:,} | {r['n_clusters']:,} | {r['theta_orig']:.4f} "
          f"| {r['theta_orig_se']:.4f} | {r['J_cluster']:.1f} | {r['J_cluster_dof']} "
          f"| {r['J_cluster_pvalue']:.4f} |")
print("data provenance:", lawio.read_provenance(spec, store=store).get("data_provenance"))
EOF

echo; echo "=== B5b: homog on the new RMS panel (Spec.source rms-xw2) $(date -Is) ==="
# Dedicated source and slug: homog.rms-xw2.  The RMS vintage reader and the
# Homescan vintage reader provide the new panel and its supporting inputs.
# NIELSEN_DATADIR stays OLD for the accepted panel law's rescaling constants.
CEREALDEMAND_LAW_STORE="$RMS_STORE" NIELSEN_RMS_XW2_DATADIR="$RMS_VINTAGE" \
NIELSEN_XW2_DATADIR="$VINTAGE" NIELSEN_DATADIR="$OLD" PANEL_LAW_STORE="$PANEL_LAW_STORE" \
"$PY" - <<'EOF'
import json, os, sys, time
from src.build import build_one
from src.spec import Spec
from src import lawio, nielsen_rms_xw2
from src.provenance import file_sha256
from src.rms_homog_compare import summarize
sys.path.insert(0, os.path.join(os.environ["XW_REPO"], "src"))
from crosswalk_chain_gates import require_current_law
store = os.environ["CEREALDEMAND_LAW_STORE"]
spec = Spec("homog", source="rms-xw2")
t0 = time.time()
seconds = None
if not lawio.exists(spec, store=store):
    _, uri = build_one(spec)
    os.chmod(uri, 0o640)
    seconds = round(time.time() - t0, 1)
    print(f"  {spec.slug()}: saved {uri} ({seconds} s)")
panel = nielsen_rms_xw2.market_panel_path()
require_current_law(lawio.read_provenance(spec, store=store),
                    source=spec.source, panel_path=panel,
                    panel_sha=file_sha256(panel),
                    producer_commit=lawio.producer_commit(),
                    expected_vintage="crosswalk-hhfix-2026-10-23")
rows = [summarize(Spec("homog"), store=os.environ["PANEL_LAW_STORE"]),
        summarize(spec, store=store)]
for row in rows:
    row["seconds_to_estimate"] = seconds if row["slug"] == spec.slug() else None
out = os.path.join(store, "homog_compare.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
os.chmod(out, 0o640)
print("| law | source | years | base | N | clusters | theta | SE | J_cluster | dof | p (wild) |")
print("|-|-|-|-|-|-|-|-|-|-|-|")
for r in rows:
    yrs = f"{r['years'][0]}-{r['years'][1]}" if r["years"] else "?"
    print(f"| {r['slug']} | {r['source']} | {yrs} | {r['base_period']} | {r['N']:,} "
          f"| {r['n_clusters']:,} | {r['theta_orig']:.4f} | {r['theta_orig_se']:.4f} "
          f"| {r['J_cluster']:.1f} | {r['J_cluster_dof']} | {r['J_cluster_pvalue']:.4f} |")
print("data provenance:", lawio.read_provenance(spec, store=store).get("data_provenance"))
EOF
echo "accepted homog.rms, for comparison:"
cat "$RMS/laws/homog_compare.json" | "$PY" -c \
    "import json,sys; [print(' ', r['slug'], round(r['theta_orig'],4), round(r['theta_orig_se'],4), round(r['J_cluster'],1)) for r in json.load(sys.stdin)]"

echo; echo "=== D4 again, now with the new D6 (homog_compare.json) $(date -Is) ==="
cd "$XW_REPO"
PANEL_CLEAN="$VINTAGE" RMS_OUT="$RMS_VINTAGE" "$PY" src/rms_validate.py \
    --out "$RMS_VINTAGE/rms-validation-b5.org" --d6 "$RMS_STORE/homog_compare.json" \
    --gate-crosswalk

sha256sum "$VINTAGE"/bartik_demog_*.parquet "$B2"/*.json "$XW_STORE"/*.law \
    "$RMS_STORE"/*.law | tee -a "$VINTAGE/SHA256SUMS.new"
xw_accepted_unchanged
echo "=== done $(date -Is) ==="
