#!/bin/bash
#SBATCH --job-name=xw-checks
#SBATCH --partition=savio3
#SBATCH --account=fc_jevons
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.out
#SBATCH --error=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19/logs/%x-%j.err

# charter-crosswalk A7 tests + A8 on the regenerated files, job 4 of 5
# (docs/crosswalk-slurm-plan.org).  Reads only; writes reports into the
# vintage dirs, never into docs/ (the committed docs/rms-validation.org and
# docs/crosswalk-before-after.org stay as they are; copying a report into
# docs/ is a separate, reviewed commit).
#
# Gate (the job fails, so B3/B2/B5 do not start, if any of these fail):
#   a. make rms-test on the new RMS panel (RMS_OUT = $RMS_VINTAGE, whose
#      per-year dirs are step 0's symlinks to the accepted ones);
#   b. the crosswalk tests on the new files: A2 (test_brand_route), A3
#      (test_pinned_brands on the new market panel), A4/A5/A6 on the UPC-dollar
#      evaluation and A5 on both new panel files (test_a5_panel_file).
#      Every unallow-listed failure fails the gate; Kashi uses the committed
#      allow-list with EL's authorized source/year ranges.
# Gated as well:
#   c. D4 runs and section 7 has no unallow-listed LOST/CREATED/shrunk/grown
#      row or misrouted legacy UPC;
#   d. src/crosswalk_checks.py --panel on both new files;
#   e. the regenerated A8 report matches the committed report apart from
#      the evaluation code commit recorded in its header;
#   f. the regenerated RMS diagnostics carry, per (year, upc), the j that the
#      evaluation's new route assigns, the exact key set, and the same dollars
#      as the accepted build (the route changed j, nothing else).
#
# PRE-STEPS (plan section "Code pre-steps"; both done in Run 1):
#   P1 tests/test_rms_market_panel.py::test_recompute_revised_cell_and_name_route
#      re-derives j through src/brand_route.py on a shared-route build
#      (upc_routes carries j_route); the name-route branch stays for the
#      accepted build.
#   P2 src/rms_validate.py section 7 counts j's merge set as j on a
#      shared-route build: DONE in Run 1 (tests/test_rms_validate_merge.py).
#
#   sbatch --dependency=afterok:<rms-d3 job> slurm/crosswalk-checks.slurm

set -uo pipefail
# Bootstrap from a verified local clone before sourcing any repository code.
BOOT_MIRROR=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data
BOOT_REPO=/local/job${SLURM_JOB_ID:?}/mirrors/nielsen-cereal-data
XW_BRANCH_NIELSEN=${XW_BRANCH_NIELSEN:-crosswalk}
[[ ${XW_PIN_NIELSEN:-} =~ ^[0-9a-f]{40}$ ]] || { echo "XW FATAL: full Nielsen pin required" >&2; exit 2; }
[[ $(git -C "$BOOT_MIRROR" rev-parse refs/heads/$XW_BRANCH_NIELSEN) == "$XW_PIN_NIELSEN" ]] \
    || { echo "XW FATAL: durable Nielsen branch is not pinned" >&2; exit 2; }
mkdir -p "$(dirname "$BOOT_REPO")" || exit 2
[[ ! -L $BOOT_REPO ]] || { echo "XW FATAL: local clone is a symlink" >&2; exit 2; }
if [[ ! -e $BOOT_REPO ]]; then
    git clone --quiet --single-branch --branch "$XW_BRANCH_NIELSEN" "$BOOT_MIRROR" "$BOOT_REPO" || exit 2
fi
BOOT_STATUS=$(git -C "$BOOT_REPO" status --porcelain --untracked-files=all) || exit 2
[[ $(git -C "$BOOT_REPO" rev-parse HEAD) == "$XW_PIN_NIELSEN" \
   && -z $BOOT_STATUS ]] \
    || { echo "XW FATAL: local Nielsen clone is dirty or unpinned" >&2; exit 2; }
cd "$BOOT_REPO" || exit 2
source slurm/crosswalk-env.sh
xw_preflight || exit 2
cd "$XW_REPO"
xw_refuse_outputs "$RMS_VINTAGE/rms-validation.org" \
    "$VINTAGE/crosswalk-before-after.org" "$VINTAGE/a8-run1-comparison.org" \
    "$VINTAGE/combined-panel-comparison.org"
NEW_PANEL=$VINTAGE/market_panel.parquet
NEW_RMS=$RMS_VINTAGE/market_panel_rms.parquet
[[ -f $NEW_PANEL && -f $NEW_RMS ]] || xw_die "new panels missing (jobs 2-3)"
export PANEL_CLEAN=$VINTAGE PANEL_RAW=$RAW RMS_OUT=$RMS_VINTAGE HH_CLEAN=$VINTAGE
GATE=0

echo; echo "=== a. make rms-test (new RMS panel) $(date -Is) ==="
make rms-test RMS_PYTHON="$PY" || GATE=1

echo; echo "=== b. crosswalk tests on the new files $(date -Is) ==="
CROSSWALK_PANEL=$NEW_PANEL CROSSWALK_RMS_PANEL=$NEW_RMS \
    "$PY" -m pytest -q -p no:cacheprovider -rs \
        tests/test_brand_route.py tests/test_pinned_brands.py \
        tests/test_crosswalk_criteria.py tests/test_rms_validate_merge.py \
        tests/test_crosswalk_chain_gates.py \
        tests/test_household_tiebreak.py tests/test_household_vintage.py || GATE=1

echo; echo "=== c. rms_validate.py (D4) -> $RMS_VINTAGE/rms-validation.org $(date -Is) ==="
"$PY" src/rms_validate.py --out "$RMS_VINTAGE/rms-validation.org" \
    --d6 "$RMS_VINTAGE/laws/homog_compare.json" --gate-crosswalk || GATE=1
grep -n -A30 -i "lost\|created" "$RMS_VINTAGE/rms-validation.org" | head -60 || true

echo; echo "=== d. A5 on the new panel files (crosswalk_checks --panel) $(date -Is) ==="
"$PY" src/crosswalk_checks.py --panel "$NEW_PANEL" --source hms || GATE=1
"$PY" src/crosswalk_checks.py --panel "$NEW_RMS" --source rms || GATE=1

echo; echo "=== e. A8 (UPC-dollar evaluation + report) into the vintage $(date -Is) ==="
# --rms stays the ACCEPTED dir: the evaluation's RMS old_j is the accepted
# upc_routes' brand_code_uc.  Inputs are unchanged, so the report should match
# the committed docs/crosswalk-before-after.org apart from the recorded
# evaluation commit line.
if env -u PANEL_CLEAN -u RMS_OUT PANEL_RAW="$RAW" \
       "$PY" src/crosswalk_eval.py --raw "$RAW" --rms "$RMS" --out "$VINTAGE/a8-eval" \
   && "$PY" src/crosswalk_report.py --eval "$VINTAGE/a8-eval" \
       --out "$VINTAGE/a8-run1-comparison.org" \
   && "$PY" src/crosswalk_report.py --eval "$VINTAGE/a8-eval" \
       --vintage combined \
       --out "$VINTAGE/crosswalk-before-after.org" \
   && [[ -s $VINTAGE/crosswalk-before-after.org ]]; then
    # The evaluation records the commit of its own run; all report content
    # beyond that metadata line must match the reviewed, committed A8.
    if diff -u <(sed '/^- evaluation:/d' docs/crosswalk-before-after.org) \
               <(sed '/^- evaluation:/d' "$VINTAGE/a8-run1-comparison.org") \
               > "$VINTAGE/a8-report.diff"; then
        echo "A8 matches the committed report"
    else
        echo "A8 differs from the committed report (first 80 lines):"
        head -80 "$VINTAGE/a8-report.diff"
        GATE=1
    fi
else
    echo "A8 evaluation or report generation failed"
    GATE=1
fi

echo; echo "=== f. the regenerated RMS build used the evaluated route (gate) $(date -Is) ==="
# EVAL: the committed Run 1 evaluation (same inputs, same route code), so
# this gate does not depend on the A8 regeneration in step (e).
RMS="$RMS" RMS_VINTAGE="$RMS_VINTAGE" EVAL="$XW1_VINTAGE/run1-eval" "$PY" - <<'PYEOF' || GATE=1
import os
from src.crosswalk_chain_gates import check_rms_route_alignment

n = check_rms_route_alignment(
    f"{os.environ['RMS_VINTAGE']}/diagnostics/upc_routes.parquet",
    f"{os.environ['RMS']}/diagnostics/upc_routes.parquet",
    f"{os.environ['EVAL']}/rms_upc.parquet")
print(f"  exact (year, upc) keys, dollars and evaluated j: {n:,} passed")
PYEOF

if [[ ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
    echo; echo "=== g. accepted/xw1/xw2 built-panel comparison $(date -Is) ==="
    "$PY" src/crosswalk_hhfix_compare.py \
        --accepted "$OLD" --xw1 "$XW1_VINTAGE" --xw2 "$VINTAGE" \
        --rms-xw1 "$XW1_RMS_VINTAGE" --rms-xw2 "$RMS_VINTAGE" \
        --out "$VINTAGE/combined-panel-comparison.org" || GATE=1
    if [[ -s $VINTAGE/combined-panel-comparison.org ]]; then
        cat "$VINTAGE/combined-panel-comparison.org" \
            >> "$VINTAGE/crosswalk-before-after.org"
    fi
fi

xw_accepted_unchanged || exit 2
if [[ $GATE == 0 && ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
    {
        echo
        echo '* Combined vintage panel gates'
        echo
        echo 'The same stage passed the A2/A3/A4/A5/A6 tests, household_size and'
        echo 'deterministic panel_year tests, D4 section 7, both panel-level A5'
        echo 'checks, RMS route alignment, and the accepted-file hash guard.'
        echo 'The UPC-dollar tables above reproduce Run 1; see this job log for'
        echo 'the panel checks and =b2/impact-xw1-to-xw2.org= for source effects.'
        echo
        echo "- Nielsen producer: =$XW_PIN_NIELSEN="
        echo "- CerealDemand producer: =$XW_PIN_CD="
        echo "- Homescan panel sha256: =$(sha256sum "$NEW_PANEL" | cut -d' ' -f1)="
        echo "- RMS panel sha256: =$(sha256sum "$NEW_RMS" | cut -d' ' -f1)="
    } >> "$VINTAGE/crosswalk-before-after.org"
fi
sha256sum "$RMS_VINTAGE/rms-validation.org" "$VINTAGE/crosswalk-before-after.org" \
    2>/dev/null | tee -a "$VINTAGE/SHA256SUMS.new" || GATE=1
echo "=== gate rc=$GATE; done $(date -Is) ==="
exit "$GATE"
