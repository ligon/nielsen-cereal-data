# Shared settings for the charter-crosswalk A7/B3/B5 jobs (slurm/crosswalk-*.slurm).
# Plan and order: docs/crosswalk-slurm-plan.org.  Sourced by each job; never
# run on its own.  Nothing here writes data.
#
# Every path is absolute.  Slurm jobs clone pinned code from the durable
# mirrors to their own /local/job<ID>/mirrors; data remain on shared storage.
# New data files go only under $VINTAGE (Homescan) and $RMS_VINTAGE (RMS),
# both group-only (umask 027).

umask 027
export PYTHONUNBUFFERED=1

# The work checkouts are used on the login node.  Jobs clone the named
# branch from the durable on-cluster mirrors, then replace XW_REPO/XW_CD.
XW_REPO=/global/scratch/fsa/fc_jevons/ligon/mirrors/crosswalk-work/nielsen-cereal-data
XW_CD=/global/scratch/fsa/fc_jevons/ligon/mirrors/crosswalk-work/CerealDemand
XW_NIELSEN_MIRROR=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data
XW_CD_MIRROR=/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand
XW_BRANCH_NIELSEN=${XW_BRANCH_NIELSEN:-crosswalk}
XW_BRANCH_CD=${XW_BRANCH_CD:-crosswalk}
# The venv the crosswalk code was tested with (pandas 3.0.3, pyarrow 24); the
# accepted RMS panel was also built with it (src/rms_build.sh PYTHON default).
PY=/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python

# Data (read-only): Homescan raw, the accepted Homescan clean dir, the RMS root.
RAW=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/raw
OLD=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean
RMS=/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms
KAGG_VAR=/global/scratch/fsa/fc_jevons/ligon/mirrors/K-Aggregators/var
PANEL_LAW_STORE=/global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/var/laws

# The authorized vintage (EL 2026-10-19; docs/crosswalk-rule.org).
# VINTAGE sits beside data/clean on purpose: rms_market_panel.py and
# tests/test_rms_market_panel.py take PANEL_RAW = PANEL_CLEAN/../raw, and
# CerealDemand's read_cpi takes <datadir>/../../config.
VINTAGE=${XW_VINTAGE:-/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19}
RMS_VINTAGE=${XW_RMS_VINTAGE:-$RMS/crosswalk-2026-10-19}
XW1_VINTAGE=${XW1_VINTAGE:-/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19}
XW1_RMS_VINTAGE=${XW1_RMS_VINTAGE:-$RMS/crosswalk-2026-10-19}
XW_LOGS=$VINTAGE/logs
ACCEPTED_SUMS=$VINTAGE/SHA256SUMS.accepted

# The files that must never change (charter A7: old files kept).
xw_accepted_files() {
    ls "$OLD"/*.parquet
    echo "$RMS/market_panel_rms.parquet"
    ls "$RMS"/diagnostics/*.parquet
}

xw_xw1_files() {
    ls "$XW1_VINTAGE"/*.parquet
    echo "$XW1_RMS_VINTAGE/market_panel_rms.parquet"
    ls "$XW1_RMS_VINTAGE"/diagnostics/*.parquet
}

xw_die() { echo "XW FATAL: $*" >&2; exit 2; }

xw_refuse_outputs() {
    local path
    for path in "$@"; do
        [[ ! -e $path ]] || xw_die "output already exists: $path; review stage before retry"
    done
}

# A job may rerun a stage inside one allocation; reuse only a local clone
# whose origin is still the intended durable mirror.  Never clean or reset it.
xw_clone_pinned() {
    local mirror=$1 dest=$2 pin=$3 branch=$4 head origin
    head=$(git -C "$mirror" rev-parse --verify "refs/heads/$branch") \
        || xw_die "cannot read $branch in $mirror"
    [[ $head == "$pin" ]] \
        || xw_die "$mirror $branch is $head, not the reviewed pin $pin"
    [[ ! -L $dest ]] || xw_die "$dest is a symlink, not a local clone"
    if [[ ! -e $dest ]]; then
        git clone --quiet --single-branch --branch "$branch" "$mirror" "$dest" \
            || xw_die "could not clone $mirror into $dest"
    fi
    origin=$(git -C "$dest" remote get-url origin) \
        || xw_die "$dest is not a clone with an origin"
    [[ $(realpath -m "$origin") == $(realpath -m "$mirror") ]] \
        || xw_die "$dest origin $origin is not $mirror"
}

xw_stage_job_code() {
    local job_root=$1
    mkdir -p "$job_root/mirrors" "$job_root/tmp" \
        || xw_die "cannot create local work directories under $job_root"
    export TMPDIR="$job_root/tmp"
    xw_clone_pinned "$XW_NIELSEN_MIRROR" \
        "$job_root/mirrors/nielsen-cereal-data" "$XW_PIN_NIELSEN" "$XW_BRANCH_NIELSEN"
    xw_clone_pinned "$XW_CD_MIRROR" \
        "$job_root/mirrors/CerealDemand" "$XW_PIN_CD" "$XW_BRANCH_CD"
    XW_REPO=$job_root/mirrors/nielsen-cereal-data
    XW_CD=$job_root/mirrors/CerealDemand
    echo "local code: $job_root/mirrors; TMPDIR=$TMPDIR"
}

# Preflight for every job: right checkout, clean tree, vintage seeded, and the
# output dirs are not the accepted ones.
xw_preflight() {
    local repo branch
    [[ ${XW_PIN_NIELSEN:-} =~ ^[0-9a-f]{40}$ && ${XW_PIN_CD:-} =~ ^[0-9a-f]{40}$ ]] \
        || xw_die "set full XW_PIN_NIELSEN and XW_PIN_CD reviewed commits"
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        [[ $SLURM_JOB_ID =~ ^[0-9]+$ ]] || xw_die "invalid SLURM_JOB_ID=$SLURM_JOB_ID"
        xw_stage_job_code "/local/job${SLURM_JOB_ID}"
    fi
    for repo in "$XW_REPO" "$XW_CD"; do
        branch=$XW_BRANCH_NIELSEN
        [[ $repo == "$XW_CD" ]] && branch=$XW_BRANCH_CD
        [[ $(git -C "$repo" rev-parse --abbrev-ref HEAD) == "$branch" ]] \
            || xw_die "$repo is not on branch $branch"
        [[ -z $(git -C "$repo" status --porcelain --untracked-files=all) ]] \
            || xw_die "$repo has uncommitted changes"
        echo "code: $repo @ $(git -C "$repo" rev-parse HEAD)"
    done
    [[ $(git -C "$XW_REPO" rev-parse HEAD) == "$XW_PIN_NIELSEN" ]] \
        || xw_die "nielsen-cereal-data HEAD is not XW_PIN_NIELSEN=$XW_PIN_NIELSEN"
    [[ $(git -C "$XW_CD" rev-parse HEAD) == "$XW_PIN_CD" ]] \
        || xw_die "CerealDemand HEAD is not XW_PIN_CD=$XW_PIN_CD"
    [[ $(realpath -m "$VINTAGE") != $(realpath -m "$OLD") ]] || xw_die "VINTAGE is data/clean"
    [[ $(realpath -m "$RMS_VINTAGE") != $(realpath -m "$RMS") ]] || xw_die "RMS_VINTAGE is the accepted RMS dir"
    if [[ ${XW_REBUILD_DEMOG:-0} == 1 ]]; then
        [[ $(realpath -m "$VINTAGE") != $(realpath -m "$XW1_VINTAGE") ]] \
            || xw_die "combined VINTAGE is xw1"
        [[ $(realpath -m "$RMS_VINTAGE") != $(realpath -m "$XW1_RMS_VINTAGE") ]] \
            || xw_die "combined RMS_VINTAGE is xw1"
        [[ -f $VINTAGE/SHA256SUMS.xw1 ]] \
            || xw_die "combined vintage has no xw1 hash guard"
    fi
    local f
    for f in nutrition.parquet nutrition_2021_brands.parquet; do
        [[ -f $VINTAGE/$f ]] || xw_die "vintage not seeded: $VINTAGE/$f (plan step 0)"
    done
    if [[ ${1:-} != seed ]]; then
        for f in demographics_household.parquet demographics_market.parquet; do
            [[ -f $VINTAGE/$f ]] || xw_die "vintage not built: $VINTAGE/$f (households stage)"
        done
    fi
    [[ -f $ACCEPTED_SUMS ]] || xw_die "no $ACCEPTED_SUMS (plan step 0)"
    echo "node $(hostname); $(date -Is); VINTAGE=$VINTAGE RMS_VINTAGE=$RMS_VINTAGE"
}

# The accepted files are byte-identical to step 0's record.  Every job ends here.
xw_accepted_unchanged() {
    sha256sum --quiet -c "$ACCEPTED_SUMS" \
        || xw_die "an accepted file changed (see sha256sum output above)"
    echo "accepted files unchanged ($(wc -l < "$ACCEPTED_SUMS") files)"
    if [[ -f $VINTAGE/SHA256SUMS.xw1 ]]; then
        sha256sum --quiet -c "$VINTAGE/SHA256SUMS.xw1" \
            || xw_die "an xw1 reference file changed"
        echo "xw1 files unchanged ($(wc -l < "$VINTAGE/SHA256SUMS.xw1") files)"
    fi
}
