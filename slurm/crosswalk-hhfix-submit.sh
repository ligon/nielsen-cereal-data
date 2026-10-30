#!/usr/bin/env bash
# Submit the NEW combined vintage after the two producer branches are reviewed.
# Usage: XW_PIN_CD=<full CerealDemand SHA> [XW_B4_AFTEROK=<jobid>] \
#          bash slurm/crosswalk-hhfix-submit.sh plan|submit
# plan is read-only.  submit seeds the new directories and queues five jobs.
set -euo pipefail

mode=${1:-plan}
[[ $mode == plan || $mode == submit ]] || { echo "usage: $0 plan|submit" >&2; exit 2; }
cd "$(dirname "$0")/.."
source slurm/crosswalk-env.sh

export XW_BRANCH_NIELSEN=crosswalk-hhfix-integration
export XW_BRANCH_CD=crosswalk-current-integration
export XW_VINTAGE=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-hhfix-2026-10-23
export XW_RMS_VINTAGE=/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/crosswalk-hhfix-2026-10-23
export XW_REBUILD_DEMOG=1
export XW1_VINTAGE=/global/scratch/fsa/fc_jevons/ligon/mirrors/nielsen-cereal-data/data/clean-crosswalk-2026-10-19
export XW1_RMS_VINTAGE=/global/scratch/fsa/fc_jevons/ligon/Data/NielsenScanner/rms/crosswalk-2026-10-19
export XW_PIN_NIELSEN=$(git rev-parse HEAD)
[[ ${XW_PIN_CD:-} =~ ^[0-9a-f]{40}$ ]] || xw_die "set full XW_PIN_CD"
if [[ -n ${XW_B4_AFTEROK:-} ]]; then
    [[ $XW_B4_AFTEROK =~ ^[0-9]+$ ]] || xw_die "XW_B4_AFTEROK must be a numeric job ID"
fi
[[ $(git rev-parse --abbrev-ref HEAD) == "$XW_BRANCH_NIELSEN" ]] \
    || xw_die "run from $XW_BRANCH_NIELSEN"
[[ -z $(git status --porcelain --untracked-files=all) ]] \
    || xw_die "Nielsen integration checkout is dirty"
[[ $(git -C "$XW_NIELSEN_MIRROR" rev-parse "refs/heads/$XW_BRANCH_NIELSEN") == "$XW_PIN_NIELSEN" ]] \
    || xw_die "durable Nielsen branch is not at the reviewed pin"
[[ $(git -C "$XW_CD_MIRROR" rev-parse "refs/heads/$XW_BRANCH_CD") == "$XW_PIN_CD" ]] \
    || xw_die "durable CerealDemand branch is not at the reviewed pin"
[[ -f $XW1_VINTAGE/run1-eval/rms_upc.parquet ]] \
    || xw_die "the frozen xw1 Run 1 route evaluation is missing"
[[ ! -e $XW_VINTAGE && ! -e $XW_RMS_VINTAGE ]] \
    || xw_die "new vintage already exists; inspect before stage-specific retry"

echo "Nielsen producer: $XW_PIN_NIELSEN ($XW_BRANCH_NIELSEN)"
echo "CerealDemand producer: $XW_PIN_CD ($XW_BRANCH_CD)"
echo "Homescan output: $XW_VINTAGE"
echo "RMS output: $XW_RMS_VINTAGE"
echo "Chain: demographics + households -> markets -> RMS d3 -> checks/A8 -> B3/B2/B5"
echo "Slurm: co_carleton / savio4_htc / carleton_htc4_normal"
echo "Resources: households 4 CPU/48G; markets 4 CPU/48G; RMS d3 8 CPU/48G; checks 2 CPU/16G; downstream 4 CPU/48G"
echo "Downstream external afterok: ${XW_B4_AFTEROK:-none} (also waits for checks)"
[[ $mode == submit ]] || exit 0

# Step 0 copies only nutrition; demographics are rebuilt in the first job.
bash slurm/crosswalk-step0-seed.sh
logs=$XW_VINTAGE/logs
scripts_dir=$(mktemp -d "${TMPDIR:-/tmp}/xw2-scripts.XXXXXXXX")
names=(crosswalk-households.slurm crosswalk-markets.slurm \
       crosswalk-rms-d3.slurm crosswalk-checks.slurm \
       crosswalk-hhfix-downstream.slurm)
cpus=(4 4 8 2 4)
memory=(48G 48G 48G 16G 48G)
for name in "${names[@]}"; do
    git show "$XW_PIN_NIELSEN:slurm/$name" > "$scripts_dir/$name"
done

prev=
for i in "${!names[@]}"; do
    name=${names[i]}
    options=(--parsable --export=ALL \
             --account=co_carleton --partition=savio4_htc \
             --qos=carleton_htc4_normal \
             --cpus-per-task="${cpus[i]}" --mem="${memory[i]}" \
             --output="$logs/%x-%j.out" --error="$logs/%x-%j.err")
    if [[ -n $prev ]]; then
        dependency="afterok:$prev"
        if [[ $name == crosswalk-hhfix-downstream.slurm && -n ${XW_B4_AFTEROK:-} ]]; then
            dependency+=":$XW_B4_AFTEROK"
        fi
        options+=(--dependency="$dependency")
    fi
    job=$(sbatch "${options[@]}" "$scripts_dir/$name")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || xw_die "sbatch returned invalid job ID: $job"
    echo "$name: job $job; dependency ${dependency:-none}; ${cpus[i]} CPU, ${memory[i]}"
    prev=$job
done
echo "Submitted pinned chain, final job $prev; scripts staged at $scripts_dir"
