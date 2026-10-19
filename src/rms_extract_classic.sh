#!/usr/bin/env bash
# Stream module-1344 movement + Annual_Files out of the classic (2006-2020)
# NielsenIQ RMS monoliths, one gzip pass per archive, without unpacking the
# whole archive.  A full member listing is captured from the same pass.
#
# Usage:  src/rms_extract_classic.sh [YEAR ...]      (default: 2006..2020)
#   SRC  (env) archive dir   default Data/Incoming/2006-2020_Scanner_Data
#   OUT  (env) output root   default Data/NielsenScanner/extracted/classic
#   JOBS (env) parallelism   default 15
#
# Layout (verified 2026-10-19 on SCANNER_DATA_2006.tgz):
#   nielsen_extracts/RMS/<yr>/Movement_Files/<group>_<yr>/<module>_<yr>.tsv
# Member order is NOT by group, so every archive costs one full pass.
#
# Output per year:  $OUT/<yr>/nielsen_extracts/RMS/<yr>/...  (1344 + Annual_Files)
#                   $OUT/logs/listing_<yr>.txt, extract_<yr>.log, <yr>.done
# Permissions: group-readable, o-rwx (Nielsen data terms).
# Idempotent: a year with <yr>.done is skipped; delete the marker to redo it.
set -euo pipefail
umask 027

DATA=/global/scratch/fsa/fc_jevons/ligon/Data
SRC=${SRC:-$DATA/Incoming/2006-2020_Scanner_Data}
OUT=${OUT:-$DATA/NielsenScanner/extracted/classic}
JOBS=${JOBS:-15}
mkdir -p "$OUT/logs"

one_year() {
    local y=$1 src=$SRC/SCANNER_DATA_$1.tgz dst=$OUT/$1 log=$OUT/logs
    if [[ -e $log/$y.done ]]; then echo "$y: done already"; return 0; fi
    mkdir -p "$dst"
    echo "$y: start $(date -Is)" > "$log/extract_$y.log"
    set +e
    pigz -dc "$src" \
        | tee >(tar -tv > "$log/listing_$y.txt") \
        | tar -x -C "$dst" --wildcards '*/1344_*.tsv' '*/Annual_Files/*' \
          2>> "$log/extract_$y.log"
    local rc=("${PIPESTATUS[@]}")
    set -e
    echo "$y: end $(date -Is) pipestatus=${rc[*]}" >> "$log/extract_$y.log"
    chmod -R g+rX,o-rwx "$dst"
    if [[ ${rc[0]} -eq 0 && ${rc[2]} -eq 0 ]]; then
        find "$dst" -type f -printf '%s\t%P\n' | sort -k2 > "$log/$y.done"
        echo "$y: ok"
    else
        echo "$y: FAILED pipestatus=${rc[*]}" | tee -a "$log/extract_$y.log"
        return 1
    fi
}
export -f one_year
export SRC OUT

years=("$@")
[[ ${#years[@]} -gt 0 ]] || years=($(seq 2006 2020))
printf '%s\n' "${years[@]}" | xargs -P "$JOBS" -I{} bash -c 'one_year {}'
