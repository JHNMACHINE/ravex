#!/usr/bin/env bash
# Run the same command on both boxes at once, and wait for both.
#
#   bash on-both.sh 'bash $KIT/20-correctness.sh run1'
#
# $KIT is exported on the far side and expands there, so the same line works
# whether the kit landed in /root (vast.ai) or /workspace (RunPod). Quote it
# single, or the laptop's shell eats it first.
#
# At once, not one after the other. Every phase here forms a process group
# across the two machines, so a serial version deadlocks on rendezvous — and
# on a rented box a machine waiting for a command is a machine being billed.
set -uo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd)"

# The throwaway key. IdentitiesOnly so ssh offers this one and nothing from the
# agent: a box that rejects it should say so now, not fall back to a personal
# key and hide the fact that the disposable one was never installed.
KEY="${KIT_KEY:-$HERE/id_throwaway}"
[ -f "$KEY" ] || { echo "missing $KEY - generate one, see README.md" >&2; exit 2; }
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SCP=(scp -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
. "$HERE/boxes.env"
KIT_ROOT="${KIT_ROOT:-/root}"

[ $# -ge 1 ] || { echo "usage: $0 '<command>' [--node 0|1]" >&2; exit 2; }
CMD="$1"; shift
ONLY=""
[ "${1:-}" = "--node" ] && ONLY="$2"

run() {
    local host="$1" port="$2" node="$3"
    "${SSH[@]}" -p "$port" "$host" \
        ". $KIT_ROOT/kit/box.env; export KIT=$KIT_ROOT/kit; cd $KIT_ROOT; $CMD" \
        2>&1 | sed "s/^/[node$node] /"
    echo "[node$node] exit ${PIPESTATUS[0]}"
}

pids=()
[ "$ONLY" = "1" ] || { run "$HOST0" "$PORT0" 0 & pids+=($!); }
[ "$ONLY" = "0" ] || { run "$HOST1" "$PORT1" 1 & pids+=($!); }
for pid in "${pids[@]}"; do wait "$pid"; done
