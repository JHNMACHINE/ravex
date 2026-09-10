#!/usr/bin/env bash
# Laptop side. What are the boxes doing *right now*.
#
#   bash watch.sh          # the last 15 progress lines from each box
#   bash watch.sh 40       # more of them
#   bash watch.sh -f       # follow, until Ctrl-C
#
# `on-both.sh` hands its output back only when the command ends, so a phase in
# the middle of a long exchange looks exactly like a phase that is wedged. That
# ambiguity cost a rented afternoon once: fifteen minutes of waiting, then an
# `ssh` and a `ps` to discover the run had already failed and torchrun was
# sitting on a 300 s teardown barrier of its own.
#
# The phases append to `$KIT_ROOT/out/phase.log` as they go — `progress()` in
# `remote/lib.sh`, and the heartbeat thread in `remote/outer_run.py`, which says
# what stage it is in and how long it has been there. So a stalled phase is not
# silence: it is the same line repeating with a growing number, which is a
# different thing and reads as one.
#
# It also prints what is running, because "the log stopped" and "the process
# died" are different diagnoses with the same last line.
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/boxes.env" ] || { echo "no boxes.env - run push.sh first" >&2; exit 2; }
# shellcheck disable=SC1091
. "$HERE/boxes.env"

KEY="${KIT_KEY:-$HERE/id_throwaway}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
     -o ConnectTimeout=15 -o BatchMode=yes)

LINES="${1:-15}"
FOLLOW=""
if [ "$LINES" = "-f" ]; then FOLLOW="-f"; LINES=15; fi

look() {
    local host="$1" port="$2" node="$3"
    # `tail -F`, not `-f`: a phase that has not started yet has no file, and a
    # watcher that exits because of that is a watcher nobody trusts.
    "${SSH[@]}" -p "$port" "$host" "
        echo '--- phase now: '\$(cat $KIT_ROOT/out/phase.log.current 2>/dev/null || echo 'none announced')
        echo '--- running: '\$(ps -eo cmd | grep -cE 'outer_[r]un|run_[t]rain|torch[r]un') ' process(es)'
        tail -n $LINES $FOLLOW $KIT_ROOT/out/phase.log 2>/dev/null || echo '(no phase.log yet)'
    " 2>&1 | sed "s/^/[$node] /" &
}

look "$HOST0" "$PORT0" node0
look "$HOST1" "$PORT1" node1
wait
