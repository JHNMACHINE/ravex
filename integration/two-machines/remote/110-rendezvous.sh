#!/usr/bin/env bash
# GPU-129: the outer loop with Ravex's own rendezvous, on two real machines.
#
#   bash on-both.sh 'bash $KIT/110-rendezvous.sh base'    # no torchrun anywhere
#   bash fetch.sh rdzv-base
#   bash on-both.sh 'bash $KIT/110-rendezvous.sh join'    # a third node arrives late
#   bash fetch.sh rdzv-join
#   bash on-both.sh 'bash $KIT/110-rendezvous.sh kill0'   # node 0's trainer dies
#   bash fetch.sh rdzv-kill0
#   bash on-both.sh 'bash $KIT/110-rendezvous.sh stop'    # the server, when done
#
# Every other outer-loop phase launches under torchrun, whose store lives in the
# agent on node 0 and is read on every fetch. These launch plain `python`, and
# the store is a `ravex rendezvous` in a process of its own on node 0 that no
# arm ever kills. The arms, and what each has to show:
#
#   base    the same rounds as 100-outer.sh base, with nothing from torch's
#           process group. The round split should match that arm's: the
#           rendezvous is on the path of a join, not of a round.
#
#   join    node 1 starts a second trainer JOIN_AFTER seconds in. It takes
#           number 2, goes through `membership.join`, and every node's rounds
#           from its admission round say "over 3 node(s)". What loopback never
#           showed: the joiner downloads parameters and momentum over the real
#           link, inside the round it enters - its `gather_wait_seconds` is the
#           number to look at.
#
#   kill0   node 0's trainer dies at DIE_AT_ROUND. Under torchrun that box took
#           the store with it; here the server outlives its trainer, and node 1
#           has to keep closing rounds, over one node.
#
# The server is started at the beginning of every arm and deliberately never
# stopped at the end of one: in `kill0` node 0's script finishes as soon as its
# trainer dies, and stopping the server then would kill exactly what that arm
# is testing. `stop` is its own arm.
. "$(dirname "$0")/lib.sh"

ARM="${1:-base}"
NAME=gpu129
DIR="$WORK/$NAME"
RDZV_PORT="${RDZV_PORT:-29400}"
UNTIL="${UNTIL:-8}"
INNER="${INNER:-50}"
RDZV_PARAMS="${RDZV_PARAMS:-5e7}"
RDZV_HIDDEN="${RDZV_HIDDEN:-2048}"
BATCH="${BATCH:-8}"
DEADLINE="${DEADLINE:-300}"
JOIN_AFTER="${JOIN_AFTER:-45}"
DIE_AT_ROUND="${DIE_AT_ROUND:-3}"

if [ "$NODE_RANK" = "0" ]; then RDZV_HOST="$SELF_ADDR"; else RDZV_HOST="$PEER_ADDR"; fi
RDZV="$RDZV_HOST:$RDZV_PORT"
export RAVEX_EXCHANGE_ADDRESS="${RAVEX_EXCHANGE_ADDRESS:-$SELF_ADDR}"

stop_server() {
    [ "$NODE_RANK" = "0" ] || return 0
    # The bracket keeps pkill from matching its own shell, which carries the
    # pattern in its command line.
    pkill -f "ravex._cli import [m]ain" || true
}

start_server() {
    [ "$NODE_RANK" = "0" ] || return 0
    stop_server
    sleep 1
    # setsid and every stream redirected, or the ssh that on-both.sh opened
    # waits on the server's stdout and the arm never returns.
    setsid nohup python -c 'import sys; from ravex._cli import main; sys.exit(main(sys.argv[1:]))' \
        rendezvous --host 0.0.0.0 --port "$RDZV_PORT" \
        > "$OUT/$NAME.$ARM.server.out" 2>&1 < /dev/null &
    progress "rendezvous server on $RDZV"
}

trainer() {
    local name="$1"; shift
    phase_begin "gpu129 $ARM $name: until round $UNTIL, $INNER steps, params $RDZV_PARAMS"
    python "$KIT_ROOT/kit/outer_run.py" \
        --rendezvous "$RDZV" --job "$NAME-$ARM" --min-nodes 2 \
        --node-index "$NODE_RANK" --name "$name" \
        --until-round "$UNTIL" --inner "$INNER" \
        --params "$RDZV_PARAMS" --hidden "$RDZV_HIDDEN" --batch "$BATCH" \
        --deadline "$DEADLINE" --root "$DIR/$ARM/$name" --out "$OUT" "$@" \
        2>&1 | tee "$OUT/$NAME.$ARM.$name.out" || true
    [ -e "$OUT/rounds.$name.json" ] && mv -f "$OUT/rounds.$name.json" "$OUT/$NAME.$ARM.rounds.$name.json"
    phase_end "gpu129 $ARM $name"
}

case "$ARM" in

base)
    section "the outer loop with no torchrun: server on node 0, one trainer per box"
    disk_guard
    rm -rf "${DIR:?}/base"
    start_server
    trainer "node$NODE_RANK"
    ;;

join)
    section "a third node joins the run in progress, $JOIN_AFTER s in"
    disk_guard
    rm -rf "${DIR:?}/join"
    start_server
    if [ "$NODE_RANK" = "1" ]; then
        trainer node1 &
        first=$!
        sleep "$JOIN_AFTER"
        trainer node2
        wait "$first" || true
    else
        trainer node0
    fi
    ;;

kill0)
    section "node 0's trainer dies at round $DIE_AT_ROUND; the server stays"
    disk_guard
    rm -rf "${DIR:?}/kill0"
    start_server
    if [ "$NODE_RANK" = "0" ]; then
        trainer node0 --die-at-round "$DIE_AT_ROUND"
    else
        trainer node1
    fi
    ;;

learn)
    # Every other arm trains on noise and can only say that rounds close. This
    # one has something to learn (outer_run.py --task teacher): the held-out
    # loss has to fall round after round, and after the same round both boxes
    # have to print the same parameter hash — one model on two continents.
    section "a task with something to learn: held-out loss and parameter hash per round"
    disk_guard
    rm -rf "${DIR:?}/learn"
    start_server
    RDZV_PARAMS="${LEARN_PARAMS:-2e7}" RDZV_HIDDEN="${LEARN_HIDDEN:-2048}" \
        UNTIL="${LEARN_UNTIL:-10}" \
        trainer "node$NODE_RANK" --task teacher --lr "${LEARN_LR:-1e-3}"
    ;;

stop)
    stop_server
    ;;

report)
    section "what the rounds said on node $NODE_RANK"
    python - "$OUT" <<'PY'
import glob, json, os, sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "gpu129.*.rounds.*.json"))):
    rounds = json.load(open(path))["rounds"]
    print("\n%s  (%d round(s))" % (os.path.basename(path), len(rounds)))
    print("  round nodes   total  network (waiting)")
    for entry in rounds:
        print("  %5d %5d %7.2f %8.2f %9.2f" % (
            entry["round"], entry["nodes"], entry["total_seconds"],
            entry["gather_seconds"], entry["gather_wait_seconds"]))
PY
    ;;

*)
    echo "usage: $0 {base|join|kill0|learn|stop|report}" >&2; exit 2 ;;
esac
