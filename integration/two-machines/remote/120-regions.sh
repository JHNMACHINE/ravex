#!/usr/bin/env bash
# GPU-143: rounds over regions, on four real machines in two regions.
#
#   bash push.sh '<eu box 0>' '<eu box 1>' '<us box 0>' '<us box 1>'
#   bash addrs.sh <a>@eu <b>@eu <c>@us <d>@us
#   bash on-both.sh 'bash $KIT/120-regions.sh flat'      # every node fetches every node
#   bash fetch.sh regions-flat
#   bash on-both.sh 'bash $KIT/120-regions.sh regions'   # one delegate per region crosses
#   bash fetch.sh regions-regions
#   bash on-both.sh 'bash $KIT/120-regions.sh report' --node 0
#   bash on-both.sh 'bash $KIT/120-regions.sh stop' --node 0
#
# The same run twice, on the same boxes, the same minute: once flat, as every
# outer-loop phase before this one, and once with each node's RAVEX_OUTER_REGION
# set from `addrs.sh`. What the two have to show:
#
#   flat     every node downloads the other three deltas, two of them across
#            the ocean: (N-1) x S per node per round, half of it on the slow
#            link. The round's `network` seconds are those downloads.
#
#   regions  a node downloads its region-mate's delta on the fast link; one
#            delegate per region downloads the other region's aggregate on the
#            slow one, (R-1) x S, and serves the result back. `network` is now
#            only the region's gather, and `between regions` is the rest. The
#            sum is the number to hold against the flat round's `network`.
#
# With two nodes per region the slow link carries one S per delegate instead
# of two per node: the gain is in what crosses, and the round pays for it with
# a hop more on the fast link. Both arms train on noise - the claim here is
# about the round, and `tests/test_dist_regions.py` is where the arithmetic and
# the bit-for-bit agreement are checked.
#
# The rendezvous server is node 0's, started by every arm and stopped by `stop`
# only, as in `110-rendezvous.sh`.
. "$(dirname "$0")/lib.sh"

ARM="${1:-flat}"
NAME=gpu143
DIR="$WORK/$NAME"
RDZV_PORT="${RDZV_PORT:-29400}"
UNTIL="${UNTIL:-5}"
INNER="${INNER:-50}"
RG_PARAMS="${RG_PARAMS:-2.5e7}"
RG_HIDDEN="${RG_HIDDEN:-2048}"
BATCH="${BATCH:-8}"
DEADLINE="${DEADLINE:-300}"
NBOXES="${NBOXES:-2}"

RDZV="${NODE0_ADDR:-$SELF_ADDR}:$RDZV_PORT"
export RAVEX_EXCHANGE_ADDRESS="${RAVEX_EXCHANGE_ADDRESS:-$SELF_ADDR}"
# One run for the four nodes, not four runs of one.
export RAVEX_RUN_ID="${RAVEX_RUN_ID:-$NAME-$ARM}"

stop_server() {
    [ "$NODE_RANK" = "0" ] || return 0
    pkill -f "ravex._cli import [m]ain" || true
}

start_server() {
    [ "$NODE_RANK" = "0" ] || return 0
    stop_server
    sleep 1
    setsid nohup python -c 'import sys; from ravex._cli import main; sys.exit(main(sys.argv[1:]))' \
        rendezvous --host 0.0.0.0 --port "$RDZV_PORT" \
        > "$OUT/$NAME.$ARM.server.out" 2>&1 < /dev/null &
    progress "rendezvous server on $RDZV"
}

trainer() {
    local name="node$NODE_RANK"
    phase_begin "gpu143 $ARM $name: region ${REGION:-none}, until round $UNTIL, params $RG_PARAMS"
    # Node 0's server needs a moment before the others dial it.
    [ "$NODE_RANK" = "0" ] || sleep 3
    python "$KIT_ROOT/kit/outer_run.py" \
        --rendezvous "$RDZV" --job "$NAME-$ARM" --min-nodes "$NBOXES" \
        --node-index "$NODE_RANK" --name "$name" \
        --until-round "$UNTIL" --inner "$INNER" \
        --params "$RG_PARAMS" --hidden "$RG_HIDDEN" --batch "$BATCH" \
        --deadline "$DEADLINE" --root "$DIR/$ARM/$name" --out "$OUT" "$@" \
        2>&1 | tee "$OUT/$NAME.$ARM.$name.out" || true
    [ -e "$OUT/rounds.$name.json" ] && mv -f "$OUT/rounds.$name.json" "$OUT/$NAME.$ARM.rounds.$name.json"
    phase_end "gpu143 $ARM $name"
}

case "$ARM" in

flat)
    section "the flat round: every node fetches every other node's delta"
    disk_guard
    rm -rf "${DIR:?}/flat"
    start_server
    trainer
    ;;

regions)
    [ -n "${REGION:-}" ] || { echo "no REGION on this box: give addrs.sh <address>@<region>" >&2; exit 2; }
    section "the round over regions: this node in $REGION"
    disk_guard
    rm -rf "${DIR:?}/regions"
    start_server
    trainer --region "$REGION"
    ;;

stop)
    stop_server
    ;;

report)
    section "what the rounds said, on node $NODE_RANK"
    python - "$OUT" <<'PY'
import glob, json, os, sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "gpu143.*.rounds.*.json"))):
    rounds = json.load(open(path))["rounds"]
    print("\n%s  (%d round(s))" % (os.path.basename(path), len(rounds)))
    print("  round nodes   total  network (waiting)  between (waiting)  delegate")
    for entry in rounds:
        print("  %5d %5d %7.2f %8.2f %9.2f %8s %9s  %8s" % (
            entry["round"], entry["nodes"], entry["total_seconds"],
            entry["gather_seconds"], entry["gather_wait_seconds"],
            "%.2f" % entry["between_seconds"] if "between_seconds" in entry else "-",
            "%.2f" % entry["between_wait_seconds"] if "between_wait_seconds" in entry else "-",
            entry.get("delegate", "-")))
PY
    ;;

*)
    echo "usage: $0 {flat|regions|report|stop}" >&2; exit 2 ;;
esac
