#!/usr/bin/env bash
# GPU-83: does the replication work across a real network between two real
# machines. Correctness, not measurement — and it goes first, because if it
# does not work there is nothing to measure.
#
# Four steps. Run each on BOTH boxes at the same time except `loss`, which is
# the machine that gets lost:
#
#   bash $KIT_ROOT/kit/20-correctness.sh run1     # both boxes
#   bash $KIT_ROOT/kit/20-correctness.sh loss     # both (only node 1 wipes)
#   bash $KIT_ROOT/kit/20-correctness.sh resume   # both boxes
#   bash $KIT_ROOT/kit/20-correctness.sh report   # both boxes
#
# Between run1 and resume nothing is rebuilt: the workspace is the same one,
# which is the whole proposition — the same command, after a machine died.
. "$(dirname "$0")/lib.sh"

PHASE="${1:-report}"
NAME=gpu83
DIR="$WORK/$NAME"
STEPS="${STEPS:-24}"
EVERY="${EVERY:-4}"
REPLICATE="${REPLICATE:-1}"
DIE_AT="${DIE_AT:-13}"

save_logs() {
    local tag="$1"
    cp -f "$DIR/ravex.log" "$OUT/gpu83.$tag.node$NODE_RANK.log" 2>/dev/null || true
    cp -f "$DIR/trace.jsonl" "$OUT/gpu83.$tag.node$NODE_RANK.jsonl" 2>/dev/null || true
    : > "$DIR/ravex.log"   # so the next phase's log is only the next phase
}

case "$PHASE" in

run1)
    section "run 1: train, replicate, then go the way a preempted box goes"
    disk_guard
    DIR="$(fresh_workspace "$NAME" "$EVERY" "$REPLICATE")"
    cd "$DIR"
    # The SIGKILL is expected; torchrun exits non-zero and that is the phase
    # succeeding, not failing.
    launch $KIT_ROOT/kit/run_train.py -- \
        --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
        --die-at "$DIE_AT" --trace "$DIR/trace.jsonl" --measure none \
        2>&1 | tee "$OUT/gpu83.run1.node$NODE_RANK.out" || true
    section "what this machine holds now"
    layout "$DIR"
    save_logs run1
    ;;

loss)
    section "losing a machine"
    if [ "$NODE_RANK" = "1" ]; then
        echo "  wiping $DIR/checkpoints entirely - a replaced machine arrives empty"
        rm -rf "$DIR/checkpoints"
        mkdir -p "$DIR/checkpoints"
    else
        echo "  node 0 keeps everything, including the copy it holds of rank 1"
    fi
    layout "$DIR"
    ;;

resume)
    section "the same command again"
    disk_guard
    cd "$DIR"
    launch $KIT_ROOT/kit/run_train.py -- \
        --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
        --trace "$DIR/trace.jsonl" --measure none \
        2>&1 | tee "$OUT/gpu83.resume.node$NODE_RANK.out" || true
    section "what this machine holds now"
    layout "$DIR"
    save_logs resume
    ;;

report)
    section "what each phase said, on node $NODE_RANK"
    for tag in run1 resume; do
        log="$OUT/gpu83.$tag.node$NODE_RANK.log"
        [ -f "$log" ] || continue
        echo "--- $tag ---"
        grep -hE "This job spans|Checkpoint storage|Replicated step|took one back|rebuilt one from|fetched one back|Resumed at|from scratch|cannot reach|did not complete|failed here" \
            "$log" | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' | sed 's/^/  /'
        echo
    done
    section "anything that would make the above a lie"
    grep -ciE "no space|traceback" "$OUT"/gpu83.*.node$NODE_RANK.* 2>/dev/null \
        | grep -v ':0$' || echo "  no 'no space', no tracebacks"
    ;;

*)
    echo "usage: $0 {run1|loss|resume|report}" >&2; exit 2 ;;
esac
