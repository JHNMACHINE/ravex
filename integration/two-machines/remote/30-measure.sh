#!/usr/bin/env bash
# GPU-81, measures 1 and 2: what a full store costs to move, and what merge_now
# costs on the training thread.
#
#   bash $KIT_ROOT/kit/30-measure.sh control      # both boxes, replication off
#   bash $KIT_ROOT/kit/30-measure.sh replicated   # both boxes, replication on
#   bash $KIT_ROOT/kit/30-measure.sh report       # both boxes
#
# The two runs are the A/B, and they go back to back on purpose: on a rented
# box the write path degrades over the session, so an arm compared against a
# baseline taken an hour earlier is comparing two machines. The workspace is
# wiped between them for the same reason — a full first checkpoint each time,
# and no chance of filling the disk into Ravex's silent self-disable.
. "$(dirname "$0")/lib.sh"

PHASE="${1:-report}"
STEPS="${STEPS:-16}"
EVERY="${EVERY:-2}"
export EVERY   # report.py turns a step time into an interval with it

run_arm() {
    local name="$1" replicate="$2"
    section "$name (checkpoint_every=$EVERY, replicate_every=$replicate)"
    disk_guard
    local dir
    dir="$(fresh_workspace "$name" "$EVERY" "$replicate")"
    export RAVEX_TIMING_OUT="$OUT/timing.$name.jsonl"
    rm -f "$OUT/timing.$name."*.jsonl
    cd "$dir"
    launch $KIT_ROOT/kit/run_train.py -- \
        --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
        --trace "$dir/trace.jsonl" --measure none \
        2>&1 | tee "$OUT/measure.$name.node$NODE_RANK.out"
    cp -f "$dir/ravex.log" "$OUT/measure.$name.node$NODE_RANK.log" 2>/dev/null || true
    # The store the copy is made of, so a rate has a numerator that was not
    # taken on trust.
    du -sh "$dir"/checkpoints/rank_* "$dir"/checkpoints/replica/rank_* 2>/dev/null \
        | sed 's/^/  /'
    # Cleared here rather than at the start of the next arm: the disk has to be
    # in the same state when the second arm begins as when the first did.
    rm -rf "$dir/checkpoints"
}

case "$PHASE" in
control)     run_arm control 0 ;;
replicated)  run_arm replicated 1 ;;
report)
    section "the numbers, node $NODE_RANK"
    python $KIT_ROOT/kit/report.py "$OUT"/timing.*.jsonl
    section "what Ravex said about the cadence"
    grep -hE "handed off|waited .* for the previous one|fraction|Replicated step" \
        "$OUT"/measure.*.node$NODE_RANK.log 2>/dev/null \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' | sed 's/^/  /' \
        | head -40
    ;;
*) echo "usage: $0 {control|replicated|report}" >&2; exit 2 ;;
esac
