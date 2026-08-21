#!/usr/bin/env bash
# Eight ranks, eight "machines", one box — the replication ring at a width the
# container bench cannot reach, with NCCL present.
#
#   bash $KIT_ROOT/kit/70-eight-ranks.sh run1
#   bash $KIT_ROOT/kit/70-eight-ranks.sh loss
#   bash $KIT_ROOT/kit/70-eight-ranks.sh resume
#   bash $KIT_ROOT/kit/70-eight-ranks.sh report
#
# What this adds over integration/multinode, which runs six containers on gloo:
# the backend is the real one. GPU-82 was invisible for exactly that reason —
# the bench could not vary the backend, and the code is correct on gloo.
#
# What it still cannot do, and must not be reported as doing: the network. Eight
# ranks on one box exchange stores over loopback. That is GPU-83, and GPU-83
# needs two machines.
set -euo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"

VENV="${VENV:-/venv/main}"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"

OUT=$KIT_ROOT/out
WORK=$KIT_ROOT/run-eight
RANKS="${RANKS:-8}"
PARAMS="${PARAMS:-8e8}"
HIDDEN="${HIDDEN:-4096}"
STEPS="${STEPS:-20}"
EVERY="${EVERY:-4}"
DIE_AT="${DIE_AT:-13}"
LOST_RANK="${LOST_RANK:-3}"
mkdir -p "$OUT"

section() { printf '\n\033[1m-- %s --------------------------------\033[0m\n' "$1"; }

start_all() {
    local die_at="$1" tag="$2" pids=() rank
    for rank in $(seq 0 $((RANKS - 1))); do
        local dir="$WORK/rank$rank"
        mkdir -p "$dir"
        cat > "$dir/ravex.yaml" <<YAML
checkpoint_every: $EVERY
backend: moonclip
keep_last: 2
sharded_checkpoints: per_rank
replicate_every: 1
storage:
  type: local
  path: ./checkpoints
log_file: ./ravex.log
log_level: INFO
YAML
        (
            cd "$dir"
            RANK=$rank WORLD_SIZE=$RANKS LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
            GROUP_RANK=$rank CUDA_VISIBLE_DEVICES=$rank \
            MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29801}" \
            python $KIT_ROOT/kit/run_train.py -- \
                --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
                --die-at "$die_at" --measure none \
                > "$OUT/eight.$tag.rank$rank.out" 2>&1
        ) &
        pids+=($!)
    done
    # A rank that SIGKILLs itself takes the collective down with it; the others
    # are expected to end badly and that is the phase working.
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    for rank in $(seq 0 $((RANKS - 1))); do
        cp -f "$WORK/rank$rank/ravex.log" "$OUT/eight.$tag.rank$rank.log" 2>/dev/null || true
    done
}

layout() {
    local rank
    for rank in $(seq 0 $((RANKS - 1))); do
        local dir="$WORK/rank$rank/checkpoints"
        printf '  rank%-2s own: %-22s replicas: %s\n' "$rank" \
            "$(ls -d "$dir"/rank_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" \
            "$(ls -d "$dir"/replica/rank_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
    done
}

case "${1:-report}" in
run1)
    section "$RANKS ranks, $RANKS machines, die at step $DIE_AT"
    rm -rf "$WORK"
    export RAVEX_TIMING_OUT="$OUT/eight.run1.jsonl"
    rm -f "$OUT/eight."*.jsonl
    df -PBG "$KIT_ROOT" | awk 'NR == 2 {print "  free: " $4}'
    start_all "$DIE_AT" run1
    section "what each disk holds"
    layout
    df -PBG "$KIT_ROOT" | awk 'NR == 2 {print "  free: " $4}'
    ;;
loss)
    section "rank $LOST_RANK's machine is replaced: its disk arrives empty"
    rm -rf "$WORK/rank$LOST_RANK/checkpoints"
    mkdir -p "$WORK/rank$LOST_RANK/checkpoints"
    layout
    ;;
resume)
    section "the same command again"
    export RAVEX_TIMING_OUT="$OUT/eight.resume.jsonl"
    start_all 0 resume
    section "what each disk holds"
    layout
    ;;
report)
    section "what the ranks said"
    for tag in run1 resume; do
        ls "$OUT"/eight.$tag.rank*.log >/dev/null 2>&1 || continue
        echo "--- $tag ---"
        grep -hE "This job spans|Replicated step|took one back|rebuilt one from|fetched one back|Resumed at|from scratch|cannot reach|did not complete|failed here|No space" \
            "$OUT"/eight.$tag.rank*.log \
            | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' \
            | sort | uniq -c | sed 's/^/  /'
        echo
    done
    section "timings at $RANKS ranks"
    EVERY="$EVERY" python $KIT_ROOT/kit/report.py "$OUT"/eight.*.jsonl
    ;;
*) echo "usage: $0 {run1|loss|resume|report}" >&2; exit 2 ;;
esac
