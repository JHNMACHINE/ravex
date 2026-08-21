#!/usr/bin/env bash
# Measure 2 of GPU-81 — the cost of merge_now() on the training thread — which
# is the one measure that does not need two machines.
#
#   bash $KIT_ROOT/kit/50-single-box.sh control
#   bash $KIT_ROOT/kit/50-single-box.sh replicated
#   bash $KIT_ROOT/kit/50-single-box.sh report
#
# The GPU-82 recipe: two ranks on this box with WORLD_SIZE=2 and
# LOCAL_WORLD_SIZE=1, so Ravex counts two machines and the replication path
# runs end to end. Not torchrun — it would set LOCAL_WORLD_SIZE=nproc_per_node
# and the fiction collapses.
#
# What this can and cannot answer, stated plainly: `merge_now` is local work
# and the number is real. The transfer beside it goes over loopback and says
# nothing about a network — that is GPU-83's and measure 1's business, and it
# is printed here only to be ignored.
set -euo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"

VENV="${VENV:-/venv/main}"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"

OUT=$KIT_ROOT/out
WORK=$KIT_ROOT/run-single
PARAMS="${PARAMS:-4e8}"
HIDDEN="${HIDDEN:-4096}"
STEPS="${STEPS:-16}"
EVERY="${EVERY:-2}"
export EVERY
mkdir -p "$OUT"

section() { printf '\n\033[1m-- %s --------------------------------\033[0m\n' "$1"; }

arm() {
    # TAG keeps a sweep's arms apart in the report: replicated-2e8,
    # replicated-4e8, and so on.
    local name="$1${TAG:+-$TAG}" replicate="$2"
    section "$name on $(hostname): 2 ranks, 2 'machines', GPUs 0 and 1"
    df -PBG "$KIT_ROOT" | awk 'NR == 2 {print "  free: " $4}'

    rm -rf "$WORK"
    export RAVEX_TIMING_OUT="$OUT/single.$name.jsonl"
    rm -f "$OUT/single.$name."*.jsonl

    local pids=()
    for rank in 0 1; do
        local dir="$WORK/rank$rank"
        mkdir -p "$dir"
        cat > "$dir/ravex.yaml" <<YAML
checkpoint_every: $EVERY
backend: moonclip
keep_last: 3
sharded_checkpoints: per_rank
replicate_every: $replicate
storage:
  type: local
  path: ./checkpoints
log_file: ./ravex.log
log_level: INFO
YAML
        (
            cd "$dir"
            RANK=$rank WORLD_SIZE=2 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
            GROUP_RANK=$rank CUDA_VISIBLE_DEVICES=$rank \
            MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29601}" \
            python $KIT_ROOT/kit/run_train.py -- \
                --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
                --trace "$dir/trace.jsonl" --measure none \
                > "$OUT/single.$name.rank$rank.out" 2>&1
        ) &
        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = "0" ] || echo "  ** a rank exited non-zero - read $OUT/single.$name.rank*.out **"

    for rank in 0 1; do
        cp -f "$WORK/rank$rank/ravex.log" "$OUT/single.$name.rank$rank.log" 2>/dev/null || true
        du -sh "$WORK/rank$rank"/checkpoints/* 2>/dev/null | sed 's/^/  /'
    done
    rm -rf "$WORK"
}

case "${1:-report}" in
control)    arm control 0 ;;
replicated) arm replicated 1 ;;
report)
    section "merge_now, on $(hostname)"
    python $KIT_ROOT/kit/report.py "$OUT"/single.*.jsonl
    echo
    echo "  The exchange line above is loopback. It is not measure 1."
    section "what Ravex said"
    grep -hE "spans|Replicated step|handed off|No space|failed" \
        "$OUT"/single.*.rank*.log 2>/dev/null \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' | sort -u | head -20 | sed 's/^/  /'
    ;;
*) echo "usage: $0 {control|replicated|report}" >&2; exit 2 ;;
esac
