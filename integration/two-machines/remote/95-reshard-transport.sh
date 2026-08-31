#!/usr/bin/env bash
# GPU-96: what a cross-machine reshard would cost, before writing the transport.
#
#   bash $KIT/95-reshard-transport.sh stores    # both boxes: real per-rank stores
#   bash $KIT/95-reshard-transport.sh ceiling   # both boxes: raw TCP, iperf3
#   bash $KIT/95-reshard-transport.sh move      # both boxes: the three legs
#
# The order the issue asks for. `_resume.py` refuses a reshard whose old stores
# are split across machines, and the refusal is deliberate - carrying on with
# the shards that happen to be visible produces a tensor with a band of
# uninitialised rows. Lifting the refusal means moving bytes between machines,
# and the question worth settling first is whether that is cheaper than putting
# the checkpoint somewhere both machines can already see.
#
# `ceiling` is separate from `move` on purpose. A slow p2p leg has two very
# different causes - a slow link, or a transport that is not using the link -
# and they call for opposite decisions. iperf3 tells them apart, and it is the
# first thing to look at if the p2p number disappoints.
. "$(dirname "$0")/lib.sh"

PHASE="${1:-move}"
STEPS="${STEPS:-8}"
EVERY="${EVERY:-4}"
PORT="${RESHARD_PORT:-29777}"

case "$PHASE" in

stores)
    # Each machine on its own, not one job across both. That looks like a
    # shortcut and is not: what this phase has to produce is a store of
    # realistic size and file structure, and running FSDP2 *across* the two
    # boxes to get one makes every step wait on a cross-machine all-gather
    # over the very link whose slowness is the thing under investigation.
    # Measured on 2026-08-31: one step every ~5 minutes, all of it link and
    # none of it store. The bytes that come out either way are the same bytes.
    section "a real per-rank store on each machine, built locally"
    disk_guard
    dir="$(fresh_workspace gpu96 "$EVERY" 0)"
    export RAVEX_TIMING_OUT="$OUT/timing.gpu96.jsonl"
    cd "$dir"
    torchrun --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
        $KIT_ROOT/kit/run_train.py -- \
        --params "$PARAMS" --hidden "$HIDDEN" --steps "$STEPS" \
        --trace "$dir/trace.jsonl" --measure none \
        2>&1 | tee "$OUT/gpu96.stores.node$NODE_RANK.out"
    cp -f "$dir/ravex.log" "$OUT/gpu96.stores.node$NODE_RANK.log" 2>/dev/null || true
    section "what landed, and who wrote it"
    layout "$dir"
    ;;

ceiling)
    section "raw TCP between the two boxes (iperf3)"
    command -v iperf3 >/dev/null || {
        apt-get update -qq && apt-get install -y -qq iperf3 >/dev/null 2>&1 || true
    }
    if command -v iperf3 >/dev/null; then
        if [ "$NODE_RANK" = "0" ]; then
            timeout 45 iperf3 -s -1 -B "$SELF_ADDR" 2>&1 | tail -6 | sed 's/^/  /' || true
        else
            sleep 3
            iperf3 -c "$PEER_ADDR" -t 20 -P 4 2>&1 | tail -6 | sed 's/^/  /' || true
        fi
    else
        echo "  no iperf3 and none installable; the ceiling stays unknown"
    fi \
        | tee "$OUT/gpu96.ceiling.node$NODE_RANK.out"
    ;;

move)
    section "moving one machine's store to the other"
    dir="$WORK/gpu96"
    # Whichever store this machine has. A `rank_<n>` directory when the run
    # that wrote it was sharded across ranks, and `checkpoints` itself when it
    # was not - the standalone runs this phase uses are world_size 1, and
    # Ravex quite reasonably writes them as an ordinary store rather than
    # inventing a per-rank layout for a single rank. Nothing here cares:
    # `open_rank_store` already says a per-rank store *is* an ordinary
    # single-rank store at a different path, so either shape is the same bytes
    # to move.
    # `|| true` for the same reason lib.sh needs it on `peer_iface`: pipefail
    # makes a no-match `ls` fail the whole assignment, and set -e then kills
    # the script before the fallback on the next line can be reached. The
    # symptom is an exit 2 with no message, because the message is on a line
    # that never runs.
    store="$(ls -d "$dir"/checkpoints/rank_* 2>/dev/null | head -1 || true)"
    [ -n "$store" ] && [ -d "$store" ] || store="$dir/checkpoints"
    [ -d "$store/snapshots" ] || { echo "  no store under $dir/checkpoints - run 'stores' first" >&2; exit 2; }
    into="$WORK/gpu96-arrived"
    rm -rf "$into" "$into-remote"; mkdir -p "$into"

    # boto3 only for the remote leg, and only if there is anything to reach.
    if [ -n "${RAVEX_S3_ACCESS_KEY:-}" ]; then
        python -c "import boto3" 2>/dev/null || pip install --quiet boto3
    fi

    du -sh "$store" | sed 's/^/  /'
    launch $KIT_ROOT/kit/reshard_transport.py \
        --store "$store" --into "$into" \
        --self-addr "$SELF_ADDR" --peer-addr "$PEER_ADDR" --port "$PORT" \
        --legs "${LEGS:-p2p,remote}" \
        --out "$OUT/gpu96.transport.json" \
        2>&1 | tee "$OUT/gpu96.move.node$NODE_RANK.out"
    ;;

refuse-write)
    # Four old ranks, two per machine, with peer replication on. Machine 0
    # ends up holding its own two stores *and* copies of machine 1's; machine
    # 1 holds only its own. That asymmetry is the whole point - it is what
    # makes the two surviving ranks disagree about what they can see, which is
    # the exact condition `_resharded_state` refuses on. A single box cannot
    # produce it, because on one box both ranks see the same directory.
    #
    # Tiny model on purpose. Every step here waits on a cross-machine
    # all-gather over an 11.7 MB/s link, so the parameter count is what
    # decides whether this phase takes a minute or an hour.
    section "four ranks over two machines, machine 0 also holding copies"
    disk_guard
    dir="$(fresh_workspace gpu96refuse "$EVERY" 1)"
    cd "$dir"
    NPROC=2 torchrun \
        --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node=2 \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        --max-restarts=0 \
        $KIT_ROOT/kit/run_train.py -- \
        --params "${PARAMS:-4e6}" --hidden 1024 --steps "$STEPS" \
        --trace "$dir/trace.jsonl" --measure none \
        2>&1 | tee "$OUT/gpu96.refuse-write.node$NODE_RANK.out"
    section "what each machine can see"
    layout "$dir"
    ;;

refuse-resume)
    # The same workspace, now with two ranks instead of four: one per machine.
    # Ravex should notice the world changed, want to reshard 4 -> 2, and then
    # refuse - because rank 0 reaches all four old stores through the copies
    # and rank 1 reaches only two. Half the ranks resuming from a step the
    # other half cannot reach is worse than neither resuming, which is what
    # the message says and what this phase is here to see it say on real
    # machines rather than on a directory layout arranged to imitate them.
    section "resuming four ranks' stores onto two, from two machines"
    dir="$WORK/gpu96refuse"
    [ -d "$dir/checkpoints" ] || { echo "  run 'refuse-write' first" >&2; exit 2; }
    cd "$dir"
    launch $KIT_ROOT/kit/run_train.py -- \
        --params "${PARAMS:-4e6}" --hidden 1024 --steps 2 \
        --trace "$dir/trace2.jsonl" --measure none \
        2>&1 | tee "$OUT/gpu96.refuse-resume.node$NODE_RANK.out"
    section "what Ravex decided"
    grep -E "reshard|Reshard|do not see|scratch|resum" "$dir/ravex.log" \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] //' | sed 's/^/  /' | tail -20
    cp -f "$dir/ravex.log" "$OUT/gpu96.refuse-resume.node$NODE_RANK.log" 2>/dev/null || true
    ;;

*) echo "usage: $0 {stores|ceiling|move|refuse-write|refuse-resume}" >&2; exit 2 ;;
esac
