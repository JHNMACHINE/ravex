#!/usr/bin/env bash
# GPU-94: tear down a live NCCL communicator and rebuild it at a different
# world size, in-process, across both machines.
#
#   bash on-both.sh 'START=2 TARGET=4 bash $KIT/45-nccl-regroup.sh grow24'
#
# The one question the CPU/gloo suite in tests/test_elastic_*.py cannot
# answer. Gloo never touches a CUDA communicator, so a green local suite says
# nothing about whether destroy_process_group() + init_process_group() is
# safe for NCCL with a model already in VRAM — and nothing at all about
# whether the device memory comes back.
#
# START ranks take part in generation 0; ranks >= START sit out, exactly like
# a box that was provisioned after the run started. TARGET ranks take part in
# generation 1. With NPROC=2 on two boxes, START=2 TARGET=4 puts the whole of
# generation 0 on node 0 and every joiner on node 1, so both the handoff and
# the regroup cross the real network rather than loopback.
#
# The values the joiners need travel over ravex._elastic.prestage_send /
# prestage_receive — the production path from GPU-94 step 4, not a stand-in.
# Not the rendezvous store: a TCPStore payload is capped at 8 MiB and a real
# checkpoint is orders of magnitude past that (tried, and it failed exactly
# there).
set -uo pipefail
KIT_ROOT="${KIT_ROOT:-/root}"
. "$KIT_ROOT/kit/lib.sh"

TAG="${1:-regroup}"
START="${START:-2}"
TARGET="${TARGET:-4}"
HIDDEN="${HIDDEN:-2048}"
LAYERS="${LAYERS:-6}"
PORT="${MASTER_PORT:-29501}"

OUT="$KIT_ROOT/out"
mkdir -p "$OUT"
LOG="$OUT/$TAG.node$NODE_RANK.log"

# Each run gets its own handoff directories: the incremental skip in
# ravex._replication matches files by name and size alone (GPU-97), on the
# assumption that store files are immutable once written. Leaving a previous
# run's file in place is how a stale copy passes for a fresh one.
rm -rf "$KIT_ROOT"/gpu94_handoff_*

echo "-- gpu94 regroup: START=$START TARGET=$TARGET nnodes=$NNODES nproc=$NPROC" | tee "$LOG"
echo "   node_rank=$NODE_RANK master=$MASTER_ADDR:$PORT hidden=$HIDDEN layers=$LAYERS" | tee -a "$LOG"

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-podnet1}" \
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-podnet1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
KIT_ROOT="$KIT_ROOT" \
START="$START" TARGET="$TARGET" HIDDEN="$HIDDEN" LAYERS="$LAYERS" \
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    "$KIT_ROOT/kit/nccl_regroup.py" 2>&1 | tee -a "$LOG"

status=${PIPESTATUS[0]}
echo "-- exit $status" | tee -a "$LOG"

# The verdict, pulled out of the log so `fetch.sh` brings home something a
# human reads rather than a wall of torchrun.
grep '^RESULT ' "$LOG" > "$OUT/$TAG.results.node$NODE_RANK.jsonl" 2>/dev/null || true
exit "$status"
