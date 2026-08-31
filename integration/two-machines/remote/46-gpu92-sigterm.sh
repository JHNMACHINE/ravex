#!/usr/bin/env bash
# GPU-92: one rank is preempted, every rank saves together.
#
#   bash on-both.sh 'bash $KIT/46-gpu92-sigterm.sh sigterm1'
#
# The coordinated emergency path only ever runs when the job is sharded AND
# spans more than one machine — on a single box every local process gets
# SIGTERM at the same instant from the same source, so
# `_emergency_coordination_active` turns the whole thing off there. That is
# why this phase exists here and not in the unit suite: a single box can only
# test the branch that does nothing.
#
# One rank per machine (NPROC=1) on purpose: WORLD_SIZE=2 with
# LOCAL_WORLD_SIZE=1 is what makes `spans_several_machines()` true, and it is
# the honest two-machine shape rather than two local ranks pretending.
set -uo pipefail
KIT_ROOT="${KIT_ROOT:-/root}"
. "$KIT_ROOT/kit/lib.sh"

TAG="${1:-sigterm}"
STEPS="${STEPS:-40}"
DIE_AT="${DIE_AT:-12}"
VICTIM="${VICTIM:-1}"
PORT="${MASTER_PORT:-29501}"

OUT="$KIT_ROOT/out"
mkdir -p "$OUT"

# A workspace with per_rank sharded checkpoints, which is the only layout the
# emergency path applies to — `gather` puts the whole state on rank 0, which
# needs nothing from anyone and already worked.
DIR="$(fresh_workspace gpu92 4 0)"
cd "$DIR"

LOG="$OUT/$TAG.node$NODE_RANK.log"
echo "-- gpu92 sigterm: steps=$STEPS die_at=$DIE_AT victim=rank$VICTIM" | tee "$LOG"
echo "   node_rank=$NODE_RANK master=$MASTER_ADDR:$PORT workspace=$DIR" | tee -a "$LOG"

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-podnet1}" \
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-podnet1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
RAVEX_ENABLED=1 \
STEPS="$STEPS" DIE_AT="$DIE_AT" VICTIM="$VICTIM" HIDDEN="$HIDDEN" \
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node=1 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    "$KIT_ROOT/kit/gpu92_sigterm.py" 2>&1 | tee -a "$LOG"

status=${PIPESTATUS[0]}
echo "-- torchrun exit $status (non-zero is expected: the victim dies)" | tee -a "$LOG"

# The verdict is on disk, not in the exit code. Three questions, each
# answered from what was actually written rather than from the log's prose:
#
#   1. did this rank write a checkpoint tagged emergency=true
#   2. did the *survivor* write one too — the whole point of coordinating
#   3. did the survivor keep training afterwards
echo "-- what this machine holds --" | tee -a "$LOG"
cp -f "$DIR/ravex.log" "$OUT/$TAG.ravex.node$NODE_RANK.log" 2>/dev/null || true
layout "$DIR" | tee -a "$LOG"

echo "-- emergency flags found in metadata --" | tee -a "$LOG"
grep -ro 'emergency[^,}]*' "$DIR/checkpoints" 2>/dev/null | head -20 | tee -a "$LOG"

echo "-- what ravex logged about the emergency path --" | tee -a "$LOG"
grep -iE "SIGTERM|emergency" "$DIR/ravex.log" 2>/dev/null | tail -20 | tee -a "$LOG"

exit 0
