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

# GPU-111: the detection is no longer a collective per step. A preempted rank
# writes one key naming the step everybody saves at, and an ordinary step is a
# `check` on a key that is not there. What that means for this phase is that
# the log now carries an announcement, and it carries it on **both** machines
# with the **same** step in it.
#
# This is the first time that protocol runs over a link with a real round trip
# in it. On loopback the key is written and read in microseconds; here the
# announcement has to cross a continent before the announced step arrives, and
# `ANNOUNCE_LEAD` (two cadence checks) is what buys the time for it. If the
# two machines print different steps, or one prints the "too late to join"
# line, the lead is too short for this link and that is the number to change.
echo "-- GPU-111: the announced step, which must match on both machines --" | tee -a "$LOG"
grep -E "saves together at step|too late to join" "$DIR/ravex.log" 2>/dev/null | tee -a "$LOG"

# GPU-125: the detection channel is one group per machine when the sharding
# stays on a machine, and one group over the job when it does not.
#
# **This shape cannot reach the new branch, and that is worth knowing before
# somebody reads its absence as a failure.** One GPU per box means
# NPROC_PER_NODE=1, so the FSDP group spans both machines - the sharding
# crosses a machine, the wide group is the correct answer, and the line below
# is expected to print nothing. Its absence is the *unchanged* branch working.
#
# Reaching the new branch needs two GPUs per box with FSDP inside each one,
# which is a different rental and a different phase.
echo "-- GPU-125: per-machine detection (expected silent at 1 GPU/box) --" | tee -a "$LOG"
grep -E "SIGTERM detection runs per machine" "$DIR/ravex.log" 2>/dev/null | tee -a "$LOG"

exit 0
