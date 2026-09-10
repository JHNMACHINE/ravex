#!/usr/bin/env bash
# GPU-120, point 7: the outer loop on two real machines in two real regions.
#
#   bash $KIT/100-outer.sh base       # both boxes
#   bash fetch.sh outer-base          # from your machine, before the next arm
#   bash $KIT/100-outer.sh dtype      # both boxes
#   bash fetch.sh outer-dtype
#   bash $KIT/100-outer.sh kill       # both boxes; node 1 goes away mid-run
#   bash fetch.sh outer-kill
#   bash $KIT/100-outer.sh resume     # both boxes, same workspace
#   bash fetch.sh outer-resume
#
# Everything the outer loop is built out of has only ever run on loopback,
# where the network is free and every round printed `took 0.0s`. That is the
# one number the whole architecture is chosen around, and it had never been
# observed. These arms observe it.
#
# The arms, and what each decides:
#
#   base    the round split over the real link. `gather_seconds` and
#           `gather_wait_seconds` against what bench/round_link_cost.py
#           predicts for the bandwidth and RTT that 40- and 41- just measured.
#           If they disagree, the bench's defaults move and H moves with them.
#
#   dtype   the same run with `outer_save_dtype: bf16`. GPU-118 measured it as
#           free in convergence and worth a fifth of every round; it ships as
#           null only because this run had never happened. This arm is what
#           unblocks it for 0.1.1 - and what it has to show is the fifth,
#           on the wire, not on loopback.
#
#   kill    node 1 stops dead mid-run, the way a preempted box stops. The
#           survivor's rounds have to keep closing, over one node, and the
#           round report says so.
#
#   resume  the same command again, into the same workspace.
#
# **The round size is the rental bill, and it is not the GPU.** Each round moves
# about `params x 4` bytes per node - the delta is fp32 and dense - so at the
# 7 MB/s measured between two RunPod pods on 2026-08-21:
#
#     params    per round     one round      six rounds
#     5e7        200 MB         29 s          ~3 min
#     1e8        400 MB         57 s          ~6 min     <- the default
#     3e8        1.2 GB        171 s         ~17 min
#
# Set OUTER_PARAMS from what 40-bandwidth.sh actually measured, not from this
# table: a round that takes longer than the rental is not a result. The `dtype`
# arm halves every figure, which is the point of that arm.
. "$(dirname "$0")/lib.sh"

ARM="${1:-base}"
NAME=gpu120
DIR="$WORK/$NAME"
ROUNDS="${ROUNDS:-6}"
INNER="${INNER:-50}"
OUTER_PARAMS="${OUTER_PARAMS:-1e8}"
OUTER_HIDDEN="${OUTER_HIDDEN:-4096}"
BATCH="${BATCH:-8}"
DEADLINE="${DEADLINE:-900}"
DIE_AT_ROUND="${DIE_AT_ROUND:-3}"

# **The address peers dial, said out loud.** Nothing a process can ask its own
# kernel returns the address a peer on another continent reaches it at: behind
# NAT the local interface address is not it, and the hostname is not it either.
# `advertise()` falls back to the routing table and then to loopback with a
# warning - and loopback here is a run where no round ever closes and every
# node trains alone, which looks exactly like a run that is working.
export RAVEX_EXCHANGE_ADDRESS="${RAVEX_EXCHANGE_ADDRESS:-$SELF_ADDR}"

run_outer() {
    local tag="$1"; shift
    mkdir -p "$DIR"
    phase_begin "gpu120 $tag: $ROUNDS rounds x $INNER steps, params $OUTER_PARAMS"
    launch "$KIT_ROOT/kit/outer_run.py" \
        --rounds "$ROUNDS" --inner "$INNER" \
        --params "$OUTER_PARAMS" --hidden "$OUTER_HIDDEN" --batch "$BATCH" \
        --deadline "$DEADLINE" \
        --root "$DIR/$tag" --out "$OUT" "$@" \
        2>&1 | tee "$OUT/gpu120.$tag.node$NODE_RANK.out" || true
    # Named per arm, or the next one overwrites the last one's numbers and the
    # comparison the whole phase exists for is gone.
    for f in "$OUT"/rounds.node*.json; do
        [ -e "$f" ] || continue
        mv -f "$f" "$OUT/gpu120.$tag.$(basename "$f")"
    done
    phase_end "gpu120 $tag"
}

case "$ARM" in

base)
    section "the outer loop over the real link, save_dtype off"
    disk_guard
    echo "  advertising $RAVEX_EXCHANGE_ADDRESS"
    rm -rf "${DIR:?}/base"
    run_outer base
    ;;

dtype)
    section "the same run with outer_save_dtype=bf16 (GPU-118)"
    disk_guard
    rm -rf "${DIR:?}/dtype"
    run_outer dtype --save-dtype bf16
    ;;

kill)
    section "node 1 goes away at round $DIE_AT_ROUND"
    disk_guard
    rm -rf "${DIR:?}/kill"
    # A shorter deadline than the base arm on purpose: what is being measured
    # is that the survivor keeps closing rounds, and every round after the
    # death pays the deadline once. At 900 s that is a phase nobody can afford
    # to watch.
    DEADLINE="${KILL_DEADLINE:-60}" \
        run_outer kill --die-at-round "$DIE_AT_ROUND" --die-on-rank 1
    ;;

resume)
    section "the same command again, into the same workspace"
    disk_guard
    run_outer resume
    ;;

report)
    section "what the rounds said on node $NODE_RANK"
    python - "$OUT" <<'PY'
import glob, json, os, sys

out = sys.argv[1]
for path in sorted(glob.glob(os.path.join(out, "gpu120.*.rounds.node*.json"))):
    payload = json.load(open(path))
    rounds = payload["rounds"]
    print("\n%s  (save_dtype=%s, %d round(s))" % (
        os.path.basename(path), payload["save_dtype"], len(rounds)))
    if not rounds:
        print("  no round closed. If the log has no 'Outer round' line at all,")
        print("  the loop never exchanged - check RAVEX_EXCHANGE_ADDRESS.")
        for line in payload["warnings"][:5]:
            print("  ! " + line)
        continue
    print("  round nodes   total  network (waiting)   delta publish   apply")
    for entry in rounds:
        print("  %5d %5d %7.2f %8.2f %9.2f %7.2f %7.2f %7.2f" % (
            entry["round"], entry["nodes"], entry["total_seconds"],
            entry["gather_seconds"], entry["gather_wait_seconds"],
            entry["delta_seconds"], entry["publish_seconds"],
            entry["apply_seconds"]))
    network = [e["gather_seconds"] for e in rounds]
    print("  network per round: min %.2f  median %.2f  max %.2f" % (
        min(network), sorted(network)[len(network) // 2], max(network)))
PY
    ;;

*)
    echo "usage: $0 {base|dtype|kill|resume|report}" >&2; exit 2 ;;
esac
