#!/usr/bin/env bash
# The multi-machine bench: N containers, one rank each, one scenario at a time.
#
#   bash integration/multinode/run.sh shared
#   bash integration/multinode/run.sh split
#   bash integration/multinode/run.sh node-loss
#   bash integration/multinode/run.sh reshuffle
#
# Run from the repository root. Builds the image on first use.
#
# Why containers and not `torchrun --nproc_per_node=6`: the question is which
# ranks can see which directory, and ranks on one machine all see the same one.
# A container per rank with LOCAL_WORLD_SIZE=1 is the smallest thing that makes
# six *machines* rather than six processes.
#
# Docker volumes, not bind mounts from the host. A bind mount on Docker Desktop
# crosses a 9p/virtiofs boundary with filesystem semantics of its own, and this
# bench is entirely about filesystem semantics — the substrate has to be an
# ordinary Linux filesystem or the result means nothing.
#
# What this bench is NOT: a network filesystem. Every scenario runs on local
# disk, shared or not. NFS attribute caching — a rank acting on a stale
# directory listing — is the one behaviour it cannot reproduce, and it stays
# open. See docs/how-it-works.md, "More than one machine".

set -euo pipefail

# Git Bash on Windows rewrites anything shaped like an absolute Unix path into
# a Windows one before the process sees it, so `/app/integration/...` reaches
# Docker as `D:/Git/app/integration/...` and the container reports a file
# nobody asked for. Ignored on Linux.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

SCENARIO="${1:-shared}"
NODES="${NODES:-6}"
IMAGE="ravex-multinode"
NET="ravex-bench-net"
DIE_AT="${DIE_AT:-25}"
RESULTS="${RESULTS:-/tmp/ravex-bench}"
WAIT_LIMIT="${WAIT_LIMIT:-180}"

vol()  { echo "ravex-bench-store-$1"; }
name() { echo "node$1"; }

cleanup() {
    for i in $(seq 0 $((NODES - 1))); do
        docker rm -f "$(name "$i")" >/dev/null 2>&1 || true
    done
}

drop_volumes() {
    for i in $(seq 0 $((NODES - 1))); do
        docker volume rm -f "$(vol "$i")" >/dev/null 2>&1 || true
    done
    docker volume rm -f ravex-bench-store-shared >/dev/null 2>&1 || true
    docker volume rm -f ravex-bench-results     >/dev/null 2>&1 || true
}

# Which volume each rank's /checkpoints comes from. This one function is the
# entire difference between the scenarios.
store_for() {
    local rank="$1" round="$2"
    case "$SCENARIO" in
        shared)          echo "ravex-bench-store-shared" ;;
        split|node-loss) vol "$rank" ;;
        # The failure no launcher promises not to cause: the same disks with
        # different ranks on them. The permutation applies to the *second*
        # round only — a mapping that is the same both times reshuffles
        # nothing and quietly reruns `split` under another name.
        reshuffle)
            if [ "$round" = "first" ]; then
                vol "$rank"
            else
                vol "$(( (rank + 1) % NODES ))"
            fi
            ;;
        *) echo "unknown scenario: $SCENARIO" >&2; exit 2 ;;
    esac
}

# `--init` is load-bearing, not tidiness. Each round ends with the training
# script sending itself SIGKILL, which is how a preempted box goes — but the
# kernel discards a SIGKILL aimed at PID 1 from inside its own namespace when
# PID 1 has no handler for it. Without an init process the script runs to
# completion, the bench reports N happy nodes, and it has proved nothing. The
# rest of integration/ never meets this: there torchrun is PID 1 and the ranks
# are its children.
#
# RAVEX_LOG_FILE matters just as much. Left unset, Ravex sends WARNING and
# above to stderr and drops everything below — including the topology
# announcement, which is the single line most of these scenarios exist to
# produce.
launch() {
    local rank="$1" die_at="$2" round="$3"
    docker run -d --init --name "$(name "$rank")" --network "$NET" \
        -v "$(store_for "$rank" "$round")":/checkpoints \
        -v ravex-bench-results:/out \
        -e RANK="$rank" \
        -e WORLD_SIZE="$NODES" \
        -e LOCAL_RANK=0 \
        -e LOCAL_WORLD_SIZE=1 \
        -e MASTER_ADDR=node0 \
        -e MASTER_PORT=29500 \
        -e RAVEX_ENABLED=1 \
        -e RAVEX_STORAGE_PATH=/checkpoints \
        -e RAVEX_SHARDED_CHECKPOINTS=per_rank \
        -e RAVEX_CHECKPOINT_EVERY=10 \
        -e RAVEX_REPLICATE_EVERY="${REPLICATE_EVERY:-2}" \
        -e RAVEX_LOG_LEVEL=INFO \
        -e RAVEX_LOG_FILE="/out/ravex.$round.rank$rank.log" \
        "$IMAGE" \
        python /app/integration/multinode/train.py \
            --trace "/out/$round.rank$rank.jsonl" --die-at "$die_at" \
        >/dev/null
}

run_round() {
    local die_at="$1" label="$2" i
    echo "--- $label ---"
    for i in $(seq 0 $((NODES - 1))); do launch "$i" "$die_at" "$label"; done

    # Bounded. A rank that dies inside a collective leaves the others waiting
    # on a participant that is not coming, and gloo's own timeout is half an
    # hour — long enough that a broken bench looks like a slow one.
    for i in $(seq 0 $((NODES - 1))); do
        timeout "$WAIT_LIMIT" docker wait "$(name "$i")" >/dev/null 2>&1 \
            || echo "  node$i did not exit within ${WAIT_LIMIT}s; stopping it"
    done
    cleanup
}

# The summary reads the volume from inside a container. Copying it to the host
# first would work on Linux and quietly not on Docker Desktop, where a host
# path in `-v` is resolved against the VM rather than against Windows — and the
# failure looks like a bench that found nothing rather than one that could not
# read its own output.
summarise() {
    # -i, or docker does not forward the heredoc to the shell inside.
    docker run --rm -i -v ravex-bench-results:/out alpine:latest sh <<'SUMMARY'
        echo "=== the topology, as each rank announced it ==="
        grep -h -oE "(Checkpoint storage .*|This job spans .*)" \
            /out/ravex.resume.rank*.log 2>/dev/null \
            | sort -u | sed "s/^/  /" || echo "  (nothing)"
        echo
        echo "=== where each rank resumed ==="
        grep -h -E "Resumed at|No checkpoint|took one back|rebuilt one from|fetched one back|from scratch" \
            /out/ravex.resume.rank*.log 2>/dev/null \
            | sed -E "s/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //" \
            | sort | uniq -c | sed "s/^/ /" || echo "  (nothing)"
SUMMARY
}

# Optional: a copy on the host for reading afterwards. Best effort — see above.
copy_out() {
    [ -n "${RESULTS:-}" ] || return 0
    mkdir -p "$RESULTS" 2>/dev/null || return 0
    docker run --rm -v ravex-bench-results:/out -v "$RESULTS":/host alpine:latest         sh -c 'cp -f /out/* /host/ 2>/dev/null; true' >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "=== scenario: $SCENARIO, $NODES nodes ==="

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "--- building $IMAGE ---"
    docker build -f integration/multinode/Dockerfile -t "$IMAGE" .
fi

cleanup
drop_volumes
mkdir -p "$RESULTS"
rm -f "$RESULTS"/*.log "$RESULTS"/*.jsonl
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# Round one: train, then go the way a preempted box goes.
run_round "$DIE_AT" "first"

# What is actually on each disk when the second round starts. Printed rather
# than assumed: the interesting scenarios all turn on which bytes are where,
# and a bench that only reports the outcome cannot tell "the data was gone"
# from "the data was there and nobody looked".
show_layout() {
    local i
    echo "--- what each disk holds now ---"
    for i in $(seq 0 $((NODES - 1))); do
        printf '  %-28s ' "$(store_for "$i" resume):"
        docker run --rm -v "$(store_for "$i" resume)":/c alpine:latest sh -c '
            own=$(ls -d /c/rank_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "
" " ")
            rep=$(ls -d /c/replica/rank_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "
" " ")
            echo "stores: ${own:-none} | replicas: ${rep:-none}"'
    done
}

# The disturbance, between the rounds. `shared` and `split` have none: they ask
# what the storage *is*, not what survives losing it.
case "$SCENARIO" in
    node-loss)
        echo "--- wiping node3's disk, as a replaced machine would arrive ---"
        docker volume rm -f "$(vol 3)" >/dev/null
        ;;
    reshuffle)
        echo "--- ranks keep their number, the disks move under them ---"
        ;;
esac

show_layout

# Round two: the same command, which is the whole proposition.
run_round 0 "resume"

echo
summarise
copy_out
echo
echo "Logs and traces live in the docker volume ravex-bench-results."
echo "Read one:  docker run --rm -v ravex-bench-results:/out alpine cat /out/ravex.resume.rank0.log"
