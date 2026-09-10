# Sourced by every phase script. Nothing here runs on its own.
#
# Two boxes, one rank each by default: WORLD_SIZE=2, LOCAL_WORLD_SIZE=1, which
# on two real machines is the honest configuration rather than the fiction that
# found GPU-82 on a single box.

set -euo pipefail

: "${NODE_RANK:?export NODE_RANK=0 on the master box, 1 on the other}"
: "${SELF_ADDR:?export SELF_ADDR=<this box's private address or name>}"
: "${PEER_ADDR:?export PEER_ADDR=<the other box's private address or name>}"

# Which interpreter has torch, decided here rather than inherited. Every phase
# arrives as `ssh host '...'` — not a login shell — so a venv that only an
# interactive shell gets on its PATH is one this never sees. vast.ai keeps
# torch in /venv/main; RunPod's images put it on the default python. Asked
# rather than assumed, because otherwise the answer is ModuleNotFoundError
# three phases in, on a box being billed by the second.
if [ -z "${VENV:-}" ]; then
    for candidate in /venv/main /workspace/venv "$HOME/venv"; do
        [ -x "$candidate/bin/python" ] || continue
        "$candidate/bin/python" -c "import torch" 2>/dev/null || continue
        VENV="$candidate"
        break
    done
fi
if [ -n "${VENV:-}" ] && [ -x "$VENV/bin/python" ]; then
    export PATH="$VENV/bin:$PATH"
fi

NNODES=2
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
if [ "$NODE_RANK" = "0" ]; then MASTER_ADDR="$SELF_ADDR"; else MASTER_ADDR="$PEER_ADDR"; fi

# Where everything lives on the box. /root on vast.ai, where the container
# disk is the disk; /workspace on RunPod, where it is the volume that was
# paid for and the container disk is small enough to fill without noticing.
# Set KIT_ROOT at push time and every path downstream follows.
KIT_ROOT="${KIT_ROOT:-/root}"
SRC="$KIT_ROOT/ravex"
OUT="$KIT_ROOT/out"
WORK="$KIT_ROOT/run"
PARAMS="${PARAMS:-4e8}"
HIDDEN="${HIDDEN:-4096}"

mkdir -p "$OUT"

# The interface that actually reaches the other box. Asked of the routing
# table rather than assumed to be eth0: on vast.ai the overlay arrives as a
# second interface, and NCCL left to itself will sometimes decide that two
# machines' eth1 can talk when they cannot. Gloo needs the same treatment —
# it picks its interface from the hostname, which here resolves to the NAT
# address, and the replication's byte transport is a gloo subgroup.
# PEER_ADDR may be a name rather than an address: RunPod's global networking
# hands out `<pod-id>.runpod.internal`, and `ip route get` wants something it
# can route to. On vast.ai's overlay this is a no-op — the address already is
# an address.
peer_ip() {
    local ip=""
    case "$PEER_ADDR" in
        *[!0-9.]*) : ;;                     # a name: resolve it below
        *) echo "$PEER_ADDR"; return ;;     # already an address
    esac
    # ahostsv4, not `hosts`: the latter answers ::1 for anything with an IPv6
    # record, and an IPv6 peer address with an IPv4 route is a confusing way to
    # fail. Both transports here are pinned to an interface, not a family.
    ip="$(getent ahostsv4 "$PEER_ADDR" 2>/dev/null | awk '{print $1; exit}')" || true
    if [ -z "$ip" ]; then
        ip="$(getent hosts "$PEER_ADDR" 2>/dev/null | awk '{print $1; exit}')" || true
    fi
    if [ -z "$ip" ]; then
        ip="$(python -c 'import socket, sys
try:
    print(socket.gethostbyname(sys.argv[1]))
except OSError:
    pass' "$PEER_ADDR" 2>/dev/null)" || true
    fi
    echo "$ip"
}

PEER_IP="${PEER_IP:-$(peer_ip || true)}"

peer_iface() {
    ip -o route get "${PEER_IP:-$PEER_ADDR}" 2>/dev/null \
        | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") print $(i + 1)}' \
        | head -1
}

# `|| true` because pipefail makes a failed `ip route get` fail the whole
# assignment, and under set -e that kills the script before 00-preflight.sh can
# say *why* there is no route. An empty IFACE is the diagnosis, not an error.
IFACE="${IFACE:-$(peer_iface || true)}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"      # no IB across a NAT overlay
export PYTHONUNBUFFERED=1

section() { printf '\n\033[1m-- %s --------------------------------\033[0m\n' "$1"; }

# **Where a phase says what it is doing, while it is doing it** - appended, on
# the box, one timestamped line at a time. `on-both.sh` only hands its output
# back when the command ends, so until this existed the difference between "the
# phase is working" and "the phase is wedged" could only be settled by guessing
# and then by `ps`. `watch.sh` from the laptop tails this file on both boxes.
PHASE_LOG="$OUT/phase.log"

progress() {
    local line
    line="$(date -u +%H:%M:%S) node$NODE_RANK $*"
    echo "$line" >> "$PHASE_LOG"
    echo "  $line"
}

# Announce the phase and leave a marker, so a tail that starts late still knows
# which phase it is looking at.
phase_begin() {
    echo "$*" > "$PHASE_LOG.current"
    progress "BEGIN $*"
}

phase_end() {
    progress "END $*"
}

# One workspace per phase. Ravex finds its config by walking up from the
# working directory, and a fresh store means the phase pays for a full first
# checkpoint instead of inheriting one.
workspace() {
    local name="$1" every="$2" replicate="$3"
    local dir="$WORK/$name"
    mkdir -p "$dir"
    cat > "$dir/ravex.yaml" <<YAML
checkpoint_every: $every
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
    echo "$dir"
}

fresh_workspace() {
    local name="$1"
    rm -rf "${WORK:?}/$name"
    workspace "$@"
}

# What is on this disk, printed rather than assumed. The reshuffle scenario in
# integration/multinode is what taught this: every replica present, none used,
# and an outcome line that could not tell the two apart.
layout() {
    local dir="$1"
    echo "  host $(hostname)  node_rank $NODE_RANK"
    for store in "$dir"/checkpoints/rank_* "$dir"/checkpoints/replica/rank_*; do
        [ -e "$store" ] || continue
        local owner="(no .ravex-owner)"
        # The record names the machine that wrote the store. On two real boxes
        # the hostnames differ, which is the thing a single box could never show.
        [ -f "$store/.ravex-owner" ] \
            && owner="$(tr -d '{}"' < "$store/.ravex-owner" | tr -s ' ')"
        printf '  %-44s %8s  %s\n' \
            "${store#$dir/}" "$(du -sh "$store" 2>/dev/null | cut -f1)" "$owner"
    done
}

# Space is the silent failure mode: Ravex logs "No space left", disables itself
# and training carries on, so every rank exits fine and the numbers say nothing.
disk_guard() {
    local free
    free=$(df -PBG "$KIT_ROOT" | awk 'NR == 2 {gsub(/G/, "", $4); print $4}')
    echo "  free on $KIT_ROOT: ${free} GiB"
    [ "${free:-0}" -lt 20 ] && echo "  ** under 20 GiB: rm -rf $WORK/* before trusting anything **"
    return 0
}

launch() {
    # torchrun with a static rendezvous. --max-restarts=0 because a phase that
    # ends in SIGKILL is the point, not a fault to retry.
    torchrun \
        --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        --max-restarts=0 \
        "$@"
}
