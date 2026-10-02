#!/usr/bin/env bash
# Tell each box who it is and where the others are.
#
#   bash addrs.sh 10.0.0.2 10.0.0.3                       # vast.ai overlay
#   bash addrs.sh abc123.runpod.internal def456.runpod.internal   # RunPod
#   bash addrs.sh a.runpod.internal@eu b.runpod.internal@eu c.runpod.internal@us d.runpod.internal@us
#
# Whatever the two boxes reach each other by, and it is never the address the
# SSH arrived on. On vast.ai that is the overlay interface, the one that is
# not the NAT eth0; on RunPod it is a name, `<pod-id>.runpod.internal`, and
# no interface on the box carries it. Everything else derives from these two: which
# box is the master, which interface NCCL and gloo are pinned to, who each
# rank's replication peer is.
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd)"

# The throwaway key. IdentitiesOnly so ssh offers this one and nothing from the
# agent: a box that rejects it should say so now, not fall back to a personal
# key and hide the fact that the disposable one was never installed.
KEY="${KIT_KEY:-$HERE/id_throwaway}"
[ -f "$KEY" ] || { echo "missing $KEY - generate one, see README.md" >&2; exit 2; }
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SCP=(scp -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
. "$HERE/boxes.env"
KIT_ROOT="${KIT_ROOT:-/root}"

NBOXES="${NBOXES:-2}"
[ $# -eq "$NBOXES" ] || {
    echo "usage: $0 <node0 address>[@region] <node1 address>[@region] ... - one per box, $NBOXES here" >&2
    exit 2
}
# `address@region` gives a box its region, for the rounds over regions of
# GPU-143 (`120-regions.sh`); every other phase ignores it.
ADDRS=(); REGIONS=()
for spec in "$@"; do
    case "$spec" in
        *@*) ADDRS+=("${spec%@*}"); REGIONS+=("${spec##*@}") ;;
        *) ADDRS+=("$spec"); REGIONS+=("") ;;
    esac
done

# The peer is node 0 for every box but node 0, whose peer is node 1: with two
# boxes that is each one's other, as it always was, and with more it is where
# the master and the rendezvous are.
write() {
    local host="$1" port="$2" rank="$3" self="$4" peer="$5" region="$6"
    "${SSH[@]}" -p "$port" "$host" "cat > $KIT_ROOT/kit/box.env <<ENV
export KIT_ROOT=$KIT_ROOT
export NODE_RANK=$rank
export SELF_ADDR=$self
export PEER_ADDR=$peer
export NODE0_ADDR=${ADDRS[0]}
export NBOXES=$NBOXES
export REGION=$region
export NPROC=\${NPROC:-1}
ENV
cat $KIT_ROOT/kit/box.env"
}

for ((i = 0; i < NBOXES; i++)); do
    host="HOST$i"; port="PORT$i"
    if [ "$i" = 0 ]; then peer="${ADDRS[1]}"; else peer="${ADDRS[0]}"; fi
    write "${!host}" "${!port}" "$i" "${ADDRS[$i]}" "$peer" "${REGIONS[$i]}"
done

for ((i = 0; i < NBOXES; i++)); do
    echo "IP$i=${ADDRS[$i]}"
done >> "$HERE/boxes.env"
echo
echo "Then: bash on-both.sh 'bash \$KIT/00-preflight.sh'"
