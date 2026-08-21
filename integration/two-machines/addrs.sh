#!/usr/bin/env bash
# Tell each box who it is and where the other one is.
#
#   bash addrs.sh 10.0.0.2 10.0.0.3                       # vast.ai overlay
#   bash addrs.sh abc123.runpod.internal def456.runpod.internal   # RunPod
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

[ $# -eq 2 ] || { echo "usage: $0 <node0 overlay IP> <node1 overlay IP>" >&2; exit 2; }
IP0="$1"; IP1="$2"

write() {
    local host="$1" port="$2" rank="$3" self="$4" peer="$5"
    "${SSH[@]}" -p "$port" "$host" "cat > $KIT_ROOT/kit/box.env <<ENV
export KIT_ROOT=$KIT_ROOT
export NODE_RANK=$rank
export SELF_ADDR=$self
export PEER_ADDR=$peer
export NPROC=\${NPROC:-1}
ENV
cat $KIT_ROOT/kit/box.env"
}

write "$HOST0" "$PORT0" 0 "$IP0" "$IP1"
write "$HOST1" "$PORT1" 1 "$IP1" "$IP0"

cat >> "$HERE/boxes.env" <<ENV
IP0=$IP0
IP1=$IP1
ENV
echo
echo "Then: bash on-both.sh 'bash \$KIT/00-preflight.sh'"
