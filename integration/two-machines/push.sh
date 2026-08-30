#!/usr/bin/env bash
# Laptop side. Copy the working tree and this kit to both boxes.
#
#   bash push.sh 'ssh -p 12345 root@ssh5.vast.ai' 'ssh -p 23456 root@ssh6.vast.ai'
#
# Paste the two connection strings from vast.ai exactly as they are given; the
# extra -L forwards and options are ignored. Re-run it after editing anything
# here — it is a few seconds and it is how a fix reaches the boxes.
#
# The working tree is copied rather than cloned: what is under test (the GPU-82
# fix) is on main, but the point of this session is to try what is on this
# laptop, not what a remote happens to have.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash otherwise rewrites /root into a drive path

HERE="$(cd "$(dirname "$0")" && pwd)"

# The throwaway key. IdentitiesOnly so ssh offers this one and nothing from the
# agent: a box that rejects it should say so now, not fall back to a personal
# key and hide the fact that the disposable one was never installed.
KEY="${KIT_KEY:-$HERE/id_throwaway}"
[ -f "$KEY" ] || { echo "missing $KEY - generate one, see README.md" >&2; exit 2; }
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SCP=(scp -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
# The working tree to send. Defaults to this checkout — the kit lives inside
# the repository it tests — and is copied rather than cloned on purpose:
# what is under test is usually not committed anywhere yet.
SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
# Where the kit lands on the box. /root suits vast.ai, where the container
# disk is the disk. On RunPod use the volume — `KIT_ROOT=/workspace bash
# push.sh ...` — because the container disk there is small and filling it is
# the silent failure: Ravex logs `No space left`, disables itself, and every
# rank still exits 0.
KIT_ROOT="${KIT_ROOT:-/root}"

parse() {  # "ssh -p N root@host ..." -> "host port"
    local port=22 host=""
    # shellcheck disable=SC2086
    set -- $1
    while [ $# -gt 0 ]; do
        case "$1" in
            -p) port="$2"; shift 2 ;;
            *@*) host="$1"; shift ;;
            *) shift ;;
        esac
    done
    [ -n "$host" ] || { echo "could not find user@host in that string" >&2; exit 2; }
    echo "$host $port"
}

[ $# -eq 2 ] || { echo "usage: $0 '<ssh string box0>' '<ssh string box1>'" >&2; exit 2; }

read -r HOST0 PORT0 <<< "$(parse "$1")"
read -r HOST1 PORT1 <<< "$(parse "$2")"

cat > "$HERE/boxes.env" <<ENV
HOST0=$HOST0
PORT0=$PORT0
HOST1=$HOST1
PORT1=$PORT1
KIT_ROOT=$KIT_ROOT
ENV
echo "node 0: $HOST0:$PORT0"
echo "node 1: $HOST1:$PORT1"

send() {
    local host="$1" port="$2" node="$3"
    echo "-- node $node --"
    "${SSH[@]}" -p "$port" "$host" \
        "mkdir -p $KIT_ROOT/ravex $KIT_ROOT/kit $KIT_ROOT/out $KIT_ROOT/run"
    tar czf - -C "$SRC" \
        --exclude .git --exclude .venv --exclude target --exclude build \
        --exclude dist --exclude checkpoints --exclude '*.egg-info' \
        --exclude results \
        --exclude __pycache__ . \
        | "${SSH[@]}" -p "$port" "$host" "tar xzf - --no-same-owner -C $KIT_ROOT/ravex"
    tar czf - -C "$HERE/remote" . | "${SSH[@]}" -p "$port" "$host" "tar xzf - --no-same-owner -C $KIT_ROOT/kit"
    "${SSH[@]}" -p "$port" "$host" "chmod +x $KIT_ROOT/kit/*.sh; ls $KIT_ROOT/kit"
}

send "$HOST0" "$PORT0" 0
send "$HOST1" "$PORT1" 1

echo
echo "Now the private addresses. What each box sees:"
echo "  vast.ai: the interface that is NOT the one this SSH arrived on."
echo "  RunPod:  not an interface at all — use <pod-id>.runpod.internal."
for spec in "$HOST0 $PORT0 0" "$HOST1 $PORT1 1"; do
    # shellcheck disable=SC2086
    set -- $spec
    echo "-- node $3 ($1) --"
    "${SSH[@]}" -p "$2" "$1" 'hostname; ip -o -4 addr show | awk "{printf \"  %-8s %s\n\", \$2, \$4}"'
done
echo
echo "Then: bash addrs.sh <node0 address-or-name> <node1 address-or-name>"
