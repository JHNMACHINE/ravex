#!/usr/bin/env bash
# Bring the results home. Run it after every phase, not at the end.
#
#   bash fetch.sh run1
#
# A vast.ai instance can be outbid and disappear mid-session; anything that
# exists only on the box is a result that can be taken away without warning.
# This is cheap, so run it more often than feels necessary.
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

TAG="${1:-$(date +%H%M%S)}"
DEST="$HERE/results/$TAG"
mkdir -p "$DEST/node0" "$DEST/node1"

"${SCP[@]}" -P "$PORT0" -r "$HOST0:$KIT_ROOT/out/." "$DEST/node0/" || echo "node0: nothing yet"
"${SCP[@]}" -P "$PORT1" -r "$HOST1:$KIT_ROOT/out/." "$DEST/node1/" || echo "node1: nothing yet"
echo
find "$DEST" -type f | sed "s|$HERE/||"
