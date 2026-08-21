#!/usr/bin/env bash
# Is the 500 MB/s the link, the disk, or the code? On loopback, so the link is
# not in the picture at all.
#
#   bash $KIT_ROOT/kit/80-transport-ceiling.sh
set -euo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"
VENV="${VENV:-/venv/main}"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"
OUT=$KIT_ROOT/out
GIB="${GIB:-4}"
mkdir -p "$OUT"

printf '\n\033[1m-- transport ceiling on %s, %s GiB --------------------------------\033[0m\n' \
    "$(hostname)" "$GIB"
df -PBG "$KIT_ROOT" | awk 'NR == 2 {print "  free: " $4}'

for rank in 0 1; do
    RANK=$rank WORLD_SIZE=2 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29701}" \
    RAVEX_ENABLED=0 \
    python $KIT_ROOT/kit/transport_ceiling.py --gib "$GIB" \
        > "$OUT/ceiling.rank$rank.out" 2>&1 &
done
wait
cat "$OUT/ceiling.rank0.out"
