#!/usr/bin/env bash
# What node 0 does while there is no overlay: the two questions a single box
# can answer honestly.
#
#   1. Where the 500 MB/s in the replication path comes from — measured on
#      loopback, so the link is not a variable.
#   2. Whether merge_now's cost is a constant or grows with the store.
set -uo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"
bash $KIT_ROOT/kit/80-transport-ceiling.sh
bash $KIT_ROOT/kit/60-size-sweep.sh
echo DONE-NODE0
