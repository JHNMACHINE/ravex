#!/usr/bin/env bash
# What node 1 does while there is no overlay: the replication ring at eight,
# with the real backend, including losing a machine and coming back.
#
# Not the network. Eight ranks on one box trade stores over loopback — that is
# GPU-83 and it needs two machines.
set -uo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"
bash $KIT_ROOT/kit/70-eight-ranks.sh run1
bash $KIT_ROOT/kit/70-eight-ranks.sh loss
bash $KIT_ROOT/kit/70-eight-ranks.sh resume
bash $KIT_ROOT/kit/70-eight-ranks.sh report
echo DONE-NODE1
