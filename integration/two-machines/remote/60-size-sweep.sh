#!/usr/bin/env bash
# Does merge_now scale with the store, and how does the handoff move with it?
#
#   bash $KIT_ROOT/kit/60-size-sweep.sh
#
# GPU-81 asks for the cost of merge_now as a number. One number at one size is
# a point, and a point cannot say whether the cost is a constant to swallow or
# a term that grows with the model — which is the thing a user with a bigger
# model needs to know. Three sizes, same box, back to back.
set -euo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"

# Same reason as everywhere else here: `ssh host '...'` is not an
# interactive shell, so vast.ai's venv is not on the PATH.
VENV="${VENV:-/venv/main}"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"

OUT=$KIT_ROOT/out
SIZES="${SIZES:-2e8 4e8 8e8}"

for size in $SIZES; do
    echo "free before $size: $(df -PBG "$KIT_ROOT" | awk 'NR == 2 {print $4}')"
    TAG="$size" PARAMS="$size" STEPS="${STEPS:-8}" \
        bash $KIT_ROOT/kit/50-single-box.sh replicated
done

printf '\n\033[1m-- the sweep --------------------------------\033[0m\n'
EVERY="${EVERY:-2}" python $KIT_ROOT/kit/report.py "$OUT"/single.replicated-*.jsonl
