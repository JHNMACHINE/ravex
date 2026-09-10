#!/usr/bin/env bash
# GPU-120, measure 2 of 2: what a round trip between these boxes costs.
#
#   bash $KIT/41-latency.sh        # both boxes, at the same time
#
# `40-bandwidth.sh` says what the pipe carries. This says what it costs to ask,
# and the two do not move together: `bench/round_link_cost.py` finds a floor of
# about 0.31 s under every arm which is the dial and the round trip, and that
# floor does not shrink when the link gets fatter. A pair of boxes with good
# bandwidth and a bad round trip is the case the bench has never been run at.
#
# Node 0 serves and prints nothing interesting; node 1 measures. Both at once.
. "$(dirname "$0")/lib.sh"

SAMPLES="${SAMPLES:-200}"
LAT_PORT="${LAT_PORT:-29610}"

section "round trip, three ways"
python "$KIT_ROOT/kit/latency.py" \
    --port "$LAT_PORT" --samples "$SAMPLES" \
    2>&1 | tee "$OUT/latency.node$NODE_RANK.out"
