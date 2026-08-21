#!/usr/bin/env bash
# GPU-81, measure 3: what the link between these two boxes actually carries.
#
#   bash $KIT_ROOT/kit/40-bandwidth.sh        # both boxes, at the same time
#
# Three numbers, and the gap between them is the interesting part:
#
#   iperf3   raw TCP, the ceiling
#   gloo     the transport the replication uses, chunked at 4 MiB as it is
#   nccl     the transport training uses, over the same overlay
#
# On vast.ai this is not a cluster interconnect and the point is to find out
# what it is before deciding a cadence from it.
. "$(dirname "$0")/lib.sh"

SIZE_MB="${SIZE_MB:-2048}"

section "raw TCP (iperf3)"
if ! command -v iperf3 >/dev/null; then
    apt-get update -qq && apt-get install -y -qq iperf3 >/dev/null 2>&1 \
        || echo "  (no iperf3 and none installable; skipping the raw ceiling)"
fi
if command -v iperf3 >/dev/null; then
    if [ "$NODE_RANK" = "0" ]; then
        echo "  server on $SELF_ADDR:5201 for 40 s"
        timeout 40 iperf3 -s -1 -B "$SELF_ADDR" | tail -5 | sed 's/^/  /' || true
    else
        sleep 3
        iperf3 -c "$PEER_ADDR" -t 20 -P 4 | tail -5 | sed 's/^/  /' || true
    fi
fi

section "gloo and nccl, through torch"
launch $KIT_ROOT/kit/bandwidth.py --size-mb "$SIZE_MB" \
    2>&1 | tee "$OUT/bandwidth.node$NODE_RANK.out"
