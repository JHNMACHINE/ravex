#!/usr/bin/env bash
# Everything that has to be true before a single expensive thing runs.
#
#   NODE_RANK=0 SELF_ADDR=10.0.0.2 PEER_ADDR=10.0.0.3 bash $KIT_ROOT/kit/00-preflight.sh
#
# Run it on both boxes at the same time: the last step is a two-rank handshake
# and it needs both. If this fails, no amount of the rest will work — and it
# fails in seconds rather than after a 3 GiB transfer.
. "$(dirname "$0")/lib.sh"

section "the box"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
    echo "no nvidia-smi: wrong template, or a VM instance" >&2; exit 1; }
echo
echo "  host $(hostname)  cores $(nproc)  RAM $(free -g | awk '/^Mem:/{print $2}') GiB"
disk_guard

section "the network"
ip -o -4 addr show | awk '{printf "  %-8s %s\n", $2, $4}'
echo
echo "  peer $PEER_ADDR${PEER_IP:+ ($PEER_IP)} reached over: ${IFACE:-<none: no route to the peer>}"
[ -n "${PEER_IP:-}" ] || {
    echo "  ** $PEER_ADDR does not resolve. On RunPod that means the pod id is" >&2
    echo "     wrong, or global networking was off at deploy — and it cannot be" >&2
    echo "     added to a running pod. Stop here. **" >&2
    exit 1; }
[ -n "${IFACE:-}" ] || {
    echo "  ** no route to the peer. The private network is not attached: on" >&2
    echo "     vast.ai the overlay is missing or the two instances are in" >&2
    echo "     different physical clusters; on RunPod one of the pods is not on" >&2
    echo "     global networking. Neither is fixable on a running box. **" >&2
    exit 1; }
if command -v ping >/dev/null 2>&1; then
    ping -c 2 -W 2 "$PEER_ADDR" || echo "  (ping blocked; not fatal on its own)"
else
    echo "  (no ping on this image — the handshake below is the real test)"
fi
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"

section "the software"
python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    ok = cap in torch.cuda.get_arch_list()
    print("  gpu is", cap, "and torch has kernels for it:", ok)
    if not ok:
        print("  ** every CUDA op will fail with 'no kernel image'."
              " Run 10-setup.sh, which now fixes this. **")
try:
    import ravex, moonclip
    print("  ravex", ravex.__version__, "| moonclip", moonclip.__version__)
except ImportError as exc:
    print("  not installed yet:", exc, "- run 10-setup.sh")
PY

section "two ranks, two machines, both transports"
# The go/no-go. NCCL is what training runs on; the gloo subgroup is what the
# replication moves bytes on (GPU-82), and on this overlay neither has ever
# been exercised. Cheap enough to run before every session.
launch $KIT_ROOT/kit/handshake.py
