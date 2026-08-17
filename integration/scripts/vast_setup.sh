#!/usr/bin/env bash
# Set up a rented multi-GPU box to measure the sharded checkpoint paths.
#
# **For measuring a released version, use `measure_handoff.sh` instead**: since
# 2026-08-17 both libraries are on PyPI, so `pip install "ravex[moonclip]"` gets
# compiled wheels and none of the copying or Rust building below is needed. This
# script is what you want when the change under measurement is *not* released —
# it copies the working trees as they are and builds Moonclip from them.
#
# Run this *on the box*, after the working trees have been copied there:
#
#   rsync -az --exclude .git --exclude target --exclude .venv \
#         D:/dev/gpuzero/ravex/    root@HOST:/root/ravex/
#   rsync -az --exclude .git --exclude target --exclude .venv \
#         D:/dev/gpuzero/moonclip/ root@HOST:/root/moonclip/
#   ssh root@HOST bash /root/ravex/integration/scripts/vast_setup.sh
#
# The working trees are copied rather than cloned on purpose: what is being
# measured is not committed anywhere.
#
# Two phases, and the first one is enough for the collection numbers:
#
#   1. torch + ravex on PYTHONPATH  →  `--measure both` runs
#   2. Rust + maturin + moonclip    →  a real checkpointing run, which is what
#      answers whether the cadence ceiling moves
#
# Pass `--with-moonclip` for phase 2. It builds a Rust extension and takes a
# few minutes.

set -euo pipefail

WITH_MOONCLIP=0
[ "${1:-}" = "--with-moonclip" ] && WITH_MOONCLIP=1

echo "── what we are working with ─────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
    echo "no nvidia-smi: this is not the box we wanted" >&2
    exit 1
}
echo
free -g | head -2
echo
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())"

echo
echo "── ravex ────────────────────────────────────────────────────"
# No install: PYTHONPATH is how the suite runs at home too, and it keeps the
# box from acquiring a second copy of anything.
python -c "import sys; sys.path.insert(0, '/root/ravex'); import ravex; print('ravex', ravex.__file__)"
pip install --quiet PyYAML

if [ "$WITH_MOONCLIP" = "1" ]; then
    echo
    echo "── moonclip (Rust build, a few minutes) ─────────────────"
    if ! command -v cargo >/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain 1.90.0 --profile minimal
    fi
    . "$HOME/.cargo/env"
    pip install --quiet maturin
    cd /root/moonclip
    maturin develop --release
    python -c "import moonclip; print('moonclip', moonclip.__version__)"
fi

cat <<'EOF'

── ready ────────────────────────────────────────────────────

Collection comparison (phase 1), N = number of GPUs:

  cd /root && torchrun --nproc_per_node=N \
      /root/ravex/integration/scripts/train_fsdp_cuda.py \
      --measure both --api fsdp2 --trace /root/measure_fsdp2.json

  ... and again with --api fsdp1, which is the one that has never run
  anywhere but a GPU box.

Sizing: the default 1.5B fp32 is ~24 GiB of weights + grads + Adam across all
ranks, so 24 GiB per rank on 2 GPUs and 3 GiB on 8. The *gather* additionally
needs rank 0 to hold the whole ~16.5 GiB in host RAM — if `free -g` above is
under 32, run with a smaller --params or the gather line will swap and the
comparison becomes a measurement of the page cache.

EOF
