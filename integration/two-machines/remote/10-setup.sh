#!/usr/bin/env bash
# Install what the phases need. A few minutes, including a Rust toolchain for
# ravex's core when the image has none.
#
#   bash $KIT_ROOT/kit/10-setup.sh
#
# Ravex comes from the working tree pushed by push.sh, because what is under
# test is never in the wheel: GPU-82's gloo subgroup is on main and not in
# 0.0.3, and GPU-84's copy-free transfer is on a branch and not even on main.
# Whatever is checked out on the laptop is what runs here — so check out the
# branch you mean to measure *before* running push.sh.
#
# Moonclip comes from PyPI: the storage engine is not what is being tried
# here, and 0.0.9 is what a user gets today. Two floors sit underneath that
# number, both of them failures that hide rather than raise. `restore_from_remote`
# exists from 0.0.8 — on 0.0.7 Ravex swallows the AttributeError and quietly
# does not restore. And `save_dtype` is declared as a string before 0.0.9, so
# the mapping form reaches it as a TypeError that `get_backend` catches along
# with everything else: one configuration line would cost the run its Moonclip
# checkpointing entirely, reported only as "Moonclip backend unavailable".
# Override with MOONCLIP_SPEC=... to pin an older one deliberately.
set -euo pipefail

# /root on vast.ai, /workspace on RunPod. Exported into every phase by
# box.env; the default keeps this runnable by hand.
KIT_ROOT="${KIT_ROOT:-/root}"

# **A floor, not a pin** (found on the rented boxes, 2026-09-10). This said
# `==0.0.9`, and ravex 0.1.0 asks moonclip for `describe`, which 0.0.9 does not
# have: the seed round failed with `'MoonclipManager' object has no attribute
# 'describe'`, every node fell back to training alone, and the phase looked
# like a network problem. An exact pin ages into a floor that is below the
# code being measured; a floor does not.
MOONCLIP_SPEC="${MOONCLIP_SPEC:-moonclip>=0.1.0}"

# vast.ai keeps torch in a venv that only an interactive shell gets on its
# PATH. Every phase here arrives as `ssh host '...'` — not a login shell — so
# the plain `python` would be /usr/bin/python3, which has no torch.
VENV="${VENV:-/venv/main}"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"


# Not `--upgrade setuptools`: torch 2.11 pins setuptools<82, and upgrading
# it here breaks that quietly. Only fill in what is missing.
pip install --quiet --upgrade pip
python -c "import setuptools, wheel" 2>/dev/null \
    || pip install --quiet "setuptools<82" wheel
# The image's torch may not have kernels for the GPU the box actually has.
# Found on RunPod 2026-08-21: an RTX PRO 4500 Blackwell (sm_120) with a torch
# built up to sm_90. `torch.cuda.is_available()` answers True — the failure
# arrives later, as `no kernel image is available for execution on the device`,
# on the first real operation and therefore three phases in.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
arch_ok() {
    python - <<'PY'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(0)                      # no GPU here: not this check's business
    # **A real operation, not the arch list** (2026-09-15). The list names the
    # architectures the kernels were built for, and a GPU runs kernels built
    # for a lower minor of its own major: an RTX 4090 is sm_89, is in no list,
    # and runs the sm_86 kernels. Asking the list failed it, reinstalled torch,
    # and failed it again. A kernel that is really missing fails here, on the
    # op — which is the failure the comment above is about.
    x = torch.randn(64, 64, device="cuda")
    float((x @ x).sum())
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# **Too old is its own failure** (2026-09-21). The RunPod images still ship
# torch 2.4.1, whose kernels run fine on a 4090 - so the check above passes -
# and whose `torch.distributed.fsdp` has no `fully_shard` yet (2.6). The
# training scripts import it, so every phase failed at its first line with an
# ImportError, after setup had said everything was ready. The version asked for
# is PyPI's, which was 7x faster than the pytorch.org index from a European
# box, and whose default wheel is cu128 - the newest CUDA a 570 driver runs,
# which is what one of the two pods had that day.
TORCH_SPEC="${TORCH_SPEC:-torch==2.8.0}"
if ! python -c "from torch.distributed.fsdp import fully_shard" 2>/dev/null; then
    echo "-- torch $(python -c 'import torch; print(torch.__version__)' 2>/dev/null) predates fully_shard: installing $TORCH_SPEC"
    pip install --quiet --root-user-action=ignore "$TORCH_SPEC"
    python -c "from torch.distributed.fsdp import fully_shard" || {
        echo "** still no fully_shard after installing $TORCH_SPEC **" >&2
        exit 1
    }
fi

if ! arch_ok; then
    echo "-- torch has no kernels for this GPU: reinstalling from $TORCH_INDEX"
    pip install -U --quiet --root-user-action=ignore \
        --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple torch
    arch_ok || {
        echo "** still no kernels for this GPU. Try a newer index than" >&2
        echo "   $TORCH_INDEX — nothing below will mean anything. **" >&2
        exit 1
    }
fi

pip install --quiet PyYAML "$MOONCLIP_SPEC"

# **Ravex has a Rust core since GPU-105**, so the editable install below builds
# it with maturin and wants `cargo`. RunPod's images have a C compiler and no
# Rust (both pods on 2026-09-15), and the failure arrives after the torch
# reinstall above, a few minutes in. Linked into /usr/local/bin because every
# phase is `ssh host '...'`, which never puts ~/.cargo/bin on the PATH.
if ! command -v cargo >/dev/null; then
    echo "-- no Rust toolchain: installing one for ravex's core"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal >/dev/null
    ln -sf "$HOME/.cargo/bin/cargo" "$HOME/.cargo/bin/rustc" \
        "$HOME/.cargo/bin/rustup" /usr/local/bin/
fi
pip install --quiet --no-deps -e $KIT_ROOT/ravex

# Checks the install rather than arming anything: the training script attaches
# through its own decorator. The
# same discipline the rest of integration/ runs under, and the thing a user
# actually installs.
ravex status
ravex status

python - <<'PY'
import moonclip, ravex, torch
print("ravex", ravex.__version__, "| moonclip", moonclip.__version__,
      "| torch", torch.__version__, "cuda", torch.version.cuda)
PY
