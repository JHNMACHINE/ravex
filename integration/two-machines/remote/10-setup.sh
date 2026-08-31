#!/usr/bin/env bash
# Install what the phases need. A couple of minutes, no Rust.
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

MOONCLIP_SPEC="${MOONCLIP_SPEC:-moonclip==0.0.9}"

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
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    sys.exit(0 if cap in torch.cuda.get_arch_list() else 1)
except Exception:
    sys.exit(1)
PY
}

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
pip install --quiet --no-deps -e $KIT_ROOT/ravex

# The autoloader, so the training script contains no reference to Ravex — the
# same discipline the rest of integration/ runs under, and the thing a user
# actually installs.
ravex enable
ravex status

python - <<'PY'
import moonclip, ravex, torch
print("ravex", ravex.__version__, "| moonclip", moonclip.__version__,
      "| torch", torch.__version__, "cuda", torch.version.cuda)
PY
