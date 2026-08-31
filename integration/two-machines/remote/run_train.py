"""Run the repository's CUDA training script with the timing patch installed.

    torchrun ... $KIT_ROOT/kit/run_train.py -- --params 4e8 --steps 12

Everything after ``--`` goes to integration/scripts/train_fsdp_cuda.py
unchanged. The patch has to be installed before the first checkpoint, and this
is the least invasive place to do it: the training script keeps knowing nothing
about Ravex, which is the property the whole integration suite is built on.
"""

import os
import runpy
import sys

# Where the kit landed, asked rather than assumed. /root suits vast.ai, where
# the container disk is the disk; RunPod needs /workspace, because its
# container disk is small and filling it is the silent failure. Hardcoding
# /root here made this file the one place the KIT_ROOT the rest of the kit
# already threads through could not reach - and the symptom is a
# ModuleNotFoundError on a box that is being billed by the second.
KIT_ROOT = os.environ.get("KIT_ROOT", "/root")

sys.path.insert(0, os.path.join(KIT_ROOT, "kit"))

import ravex_timing

ravex_timing.install()

SCRIPT = os.path.join(KIT_ROOT, "ravex/integration/scripts/train_fsdp_cuda.py")

argv = sys.argv[1:]
if argv and argv[0] == "--":
    argv = argv[1:]
sys.argv = [SCRIPT] + argv
runpy.run_path(SCRIPT, run_name="__main__")
