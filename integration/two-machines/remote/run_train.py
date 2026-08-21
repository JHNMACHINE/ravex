"""Run the repository's CUDA training script with the timing patch installed.

    torchrun ... /root/kit/run_train.py -- --params 4e8 --steps 12

Everything after ``--`` goes to integration/scripts/train_fsdp_cuda.py
unchanged. The patch has to be installed before the first checkpoint, and this
is the least invasive place to do it: the training script keeps knowing nothing
about Ravex, which is the property the whole integration suite is built on.
"""

import runpy
import sys

sys.path.insert(0, "/root/kit")

import ravex_timing

ravex_timing.install()

SCRIPT = "/root/ravex/integration/scripts/train_fsdp_cuda.py"

argv = sys.argv[1:]
if argv and argv[0] == "--":
    argv = argv[1:]
sys.argv = [SCRIPT] + argv
runpy.run_path(SCRIPT, run_name="__main__")
