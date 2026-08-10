#!/usr/bin/env bash
# Everything the rented GPU box has to do, in one command.
#
#   bash integration/run_gpu_tests.sh
#
# Run it from the repository root on a machine that already has torch with
# CUDA — a vast.ai PyTorch template does. The point is that the meter runs for
# a couple of minutes, not while someone works out what to type.

set -euo pipefail

echo "=== environment ==="
python - <<'PY'
import torch
print("torch     ", torch.__version__)
print("cuda       ", torch.version.cuda)
print("devices    ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

echo
echo "=== install ==="
pip install -q -e . pytest

# Writes the one-line .pth into site-packages, so the training scripts below
# stay free of any reference to Ravex.
ravex enable

echo
echo "=== unit suite (fast, catches a broken install) ==="
pytest tests -q

echo
echo "=== CPU integration suite ==="
pytest integration -q --ignore=integration/test_cuda.py

echo
echo "=== GPU suite ==="
pytest integration/test_cuda.py -v --tb=short

echo
echo "All done. Destroy the instance — stopping it still bills storage."
