#!/usr/bin/env bash
# GPU-90: a plain PyTorch script resumes from a DeepSpeed checkpoint, with no
# code of its own to make that happen.
#
#   bash check_resume_conversion.sh [stage] [world] [hidden]
#
# Everything the earlier checks prove happens inside a function somebody calls.
# This one proves the thing the feature is actually for: a `ravex.yaml`, a
# storage path pointing at a DeepSpeed checkpoint, and a training script with
# not one reference to Ravex or DeepSpeed in it. Ravex arrives through the
# autoloader, sees a foreign checkpoint where its own store would be, converts
# it, and the script's model comes up holding DeepSpeed's weights.
#
# `checkpoint_every` is set absurdly high on purpose. Ravex writing its own
# store into the same directory it just converted *from* would leave two
# layouts in one place, and this check would then be measuring that instead.
# It is a real wrinkle of pointing `storage.path` at a foreign checkpoint, and
# it is worth knowing rather than tripping over.
set -uo pipefail

STAGE="${1:-1}"
WORLD="${2:-2}"
HIDDEN="${3:-97}"
LAYERS=3
ROOT=/tmp/convert_resume
KIT=/app/integration/frameworks

rm -rf "$ROOT"; mkdir -p "$ROOT/run"

echo "-- a DeepSpeed ZeRO stage $STAGE checkpoint over $WORLD ranks --"
torchrun --nproc-per-node="$WORLD" --master-port=38001 \
    "$KIT/make_zero.py" --stage "$STAGE" --hidden "$HIDDEN" --layers "$LAYERS" \
    --out "$ROOT/zero" >/dev/null 2>&1 \
    || { echo "  ** could not produce the checkpoint **"; exit 1; }
ls "$ROOT/zero" | sed 's/^/     /'

cat > "$ROOT/run/ravex.yaml" <<YAML
checkpoint_every: 1000000
backend: torch_save
convert_foreign: true
storage:
  type: local
  path: $ROOT/zero
log_file: ./ravex.log
log_level: INFO
YAML

echo
echo "-- a training script that imports neither Ravex nor DeepSpeed --"
# Imports, not mentions: the script's own docstring says it has never heard of
# either, and a `grep -c` for the words counts that and reports the opposite of
# what it means.
grep -cE "^ *(import|from) +(ravex|deepspeed)" "$KIT/train_plain.py"     | sed 's/^/     imports of either: /'

cd "$ROOT/run"
python "$KIT/train_plain.py" --hidden "$HIDDEN" --layers "$LAYERS" \
    --out "$ROOT/after.pt" 2>&1 | tail -3

echo
echo "-- what Ravex decided --"
grep -aE "written by something else|Converting from|Conversion note|Resumed" ravex.log \
    | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' | sed 's/^/     /'

echo
echo "-- did the model come up holding DeepSpeed's weights? --"
python - <<PY
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

after = torch.load("$ROOT/after.pt", map_location="cpu", weights_only=False)
oracle = get_fp32_state_dict_from_zero_checkpoint("$ROOT/zero")

missing = [k for k in oracle if k not in after]
if missing:
    print("     ** not in the live model: %s **" % missing[:5]); raise SystemExit(1)

bad = [k for k, v in oracle.items() if not torch.equal(after[k], v)]
if bad:
    for k in bad[:5]:
        print("     %s differs, largest gap %g"
              % (k, (after[k].float() - oracle[k].float()).abs().max().item()))
    print("     ** %d of %d parameters do not match **" % (len(bad), len(oracle)))
    raise SystemExit(1)
print("     all %d parameters match zero_to_fp32" % len(oracle))
PY
