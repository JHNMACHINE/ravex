#!/usr/bin/env bash
# Answer, on a real multi-GPU box, the one question the instrumentation was added
# for: where the 10.6 s of a checkpoint handoff actually goes.
#
# Run it *on the box*:
#
#   curl -sSLo measure_handoff.sh \
#     https://codeberg.org/JHNMACHINE/ravex/raw/branch/main/integration/scripts/measure_handoff.sh
#   bash measure_handoff.sh 2>&1 | tee /root/measure.log
#
# Nothing is copied from a laptop and nothing is compiled: both libraries come
# from PyPI as wheels, so this also exercises exactly what a user installs.
# `vast_setup.sh` predates that and still builds Moonclip with maturin — it is
# only needed for measuring an unreleased change.
#
# Four runs, and the third one may well contradict us. Each writes its own
# ravex.log; the phase breakdown is the payload.
set -euo pipefail

RUNS=${RUNS:-/root/handoff}
PARAMS=${PARAMS:-1.5e9}
STEPS=${STEPS:-7}
EVERY=${EVERY:-2}

# Phases skipped for want of disk. Collected rather than fatal: a skip is not a
# crash, and on a rented box the space can go away mid-sweep — see below.
SKIPPED=()

section() { printf '\n\033[1m── %s ─────────────────────────────────\033[0m\n' "$1"; }

section "the box"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
    echo "no nvidia-smi: this is not the box we wanted" >&2; exit 1; }
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo
echo "GPUs: $N   cores: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GiB"

section "install from PyPI"
pip install --quiet --upgrade "ravex[moonclip]"
python - <<'EOF'
import moonclip, ravex, torch
print("ravex", ravex.__version__, "| moonclip", moonclip.__version__,
      "| torch", torch.__version__, "cuda", torch.version.cuda)
EOF
# Only the driving script comes from git: the packages are the published ones.
[ -d /root/ravex-src ] || git clone --quiet --depth 1 \
    https://codeberg.org/JHNMACHINE/ravex.git /root/ravex-src
SCRIPT=/root/ravex-src/integration/scripts/train_fsdp_cuda.py

# One workspace per run: Ravex finds its config by walking up from the working
# directory, and a fresh store means every run pays for a full first checkpoint.
workspace() {
    local name="$1" every="$2"
    local dir="$RUNS/$name"
    rm -rf "$dir"; mkdir -p "$dir"
    cat > "$dir/ravex.yaml" <<EOF
checkpoint_every: $every
backend: moonclip
keep_last: 3
sharded_checkpoints: per_rank
storage:
  type: local
  path: ./checkpoints
log_file: ./ravex.log
log_level: INFO
EOF
    echo "$dir"
}

# The phase breakdown is the whole point of the exercise; print it and nothing
# else, so a wall of NCCL chatter cannot bury it.
phases() {
    local dir="$1"
    echo
    grep -h "handed off" "$dir"/ravex.log | sed 's/^.*INFO //' || echo "(no checkpoint lines — look at $dir/ravex.log)"
    grep -h "waited .* for the previous one" "$dir"/ravex.log "$dir"/stderr.txt 2>/dev/null || true
}

# Free GiB on the filesystem the runs write to.
free_gib() { df -PBG "$RUNS" | awk 'NR==2 {gsub(/G/,"",$4); print $4}'; }

# Below this a phase cannot write a full set of shards and would be measuring
# ENOSPC instead of the handoff.
NEED_GIB=${NEED_GIB:-40}

# Each phase writes N shards of the whole model and keeps `keep_last` of them,
# which on 8 ranks at 1.5B is tens of GiB — enough that four phases in a row
# fill an ordinary box disk. The first phase to hit ENOSPC does not stop: Ravex
# reports the failed checkpoint, disables itself and lets training continue, so
# the phase still prints timings — for the checkpoints that happened, on a full
# filesystem. That is a measurement of the disk wearing the costume of a
# measurement of the handoff, and it is not obvious from the numbers alone.
#
# So each phase drops its own checkpoints as soon as its numbers have been
# read, and refuses to start if the space is not there. The logs stay: they are
# kilobytes, and they are the payload.
run() {
    local name="$1"; shift
    local dir
    dir=$(workspace "$name" "$EVERY")
    local before; before=$(free_gib)
    if [ "$before" -lt "$NEED_GIB" ]; then
        echo "  SKIPPED $name: ${before} GiB free, needs $NEED_GIB." >&2
        echo "  Free space or lower --params; raising NEED_GIB only hides it." >&2
        SKIPPED+=("$name")
        # 0, not 1: `run` is called as a bare top-level command under `set -e`,
        # so a non-zero return here does not skip one phase — it ends the
        # script, and the phases after it never run. The sweep is reported as
        # having crashed when it merely ran short of disk. The skip is already
        # on stderr and it is counted; the exit status is settled at the end.
        return 0
    fi
    echo "→ $name  ($*)   [${before} GiB free]"
    ( cd "$dir" && env "$@" torchrun --nproc_per_node="$N" --master_port=29566         "$SCRIPT" --params "$PARAMS" --api fsdp2 --steps "$STEPS"         >stdout.txt 2>stderr.txt ) || {
            echo "  FAILED — tail of stderr:"; tail -20 "$dir/stderr.txt"; }
    # ENOSPC never fails the run, so the log has to be asked directly — before
    # anybody reads the timings as an answer.
    if grep -qs "No space left on device" "$dir"/ravex.log "$dir"/stderr.txt; then
        echo "  DISK FULL during $name — these timings measure the filesystem." >&2
    fi
    phases "$dir"
    rm -rf "$dir/checkpoints"
    echo "  (freed $name's checkpoints — $(free_gib) GiB free)"
}

section "1. baseline: per-rank, defaults"
run baseline IGNORE=1

section "2. same, with Moonclip's profiler"
# Splits `store` into the shadow copy and the wait for the previous writer.
run profile MOONCLIP_PROFILE=1

section "3. the whole machine to Moonclip"
# The pool defaults to cores/LOCAL_WORLD_SIZE. Capping the threads measured
# *slower* than not capping (11.6 s against 10.6 s), so this run is where the
# private pool either costs something real or turns out not to matter.
run all-threads "MOONCLIP_THREADS=$(nproc)"

section "4. one thread each"
# The other end of the range, to see whether the handoff moves with threads at
# all — three earlier A/Bs said it does not.
run one-thread MOONCLIP_THREADS=1

section "the launcher does not hold a runtime"
# `torchrun` imports torch to parse its own arguments, so it used to come
# through the autoloader too: N ranks announced N+1 runtimes. Expect exactly N.
found=$(grep -hc "Ravex active" "$RUNS/baseline"/ravex.log || echo 0)
echo "Ravex active lines: $found (expected $N)"
[ "$found" = "$N" ] && echo "OK" || echo "MISMATCH — the launcher guard needs a look"

section "regression: the CUDA suite"
pip install --quiet pytest
( cd /root/ravex-src && pytest integration/test_cuda.py -q ) || \
    echo "the CUDA suite is not green — that matters more than the numbers above"

section "done"
echo "logs under $RUNS/*/ravex.log"

# Reported here, and reflected in the exit status, so a sweep with a hole in it
# is not mistaken for a complete one — but only after every phase that *could*
# run has run.
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "incomplete: ${#SKIPPED[@]} phase(s) skipped for disk — ${SKIPPED[*]}" >&2
    exit 1
fi
