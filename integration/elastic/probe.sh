#!/usr/bin/env bash
# GPU-94: does torchrun's own elastic agent, plus Ravex's ordinary resume,
# already deliver an elastic shrink with no new Ravex code?
#
# N agents on one machine, joined to the same c10d rendezvous with
# `--nnodes=1:N`. That is a real elastic job: the agents negotiate membership
# between themselves, and killing one is a membership change of exactly the
# kind a preempted node produces. One box is enough *for this question* —
# what is under test is the launcher's restart and Ravex's resume, not
# anything about the network. Every agent still writes its own `rank_<n>`
# store, so the layout is the one a multi-machine job produces; they share the
# directory those stores sit in, which is the shared-storage configuration
# GPU-96 measured its way to on the same day.
#
#   AGENTS=3 bash probe.sh      # 3 -> 2, the ordinary shrink
#   AGENTS=2 bash probe.sh      # 2 -> 1, the far end
#
# What counts as success, decided before running it:
#
#   1. every agent trains at world_size N, and per-rank checkpoints land
#   2. the last agent is killed
#   3. the survivors' workers are restarted by their own agents at N-1
#   4. Ravex resumes them from the N-rank checkpoint through the reshard
#      path, rather than starting from scratch
#
# Step 4 is the one that matters. Steps 1-3 are torchrun's; if 4 works,
# GPU-94 is a documentation issue rather than a feature.
set -uo pipefail

WORK=/tmp/gpu94
AGENTS="${AGENTS:-3}"
STEPS="${STEPS:-4000}"
PORT="${PORT:-29500}"
LAST=$((AGENTS - 1))

rm -rf "$WORK"

for i in $(seq 0 "$LAST"); do
    mkdir -p "$WORK/a$i"
    # `reshard_on_resume` is off by default *by design*: a launcher that
    # starts the wrong number of ranks should fail visibly rather than train
    # on. An elastic job is precisely the case where the world changing is
    # intended, so it is the one place the flag belongs.
    # One store directory for every agent, not one each. That is not a
    # convenience: with a directory per agent this probe reproduces GPU-96's
    # refusal exactly — "no store and no complete copy here for rank(s) 1, 2"
    # — because a survivor can only see the shard it wrote itself. GPU-96 was
    # closed on 2026-08-31 having measured that shared or remote storage is
    # the answer to that (25x faster than moving shards between machines), so
    # this is that answer applied, and the two issues agree rather than each
    # waiting on the other.
    cat > "$WORK/a$i/ravex.yaml" <<YAML
checkpoint_every: 5
backend: torch_save
keep_last: 3
sharded_checkpoints: per_rank
reshard_on_resume: true
storage:
  type: local
  path: $WORK/store
log_file: ./ravex.log
log_level: INFO
YAML
    cp /app/integration/elastic/elastic_shrink.py "$WORK/a$i/"
done

launch() {  # working directory
    local d="$1"
    cd "$WORK/$d"
    # `setsid` so the agent and the workers it spawns share one process group
    # of their own, and the kill below can take the whole thing.
    STEPS="$STEPS" TRACE="$WORK/$d/trace.jsonl" \
    GLOO_TIMEOUT="${GLOO_TIMEOUT:-30}" \
    setsid torchrun \
        --nnodes="1:$AGENTS" --nproc-per-node=1 \
        --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$PORT" \
        --rdzv-id=gpu94 --max-restarts=3 \
        elastic_shrink.py > "$WORK/$d/agent.out" 2>&1 &
    echo $!
}

# Everything whose working directory is under `$1`, found through /proc.
#
# Not `pkill`: this image is python:3.12-slim and has no procps, so `pgrep`
# and `pkill` are simply absent — and the shape that hides it is
# `$(pgrep -c ... || echo 0)`, which turns "command not found" into a
# confident `0 survivors`. That is how a first run of this probe reported a
# clean kill while both of the victim's processes were still training. /proc
# is always there and cannot be mistaken for an answer it did not give.
in_dir() {
    local marker="$1" found=""
    for p in /proc/[0-9]*; do
        case "$(readlink "$p/cwd" 2>/dev/null)" in
            "$marker"|"$marker"/*) found="$found ${p#/proc/}" ;;
        esac
    done
    echo $found
}

echo "-- $AGENTS agents up, expecting world_size $AGENTS --"
for i in $(seq 0 "$LAST"); do
    eval "PID$i=\$(launch a$i)"
    sleep 1
done

# Wait for a checkpoint, not for a duration: a sleep long enough on a fast
# machine is short enough on a slow one, and what is being waited for is an
# event.
# A *checkpoint*, not just any entry. `ls -A` was the first attempt and it is
# wrong in a way that looks right: Ravex writes a `.ravex-owner` record into
# the store as soon as it opens it, so the directory becomes non-empty long
# before a checkpoint exists, and the probe went on to kill an agent at step 2
# and then report that the stores "hold no checkpoint" - which was true, and
# was the probe's own doing.
for _ in $(seq 120); do
    ls "$WORK"/store/rank_0/step_* >/dev/null 2>&1 && break
    ls "$WORK"/store/rank_0/snapshots/* >/dev/null 2>&1 && break
    sleep 1
done
echo "  agent 0 trace: $(tail -1 "$WORK/a0/trace.jsonl" 2>/dev/null)"
echo "  store: $(ls "$WORK/store/rank_0" 2>/dev/null | tail -2 | tr '\n' ' ')"

echo
echo "-- killing agent $LAST and everything under it --"
victims="$(in_dir "$WORK/a$LAST")"
echo "  processes with cwd in a$LAST: ${victims:-none}"
for pid in $victims; do kill -9 "$pid" 2>/dev/null; done
sleep 2
echo "  survivors: $(in_dir "$WORK/a$LAST" | tr -s ' ')"

echo
echo "-- waiting for agent 0 to come back at world_size $((AGENTS - 1)) --"
eval "PID0V=\$PID0"
for i in $(seq "${WAIT:-180}"); do
    grep -q "\"world\": $((AGENTS - 1))" "$WORK/a0/trace.jsonl" 2>/dev/null \
        && { echo "  restarted after ${i}s"; break; }
    kill -0 "$PID0V" 2>/dev/null || { echo "  agent 0 exited after ${i}s"; break; }
    sleep 1
done

echo
echo "=== what agent 0 saw ==="
python - <<'PY'
import json
seen = []
for line in open("/tmp/gpu94/a0/trace.jsonl", encoding="utf-8"):
    e = json.loads(line)
    key = (e["world"], e["pid"])
    if not seen or seen[-1][0] != key:
        seen.append((key, e["step"]))
for (world, pid), step in seen:
    print("  world_size %d, pid %d, first step recorded %d" % (world, pid, step))
PY

echo
echo "=== what Ravex decided ==="
grep -aiE "resum|reshard|scratch|Restored|world" "$WORK/a0/ravex.log" | tail -15

echo
echo "=== agent 0, still running? ==="
echo "  processes under a0: $(in_dir "$WORK/a0" | tr -s ' ')"
echo "  store now: $(ls "$WORK"/store/rank_* -d 2>/dev/null | tr '
' ' ')"
for d in "$WORK"/store/rank_*; do
    echo "    ${d##*/}: $(ls "$d" 2>/dev/null | grep -c step_ ) checkpoints"
done

echo
echo "=== agent 0, last lines ==="
tail -25 "$WORK/a0/agent.out"

for i in $(seq 0 "$LAST"); do
    for pid in $(in_dir "$WORK/a$i"); do kill -9 "$pid" 2>/dev/null; done
done
wait 2>/dev/null
