# The multi-machine bench

One container per rank, so that "which ranks can see which directory" has an
answer. Ranks on one machine all see the same one, which is why
`torchrun --nproc_per_node=6` cannot ask this question at all: it makes six
processes, and this bench needs six *machines*.

```bash
bash integration/multinode/run.sh shared
bash integration/multinode/run.sh split
bash integration/multinode/run.sh node-loss
bash integration/multinode/run.sh reshuffle
```

Run from the repository root. The image builds on first use. Every scenario
runs twice: once until the ranks kill themselves mid-training the way a
preempted box goes, then again with the same command — which is the whole
proposition.

## What each scenario asks

| Scenario | The disks | The question |
|---|---|---|
| `shared` | one volume, all six ranks | does a shared filesystem get recognised, and does everyone resume from it |
| `split` | one volume per rank | is a split correctly detected rather than assumed, and are copies made |
| `node-loss` | one per rank, node3's wiped between rounds | does a replaced machine get its store back |
| `reshuffle` | one per rank, permuted between rounds | what happens when the nodes come back in a different order |

## What they said, 2026-08-20

`shared` — the announcement, and six ranks resuming together:

```
Checkpoint storage (/checkpoints) is shared across all 6 machines -
per-rank checkpoints resume no matter which machine gets which ranks.
      6 Resumed at step 20
```

`node-loss` — node3 arrives with an empty disk and is given its store back by
the peer that had been holding a copy. This is the first end-to-end proof of
the peer replication:

```
This rank had no store of its own and took one back from rank 4,
where a copy had been kept.
Resumed at step 20
```

`reshuffle` — every rank rebuilds itself from the copy that came back on its
own disk, without asking anyone:

```
6 This rank had no store of its own and rebuilt one from the copy that came
  back on this machine's disk - nothing had to be fetched.
6 Resumed at step 20
```

**This is the scenario that earned the bench its keep.** On its first honest
run it printed something else — *"A checkpoint exists but this topology cannot
reach it - starting from scratch"* — while the layout dump above it showed
every rank sitting on the disk holding its own copy. Six intact copies, none
used, because the recovery addressed peers by rank while the copies had
travelled with the disks. That was GPU-79, fixed on 2026-08-21; this scenario
is what would catch it coming back.

## Why it is built the way it is

**Docker volumes, not bind mounts.** A bind mount on Docker Desktop crosses a
9p/virtiofs boundary with filesystem semantics of its own, and this bench is
entirely about filesystem semantics.

**`--init` is load-bearing.** A round ends with the training script sending
itself SIGKILL. The kernel discards a SIGKILL aimed at PID 1 from inside its
own namespace when PID 1 has no handler, so without an init process the script
runs to completion, the bench reports six happy nodes, and it has proved
nothing. The rest of `integration/` never meets this: there `torchrun` is PID 1
and the ranks are its children.

**`RAVEX_LOG_FILE` is set.** Left unset, Ravex sends WARNING and above to
stderr and drops the rest — including the topology announcement, which is the
single line most of these scenarios exist to produce.

**The layout is printed, not assumed.** Every interesting scenario turns on
which bytes are on which disk, and a bench that reports only the outcome cannot
tell "the data was gone" from "the data was there and nobody looked". That
distinction is exactly what the reshuffle run turned out to hinge on.

**moonclip comes from PyPI**, so this exercises the working tree's Ravex
against the *released* engine — which is what a user gets, and it keeps a Rust
toolchain out of the image. It matters for anything touching remote storage,
which reached the released engine in 0.0.8 — before that the remote was
push-only and there was nothing to pull back.

## What it cannot do

**It is not a network filesystem.** Every scenario runs on local disk, shared
or not. NFS attribute caching — a rank acting on a directory listing that is up
to a minute stale — is the one behaviour it cannot reproduce, and it stays
open. A real NFS needs `nfs`/`nfsd` kernel modules, and the Docker Desktop VM
has no `/lib/modules` to load them from; on this kernel both are `=m`. It wants
an ordinary Linux host.

**It is not a measurement.** Six containers on one machine share a disk and a
CPU, so nothing here says what replication costs in bandwidth or in time. See
the closing note on GPU-76 for what still needs two real boxes.

## Knobs

| Variable | Default | |
|---|---|---|
| `NODES` | 6 | ranks, one container each |
| `DIE_AT` | 25 | step at which round one is cut short |
| `REPLICATE_EVERY` | 2 | checkpoints between peer copies; `0` turns them off |
| `WAIT_LIMIT` | 180 | seconds before a container that will not exit is stopped |
| `RESULTS` | `/tmp/ravex-bench` | best-effort host copy of the logs |

Logs live in the `ravex-bench-results` volume, one per rank per round:

```bash
docker run --rm -v ravex-bench-results:/out alpine cat /out/ravex.resume.rank3.log
```
